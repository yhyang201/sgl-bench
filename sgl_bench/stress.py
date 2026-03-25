"""Stress/soak test for VLM servers.

Continuously sends randomized image requests to detect OOM, hangs,
or performance degradation over time.

Key design: the pool stores "recipes" (text prompt + image specs + token counts).
Fresh random images are generated per-request via ThreadPoolExecutor to avoid
hitting sglang's RadixAttention cache. Image generation (numpy/PIL/base64) is
C code that releases the GIL, so threads work well.
"""

import asyncio
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import aiohttp
import numpy as np

from .runner import (
    DatasetRow,
    RequestOutput,
    _create_client_session,
    _gen_random_image,
    _gen_random_text,
    _send_chat_request,
    parse_image_resolution,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoolRecipe:
    """A recipe for generating a stress request. Images are regenerated each time."""
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: int
    vision_prompt_len: int
    image_specs: List[Tuple[int, int]]  # [(width, height), ...] per image


@dataclass
class StressWindow:
    window_index: int
    start_time: str
    end_time: str
    duration_s: float
    completed: int
    failed: int
    request_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    health_ok: bool

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class StressReport:
    total_duration_s: float
    total_completed: int
    total_failed: int
    windows: List[StressWindow]
    abort_reason: Optional[str]
    pool_info: dict
    overall_request_throughput: float
    overall_output_throughput: float

    def to_dict(self) -> dict:
        return {
            "total_duration_s": self.total_duration_s,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "abort_reason": self.abort_reason,
            "pool_info": self.pool_info,
            "overall_request_throughput": self.overall_request_throughput,
            "overall_output_throughput": self.overall_output_throughput,
            "windows": [w.to_dict() for w in self.windows],
        }


# ---------------------------------------------------------------------------
# Pool generation (recipes only, no image data stored)
# ---------------------------------------------------------------------------

def _generate_recipe_pool(
    pool_size: int,
    image_count_range: Tuple[int, int],
    image_resolutions: List[str],
    input_len_range: Tuple[int, int],
    output_len_range: Tuple[int, int],
    model_id: str,
) -> List[PoolRecipe]:
    """Generate a pool of recipes. Each recipe stores text + image specs + token counts.

    Images are NOT stored — they are regenerated fresh per request.
    Only one sample image set is created per recipe for token counting, then discarded.
    """
    from transformers import AutoProcessor

    print(f"Generating recipe pool ({pool_size} entries)...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image_pad_id = getattr(processor, "image_token_id", None)

    pool: List[PoolRecipe] = []

    for i in range(pool_size):
        n_images = random.randint(*image_count_range)
        input_len = random.randint(*input_len_range)
        output_len = random.randint(*output_len_range)

        # Generate sample images for token counting (discarded after)
        image_specs = []
        images = []
        images_base64 = []
        for _ in range(n_images):
            res = random.choice(image_resolutions)
            w, h = parse_image_resolution(res)
            image_specs.append((w, h))
            img, data_uri = _gen_random_image(w, h)
            images.append(img)
            images_base64.append(data_uri)

        # Generate text prompt
        text_prompt = _gen_random_text(processor.tokenizer, input_len, image_pad_id)

        # Compute token counts
        content_items = [
            {"type": "image", "image": {"url": uri}}
            for uri in images_base64
        ]
        content_items.append({"type": "text", "text": text_prompt})

        try:
            prompt_str = processor.apply_chat_template(
                [{"role": "user", "content": content_items}],
                add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            prompt_str = f"<image>{text_prompt}"

        total_len = processor(
            text=[prompt_str], images=images, padding=False, return_tensors="pt"
        )["input_ids"].numel()

        try:
            text_only = processor.apply_chat_template(
                [{"role": "user", "content": text_prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            text_len = processor(
                text=[text_only], padding=False, return_tensors="pt"
            )["input_ids"].numel()
        except Exception:
            text_len = len(processor.tokenizer.encode(text_prompt))

        vision_len = total_len - text_len

        pool.append(PoolRecipe(
            prompt=text_prompt,
            prompt_len=total_len,
            output_len=output_len,
            text_prompt_len=text_len,
            vision_prompt_len=vision_len,
            image_specs=image_specs,
        ))

        # images/images_base64 go out of scope here — memory freed
        res_names = [f"{w}x{h}" for w, h in image_specs]
        print(f"  [{i+1}/{pool_size}] imgs={n_images} res={res_names} "
              f"text={text_len} vision={vision_len} output={output_len}",
              flush=True)

    print(f"Pool ready: {pool_size} recipes (no images stored, regenerated per request)",
          flush=True)
    return pool


def _make_fresh_row(recipe: PoolRecipe) -> DatasetRow:
    """Generate fresh random images from a recipe. Runs in a thread (GIL-safe)."""
    fresh_base64 = []
    for w, h in recipe.image_specs:
        _, data_uri = _gen_random_image(w, h)
        fresh_base64.append(data_uri)

    return DatasetRow(
        prompt=recipe.prompt,
        prompt_len=recipe.prompt_len,
        output_len=recipe.output_len,
        text_prompt_len=recipe.text_prompt_len,
        vision_prompt_len=recipe.vision_prompt_len,
        image_data=fresh_base64,
    )


def _get_pool_info(
    pool: List[PoolRecipe],
    image_count_range: Tuple[int, int],
    image_resolutions: List[str],
    input_len_range: Tuple[int, int],
    output_len_range: Tuple[int, int],
) -> dict:
    img_counts = [len(r.image_specs) for r in pool]
    return {
        "pool_size": len(pool),
        "image_count_range": list(image_count_range),
        "image_resolutions": image_resolutions,
        "input_len_range": list(input_len_range),
        "output_len_range": list(output_len_range),
        "actual_images_per_req": {"min": min(img_counts), "max": max(img_counts),
                                  "mean": round(float(np.mean(img_counts)), 1)},
        "actual_prompt_len": {"min": min(r.prompt_len for r in pool),
                              "max": max(r.prompt_len for r in pool)},
        "fresh_images_per_request": True,
    }


# ---------------------------------------------------------------------------
# Window metrics
# ---------------------------------------------------------------------------

def _compute_window_metrics(outputs: List[RequestOutput], duration_s: float) -> dict:
    ttfts, itls, e2es = [], [], []
    total_output = 0
    completed = failed = 0

    for out in outputs:
        if out.success:
            completed += 1
            ttfts.append(out.ttft)
            itls.extend(out.itl)
            e2es.append(out.latency)
            total_output += out.output_len
        else:
            failed += 1

    return {
        "completed": completed,
        "failed": failed,
        "request_throughput": completed / duration_s if duration_s > 0 else 0,
        "output_throughput": total_output / duration_s if duration_s > 0 else 0,
        "mean_ttft_ms": float(np.mean(ttfts) * 1000) if ttfts else 0,
        "p99_ttft_ms": float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
        "mean_itl_ms": float(np.mean(itls) * 1000) if itls else 0,
        "p99_itl_ms": float(np.percentile(itls, 99) * 1000) if itls else 0,
        "mean_e2e_latency_ms": float(np.mean(e2es) * 1000) if e2es else 0,
        "p99_e2e_latency_ms": float(np.percentile(e2es, 99) * 1000) if e2es else 0,
    }


# ---------------------------------------------------------------------------
# Async stress loop
# ---------------------------------------------------------------------------

async def _check_health(session: aiohttp.ClientSession, health_url: str) -> bool:
    try:
        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _stress_loop(
    api_url: str,
    model: str,
    pool: List[PoolRecipe],
    duration_s: float,
    request_rate: float,
    max_concurrency: int,
    health_url: str,
    health_interval_s: float,
    window_s: float,
    server_process: Optional[subprocess.Popen],
) -> StressReport:
    """Main async stress loop. Generates fresh images per request via ThreadPoolExecutor."""

    semaphore = asyncio.Semaphore(max_concurrency)
    stop_event = asyncio.Event()
    abort_reason = None
    executor = ThreadPoolExecutor(max_workers=4)

    # Shared state
    lock = asyncio.Lock()
    current_window_outputs: List[RequestOutput] = []
    all_windows: List[StressWindow] = []
    total_completed = 0
    total_failed = 0
    total_output_tokens = 0
    health_ok = True

    start_time = time.monotonic()
    window_start = time.monotonic()
    window_index = 0

    async def send_and_collect(session: aiohttp.ClientSession, recipe: PoolRecipe):
        """Acquire semaphore FIRST (backpressure), then generate images + send."""
        nonlocal total_completed, total_failed, total_output_tokens

        async with semaphore:
            # Generate fresh images in thread pool (releases GIL)
            loop = asyncio.get_event_loop()
            row = await loop.run_in_executor(executor, _make_fresh_row, recipe)
            # Send request
            output = await _send_chat_request(session, api_url, model, row)

        async with lock:
            current_window_outputs.append(output)
            if output.success:
                total_completed += 1
                total_output_tokens += output.output_len
            else:
                total_failed += 1

    async def producer(session: aiohttp.ClientSession):
        """Continuously fire requests until stop. Backpressure via semaphore."""
        pending: set[asyncio.Task] = set()

        while not stop_event.is_set():
            recipe = random.choice(pool)
            task = asyncio.create_task(send_and_collect(session, recipe))
            pending.add(task)
            task.add_done_callback(pending.discard)

            if request_rate != float("inf"):
                interval = np.random.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)
            else:
                # With inf rate, semaphore provides backpressure.
                # Yield to event loop so tasks can make progress.
                await asyncio.sleep(0)

        # Drain in-flight
        if pending:
            await asyncio.wait(pending, timeout=120)

    async def health_monitor(session: aiohttp.ClientSession):
        nonlocal health_ok, abort_reason
        consecutive_failures = 0

        while not stop_event.is_set():
            await asyncio.sleep(health_interval_s)
            if stop_event.is_set():
                break

            if server_process and server_process.poll() is not None:
                abort_reason = f"Server crashed (exit code {server_process.returncode})"
                health_ok = False
                stop_event.set()
                return

            ok = await _check_health(session, health_url)
            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    abort_reason = f"Server unhealthy ({consecutive_failures} consecutive failures)"
                    health_ok = False
                    stop_event.set()
                    return

    async def window_reporter():
        nonlocal current_window_outputs, window_start, window_index

        while not stop_event.is_set():
            await asyncio.sleep(window_s)
            if stop_event.is_set():
                break

            now = time.monotonic()
            elapsed = now - start_time
            window_duration = now - window_start

            async with lock:
                outputs = current_window_outputs
                current_window_outputs = []

            mins = int(elapsed // 60)
            secs = int(elapsed % 60)

            if not outputs:
                print(f"[{mins:02d}:{secs:02d}] Window {window_index}: (no completed requests)",
                      flush=True)
                window_index += 1
                window_start = now
                continue

            metrics = _compute_window_metrics(outputs, window_duration)
            window = StressWindow(
                window_index=window_index,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_s=round(window_duration, 1),
                health_ok=health_ok,
                **metrics,
            )
            all_windows.append(window)

            status = "OK" if health_ok else "UNHEALTHY"
            print(
                f"[{mins:02d}:{secs:02d}] Window {window_index}: "
                f"{metrics['completed']} ok, {metrics['failed']} fail | "
                f"{metrics['request_throughput']:.2f} req/s | "
                f"TTFT: {metrics['mean_ttft_ms']:.0f}ms "
                f"(p99: {metrics['p99_ttft_ms']:.0f}ms) | "
                f"{status}",
                flush=True,
            )

            window_index += 1
            window_start = now

    # --- Main execution ---
    async with _create_client_session() as session:
        producer_task = asyncio.create_task(producer(session))
        health_task = asyncio.create_task(health_monitor(session))
        reporter_task = asyncio.create_task(window_reporter())

        try:
            await asyncio.sleep(duration_s)
        except asyncio.CancelledError:
            abort_reason = "Cancelled by user"
        finally:
            stop_event.set()

        try:
            await asyncio.wait_for(producer_task, timeout=120)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        for task in [health_task, reporter_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    executor.shutdown(wait=False)

    # Final window
    if current_window_outputs:
        final_duration = time.monotonic() - window_start
        metrics = _compute_window_metrics(current_window_outputs, final_duration)
        all_windows.append(StressWindow(
            window_index=window_index,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_s=round(final_duration, 1),
            health_ok=health_ok,
            **metrics,
        ))

    total_duration = time.monotonic() - start_time

    return StressReport(
        total_duration_s=round(total_duration, 1),
        total_completed=total_completed,
        total_failed=total_failed,
        windows=all_windows,
        abort_reason=abort_reason,
        pool_info={},
        overall_request_throughput=total_completed / total_duration if total_duration > 0 else 0,
        overall_output_throughput=total_output_tokens / total_duration if total_duration > 0 else 0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stress_test(
    base_url: str,
    model_id: str,
    stress_cfg: dict,
    server_process: Optional[subprocess.Popen] = None,
    server_backend: str = "sglang",
    output_file: Optional[str] = None,
) -> StressReport:
    """Run a complete stress test."""
    from .server import get_health_url

    duration_s = stress_cfg.get("duration_minutes", 60) * 60
    max_concurrency = stress_cfg.get("max_concurrency", 32)
    pool_size = stress_cfg.get("pool_size", 10)
    image_count_range = tuple(stress_cfg.get("image_count_range", [1, 5]))
    image_resolutions = stress_cfg.get("image_resolutions", ["720p", "1080p"])
    input_len_range = tuple(stress_cfg.get("input_len_range", [256, 4096]))
    output_len_range = tuple(stress_cfg.get("output_len_range", [64, 300]))
    health_interval_s = stress_cfg.get("health_check_interval_s", 30)
    window_s = stress_cfg.get("window_minutes", 5) * 60

    rate_str = stress_cfg.get("request_rate", "inf")
    request_rate = float("inf") if rate_str == "inf" else float(rate_str)

    import re
    port_match = re.search(r":(\d+)$", base_url)
    port = int(port_match.group(1)) if port_match else 30000
    health_url = get_health_url(port, server_backend)

    api_url = f"{base_url}/v1/chat/completions"

    # Generate recipe pool (no images stored)
    pool = _generate_recipe_pool(
        pool_size=pool_size,
        image_count_range=image_count_range,
        image_resolutions=image_resolutions,
        input_len_range=input_len_range,
        output_len_range=output_len_range,
        model_id=model_id,
    )

    pool_info = _get_pool_info(pool, image_count_range, image_resolutions,
                               input_len_range, output_len_range)

    print(f"\nStarting stress test: {duration_s/60:.0f} min, "
          f"concurrency={max_concurrency}, rate={rate_str}", flush=True)
    print(f"Fresh images generated per request (no cache reuse)", flush=True)
    print(f"Health check: {health_url} every {health_interval_s}s", flush=True)
    print(f"Window: {window_s/60:.0f} min\n", flush=True)

    report = asyncio.run(_stress_loop(
        api_url=api_url,
        model=model_id,
        pool=pool,
        duration_s=duration_s,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        health_url=health_url,
        health_interval_s=health_interval_s,
        window_s=window_s,
        server_process=server_process,
    ))

    report.pool_info = pool_info

    _print_summary(report, duration_s)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {output_file}", flush=True)

    return report


def _print_summary(report: StressReport, target_duration_s: float) -> None:
    total = report.total_completed + report.total_failed
    error_rate = report.total_failed / total * 100 if total > 0 else 0

    passed = report.abort_reason is None and error_rate < 5.0
    verdict = "PASSED" if passed else "FAILED"

    print(f"\n{'='*60}", flush=True)
    print(f"Stress Test: {verdict}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Duration: {report.total_duration_s/60:.1f} min "
          f"(target: {target_duration_s/60:.0f} min)", flush=True)
    print(f"Total: {report.total_completed} ok, {report.total_failed} fail "
          f"({error_rate:.2f}% error rate)", flush=True)
    print(f"Throughput: {report.overall_request_throughput:.2f} req/s, "
          f"{report.overall_output_throughput:.1f} tok/s", flush=True)

    if report.abort_reason:
        print(f"Abort reason: {report.abort_reason}", flush=True)

    if len(report.windows) >= 2:
        ttfts = [w.mean_ttft_ms for w in report.windows if w.completed > 0]
        if ttfts:
            first, last = ttfts[0], ttfts[-1]
            change = (last - first) / first * 100 if first > 0 else 0
            trend = " -> ".join(f"{t:.0f}" for t in ttfts[:5])
            if len(ttfts) > 5:
                trend += f" -> ... -> {ttfts[-1]:.0f}"
            print(f"TTFT trend (ms): {trend} ({change:+.1f}%)", flush=True)

    print(f"{'='*60}", flush=True)
