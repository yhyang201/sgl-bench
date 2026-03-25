"""Image limit probe: find the max number of images a server can handle per request.

Sends single requests with increasing image counts (1, 2, 3, ...) for a given
resolution until the server crashes or the request fails.

The server is restarted between resolutions (handled by cli.py).
"""

import asyncio
import json
import time
from typing import List, Optional, Tuple

import aiohttp

from .runner import (
    DatasetRow,
    RequestOutput,
    _gen_random_image,
    _gen_random_text,
    _send_chat_request,
    parse_image_resolution,
)


def _build_row(
    n_images: int,
    width: int,
    height: int,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> DatasetRow:
    """Generate a DatasetRow with n_images fresh random images."""
    images_base64 = []
    for _ in range(n_images):
        _, data_uri = _gen_random_image(width, height)
        images_base64.append(data_uri)

    return DatasetRow(
        prompt=prompt,
        prompt_len=prompt_len,
        output_len=output_len,
        image_data=images_base64,
    )


async def _send_one(api_url: str, model: str, row: DatasetRow, timeout_s: int = 300) -> RequestOutput:
    """Send a single request with custom timeout."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=10 * 1024**2) as session:
        return await _send_chat_request(session, api_url, model, row)


async def _check_health(backend: str, port: int) -> bool:
    from .server import get_health_url
    health_url = get_health_url(port, backend)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(health_url) as resp:
                return resp.status == 200
    except Exception:
        return False


def run_probe_single(
    base_url: str,
    model_id: str,
    resolution: str,
    server_process=None,
    server_backend: str = "sglang",
    port: int = 30000,
    max_images: int = 500,
    input_len: int = 256,
    output_len: int = 32,
    timeout_s: int = 300,
    prompt: str = None,
    processor=None,
) -> dict:
    """Probe max image count for a single resolution.

    Returns dict with resolution, max_images_ok, failed_at, and per-step details.
    """
    api_url = f"{base_url}/v1/chat/completions"

    # Generate text prompt if not provided
    if prompt is None:
        from transformers import AutoProcessor
        print(f"Loading processor...", flush=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        image_pad_id = getattr(processor, "image_token_id", None)
        prompt = _gen_random_text(processor.tokenizer, input_len, image_pad_id)

    w, h = parse_image_resolution(resolution)
    print(f"\n{'='*60}", flush=True)
    print(f"Probing: {resolution} ({w}x{h})", flush=True)
    print(f"{'='*60}", flush=True)

    max_ok = 0
    steps = []

    for n in range(1, max_images + 1):
        # Check server is alive before sending
        if server_process and server_process.poll() is not None:
            print(f"  [{n} images] SERVER CRASHED (exit code {server_process.returncode})", flush=True)
            break

        alive = asyncio.run(_check_health(server_backend, port))
        if not alive:
            print(f"  [{n} images] Server health check failed, stopping.", flush=True)
            break

        print(f"  [{n} images] Generating {n}x {resolution}...", end=" ", flush=True)
        t0 = time.time()
        row = _build_row(n, w, h, prompt, input_len, output_len)
        gen_time = time.time() - t0
        print(f"({gen_time:.1f}s) Sending...", end=" ", flush=True)

        output = asyncio.run(_send_one(api_url, model_id, row, timeout_s))

        step = {"n_images": n, "gen_time_s": round(gen_time, 2)}

        if output.success:
            max_ok = n
            step["status"] = "ok"
            step["ttft_ms"] = round(output.ttft * 1000)
            step["e2e_s"] = round(output.latency, 1)
            print(f"OK (TTFT={output.ttft*1000:.0f}ms, e2e={output.latency:.1f}s)", flush=True)
        else:
            err_short = output.error[:200] if output.error else "unknown"
            step["status"] = "failed"
            step["error"] = err_short
            print(f"FAILED: {err_short[:120]}", flush=True)

            if server_process and server_process.poll() is not None:
                step["server_crashed"] = True
                print(f"  Server crashed after {n} images!", flush=True)
            steps.append(step)
            break

        steps.append(step)

    result = {
        "resolution": resolution,
        "width": w,
        "height": h,
        "max_images_ok": max_ok,
        "failed_at": max_ok + 1 if max_ok < max_images else None,
        "steps": steps,
    }
    print(f"\n  Result: {resolution} max = {max_ok} images", flush=True)
    return result


def print_probe_summary(all_results: List[dict]) -> None:
    """Print a summary table across all resolutions."""
    print(f"\n{'='*60}", flush=True)
    print(f"Image Limit Probe Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Resolution':<15} {'Max Images':>12} {'Failed At':>12}", flush=True)
    print(f"{'-'*15} {'-'*12} {'-'*12}", flush=True)
    for r in all_results:
        failed = str(r["failed_at"]) if r["failed_at"] else "N/A"
        print(f"{r['resolution']:<15} {r['max_images_ok']:>12} {failed:>12}", flush=True)
    print(f"{'='*60}", flush=True)
