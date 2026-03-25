"""Self-contained benchmark runner.

Generates image datasets and sends requests via OpenAI-compatible chat API.
Works with both sglang and vllm servers. Replaces the dependency on
sglang.bench_serving while keeping the same logic.
"""

import asyncio
import io
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pybase64
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: int = 0
    vision_prompt_len: int = 0
    image_data: Optional[List[str]] = None

    def __post_init__(self):
        if not self.text_prompt_len:
            self.text_prompt_len = self.prompt_len
        if not self.vision_prompt_len:
            self.vision_prompt_len = 0


@dataclass
class RequestOutput:
    success: bool = False
    generated_text: str = ""
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    output_len: int = 0
    prompt_len: int = 0
    start_time: float = 0.0
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated metrics from a benchmark run."""
    completed: int = 0
    failed: int = 0
    total_input: int = 0
    total_input_text: int = 0
    total_input_vision: int = 0
    total_output: int = 0
    duration_s: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    median_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Image dataset generation (ported from sglang benchmark/datasets/image.py)
# ---------------------------------------------------------------------------

RESOLUTION_MAP = {
    "4k": (3840, 2160),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "360p": (640, 360),
}


def parse_image_resolution(resolution: str) -> Tuple[int, int]:
    """Parse resolution to (width, height)."""
    if resolution in RESOLUTION_MAP:
        return RESOLUTION_MAP[resolution]
    res = resolution.strip().lower()
    if "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[1]), int(parts[0]))  # heightxwidth -> (w, h)
    raise ValueError(
        f"Unsupported resolution: {resolution}. "
        "Use 4k, 1080p, 720p, 360p, or 'heightxwidth' (e.g. 1080x1920)."
    )


def _gen_random_image(width: int, height: int, fmt: str = "jpeg") -> Tuple[Image.Image, str]:
    """Generate a random image and return (PIL image, base64 data URI)."""
    arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/{fmt};base64,{encoded}"
    return img, data_uri


def _gen_random_text(tokenizer, token_num: int, image_pad_id=None) -> str:
    """Generate random text of specified token length, avoiding image pad tokens."""
    all_tokens = list(tokenizer.get_vocab().values())
    if image_pad_id is not None:
        all_tokens.remove(image_pad_id)
    selected = random.choices(all_tokens, k=token_num)
    return tokenizer.decode(selected)


def generate_image_dataset(
    num_prompts: int,
    image_count: int,
    input_len: int,
    output_len: int,
    image_resolution: str,
    model_id: str,
    range_ratio: float = 1.0,
    image_format: str = "jpeg",
) -> List[DatasetRow]:
    """Generate a dataset of image benchmark requests.

    Args:
        num_prompts: Number of requests to generate.
        image_count: Number of images per request.
        input_len: Target text token length.
        output_len: Target output token length.
        image_resolution: Resolution string (e.g. "1080p", "720p", "1440x2560").
        model_id: HuggingFace model ID for the processor.
        range_ratio: Ratio for random length variation (1.0 = no variation).
        image_format: Image format (jpeg/png).

    Returns:
        List of DatasetRow with prompts and base64 image data.
    """
    width, height = parse_image_resolution(image_resolution)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Sample text/output lengths
    input_lens = _compute_random_lens(input_len, range_ratio, num_prompts)
    output_lens = _compute_random_lens(output_len, range_ratio, num_prompts)

    image_pad_id = getattr(processor, "image_token_id", None)

    dataset: List[DatasetRow] = []
    for i in range(num_prompts):
        # Generate text prompt
        text_prompt = _gen_random_text(processor.tokenizer, int(input_lens[i]), image_pad_id)

        # Generate images
        images = []
        images_base64 = []
        for _ in range(image_count):
            img, data_uri = _gen_random_image(width, height, image_format)
            images.append(img)
            images_base64.append(data_uri)

        # Build multimodal content and compute token counts via processor
        content_items = [
            {"type": "image", "image": {"url": uri}}
            for uri in images_base64
        ]
        content_items.append({"type": "text", "text": text_prompt})

        try:
            prompt_str = processor.apply_chat_template(
                [{"role": "user", "content": content_items}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            prompt_str = f"<image>{text_prompt}"

        # Total tokens (text + vision)
        total_len = processor(
            text=[prompt_str], images=images, padding=False, return_tensors="pt"
        )["input_ids"].numel()

        # Text-only tokens
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

        dataset.append(DatasetRow(
            prompt=text_prompt,
            prompt_len=total_len,
            output_len=int(output_lens[i]),
            text_prompt_len=text_len,
            vision_prompt_len=vision_len,
            image_data=images_base64,
        ))

    total_input = sum(r.prompt_len for r in dataset)
    total_vision = sum(r.vision_prompt_len for r in dataset)
    print(f"Generated {num_prompts} requests: "
          f"{total_input} input tokens ({total_vision} vision), "
          f"{image_count} images/req @ {image_resolution}", flush=True)
    return dataset


def _compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1), full_len + 1, size=num
    ).tolist()


# ---------------------------------------------------------------------------
# Async request sending (OpenAI chat completions API, works for sglang+vllm)
# ---------------------------------------------------------------------------

def _create_client_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    return aiohttp.ClientSession(timeout=timeout, read_bufsize=10 * 1024**2)


async def _send_chat_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    row: DatasetRow,
) -> RequestOutput:
    """Send a single streaming chat completion request and measure timing."""
    # Build message content
    if row.image_data:
        content_items = [
            {"type": "image_url", "image_url": {"url": uri}}
            for uri in row.image_data
        ]
        content_items.append({"type": "text", "text": row.prompt})
        messages = [{"role": "user", "content": content_items}]
    else:
        messages = [{"role": "user", "content": row.prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": row.output_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }

    output = RequestOutput(prompt_len=row.prompt_len, output_len=row.output_len)
    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk_str = chunk_bytes.decode("utf-8")
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]
                    if chunk_str == "[DONE]":
                        continue

                    latency = time.perf_counter() - st
                    data = json.loads(chunk_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)
                        most_recent_timestamp = timestamp
                        generated_text += content

                    # Check for usage in final chunk
                    usage_tokens = (data.get("usage") or {}).get("completion_tokens")
                    if usage_tokens is not None:
                        output.output_len = usage_tokens

                output.generated_text = generated_text
                output.success = True
                output.latency = time.perf_counter() - st
            else:
                output.error = f"{response.status}: {await response.text()}"
                output.success = False
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))

    return output


async def _request_generator(
    requests: List[DatasetRow],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    """Yield requests with optional rate limiting."""
    for req in requests:
        yield req
        if request_rate != float("inf"):
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------

async def _run_benchmark_async(
    api_url: str,
    model: str,
    dataset: List[DatasetRow],
    request_rate: float,
    max_concurrency: Optional[int],
) -> Tuple[List[RequestOutput], float]:
    """Run the benchmark asynchronously. Returns (outputs, duration)."""
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_send(session, row):
        if semaphore:
            async with semaphore:
                return await _send_chat_request(session, api_url, model, row)
        return await _send_chat_request(session, api_url, model, row)

    pbar = tqdm(total=len(dataset), desc="Benchmark")
    start_time = time.perf_counter()

    async with _create_client_session() as session:
        tasks = []
        async for row in _request_generator(dataset, request_rate):
            task = asyncio.create_task(limited_send(session, row))
            task.add_done_callback(lambda _: pbar.update(1))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time
    pbar.close()
    return list(outputs), duration


def _calculate_metrics(
    dataset: List[DatasetRow],
    outputs: List[RequestOutput],
    duration: float,
) -> BenchmarkResult:
    """Calculate benchmark metrics from outputs."""
    ttfts = []
    itls = []
    e2e_latencies = []
    total_input = 0
    total_input_text = 0
    total_input_vision = 0
    total_output = 0
    completed = 0

    for i, out in enumerate(outputs):
        if out.success:
            completed += 1
            total_input += dataset[i].prompt_len
            total_input_text += dataset[i].text_prompt_len
            total_input_vision += dataset[i].vision_prompt_len
            total_output += out.output_len
            ttfts.append(out.ttft)
            itls.extend(out.itl)
            e2e_latencies.append(out.latency)

    failed = len(outputs) - completed

    return BenchmarkResult(
        completed=completed,
        failed=failed,
        total_input=total_input,
        total_input_text=total_input_text,
        total_input_vision=total_input_vision,
        total_output=total_output,
        duration_s=duration,
        request_throughput=completed / duration if duration > 0 else 0,
        output_throughput=total_output / duration if duration > 0 else 0,
        mean_ttft_ms=float(np.mean(ttfts) * 1000) if ttfts else 0,
        median_ttft_ms=float(np.median(ttfts) * 1000) if ttfts else 0,
        p99_ttft_ms=float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
        mean_itl_ms=float(np.mean(itls) * 1000) if itls else 0,
        median_itl_ms=float(np.median(itls) * 1000) if itls else 0,
        p99_itl_ms=float(np.percentile(itls, 99) * 1000) if itls else 0,
        mean_e2e_latency_ms=float(np.mean(e2e_latencies) * 1000) if e2e_latencies else 0,
        median_e2e_latency_ms=float(np.median(e2e_latencies) * 1000) if e2e_latencies else 0,
        p99_e2e_latency_ms=float(np.percentile(e2e_latencies, 99) * 1000) if e2e_latencies else 0,
    )


def run_image_benchmark(
    base_url: str,
    model_id: str,
    num_prompts: int,
    image_count: int,
    input_len: int,
    output_len: int,
    image_resolution: str,
    request_rate: float = float("inf"),
    max_concurrency: Optional[int] = None,
    output_file: Optional[str] = None,
) -> BenchmarkResult:
    """Run a complete image benchmark.

    Args:
        base_url: Server base URL (e.g. "http://127.0.0.1:30000").
        model_id: HuggingFace model ID.
        num_prompts: Number of requests.
        image_count: Images per request.
        input_len: Text token length.
        output_len: Output token length.
        image_resolution: e.g. "1080p", "720p", "1440x2560".
        request_rate: Requests per second (inf = all at once).
        max_concurrency: Max concurrent requests.
        output_file: Optional path to save results JSON.

    Returns:
        BenchmarkResult with all metrics.
    """
    api_url = f"{base_url}/v1/chat/completions"

    print(f"Generating dataset: {num_prompts} prompts, {image_count} images @ {image_resolution}...",
          flush=True)
    dataset = generate_image_dataset(
        num_prompts=num_prompts,
        image_count=image_count,
        input_len=input_len,
        output_len=output_len,
        image_resolution=image_resolution,
        model_id=model_id,
    )

    print(f"Running benchmark: rate={request_rate}, concurrency={max_concurrency}...", flush=True)
    outputs, duration = asyncio.run(_run_benchmark_async(
        api_url=api_url,
        model=model_id,
        dataset=dataset,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
    ))

    result = _calculate_metrics(dataset, outputs, duration)

    # Print summary
    print(f"\n{'='*50}", flush=True)
    print(f"Completed: {result.completed}/{len(dataset)}, Failed: {result.failed}", flush=True)
    print(f"Duration: {result.duration_s:.1f}s", flush=True)
    print(f"Request throughput: {result.request_throughput:.2f} req/s", flush=True)
    print(f"Output throughput: {result.output_throughput:.1f} tok/s", flush=True)
    print(f"Mean TTFT: {result.mean_ttft_ms:.1f} ms (p99: {result.p99_ttft_ms:.1f})", flush=True)
    print(f"Mean ITL: {result.mean_itl_ms:.2f} ms (p99: {result.p99_itl_ms:.2f})", flush=True)
    print(f"Mean E2E: {result.mean_e2e_latency_ms:.1f} ms (p99: {result.p99_e2e_latency_ms:.1f})", flush=True)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {output_file}", flush=True)

    return result
