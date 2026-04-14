"""Prefix cache hit rate benchmark for multimodal workloads.

Measures RadixAttention cache efficiency by sending controlled request
sequences and checking per-request cached_tokens from the server.

Three scenarios:
  A) same_image_reuse     — same image, different text across phases
  B) partial_prefix_sharing — shared + divergent images across phases
  C) multiturn_image       — multi-turn conversation with images
"""

import asyncio
import io
import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
import pybase64
import requests
from PIL import Image

from .runner import _gen_random_text, parse_image_resolution


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheRequestOutput:
    success: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    output_len: int = 0
    generated_text: str = ""
    error: str = ""


@dataclass
class PhaseMetrics:
    phase_index: int
    phase_label: str
    request_count: int
    total_prompt_tokens: int
    total_cached_tokens: int
    cache_hit_rate: float
    expected_min_cached: int
    mean_ttft_ms: float
    p99_ttft_ms: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ScenarioResult:
    scenario_name: str
    phases: List[PhaseMetrics]
    overall_cache_hit_rate: float
    total_prompt_tokens: int
    total_cached_tokens: int
    page_size: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "phases": [p.to_dict() for p in self.phases],
            "overall_cache_hit_rate": self.overall_cache_hit_rate,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "page_size": self.page_size,
            "duration_s": self.duration_s,
        }


@dataclass
class CacheReport:
    model: str
    image_resolution: str
    scenarios: List[ScenarioResult]
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "image_resolution": self.image_resolution,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "duration_s": self.duration_s,
        }


# ---------------------------------------------------------------------------
# Deterministic image generation
# ---------------------------------------------------------------------------


def _gen_deterministic_image(
    width: int, height: int, image_id: int, fmt: str = "jpeg"
) -> str:
    """Generate a deterministic image from image_id, return base64 data URI.

    Same image_id always produces identical bytes, which is critical for
    cache testing: identical image bytes -> identical radix tree hash -> cache hit.
    """
    rng = np.random.RandomState(seed=image_id)
    arr = (rng.rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _build_image_messages(image_uris: List[str], text: str) -> list:
    """Build a single-turn user message with images + text."""
    content = [
        {"type": "image_url", "image_url": {"url": uri}} for uri in image_uris
    ]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _append_turn(
    messages: list,
    assistant_response: str,
    user_question: str,
    image_uris: Optional[List[str]] = None,
) -> list:
    """Append an assistant response and new user question to a conversation.

    If image_uris is provided, the new user message includes images + text.
    """
    new_msgs = list(messages)
    new_msgs.append({"role": "assistant", "content": assistant_response})

    if image_uris:
        content = [
            {"type": "image_url", "image_url": {"url": uri}} for uri in image_uris
        ]
        content.append({"type": "text", "text": user_question})
        new_msgs.append({"role": "user", "content": content})
    else:
        new_msgs.append({"role": "user", "content": user_question})

    return new_msgs


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _flush_cache(base_url: str) -> None:
    """Flush the radix cache and wait for it to take effect."""
    try:
        requests.post(f"{base_url}/flush_cache", timeout=10)
    except Exception:
        pass
    time.sleep(1)


def _get_page_size(base_url: str) -> int:
    """Query server for radix cache page_size."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        resp.raise_for_status()
        return resp.json().get("page_size", 1)
    except Exception:
        return 1


def _get_server_loads(base_url: str) -> dict:
    """Query /v1/loads for aggregate metrics."""
    try:
        resp = requests.get(f"{base_url}/v1/loads", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _compute_expected_cached(cacheable_tokens: int, page_size: int) -> int:
    """Compute expected cached tokens accounting for page alignment."""
    if cacheable_tokens <= 0:
        return 0
    return (cacheable_tokens // page_size) * page_size


# ---------------------------------------------------------------------------
# Cache-aware request sender
# ---------------------------------------------------------------------------


async def _send_cache_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    messages: list,
    output_len: int,
) -> CacheRequestOutput:
    """Send a streaming chat request with stream_options for cache metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    output = CacheRequestOutput()
    generated_text = ""
    st = time.perf_counter()

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

                    data = json.loads(chunk_str)

                    # Extract content tokens
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if output.ttft == 0.0:
                                output.ttft = time.perf_counter() - st
                            generated_text += content

                    # Extract usage from final chunk
                    usage = data.get("usage")
                    if usage:
                        output.prompt_tokens = usage.get("prompt_tokens", 0)
                        output.output_len = usage.get("completion_tokens", 0)
                        details = usage.get("prompt_tokens_details") or {}
                        output.cached_tokens = details.get("cached_tokens", 0)

                output.generated_text = generated_text
                output.success = True
                output.latency = time.perf_counter() - st
            else:
                output.error = f"{response.status}: {await response.text()}"
    except Exception as e:
        output.error = str(e)

    return output


async def _send_phase(
    api_url: str,
    model: str,
    messages_list: List[list],
    output_len: int,
    max_concurrency: int,
) -> List[CacheRequestOutput]:
    """Send a batch of requests sequentially for deterministic cache behavior."""
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    results = []

    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=10 * 1024**2) as session:
        for messages in messages_list:
            result = await _send_cache_request(session, api_url, model, messages, output_len)
            results.append(result)
            if not result.success:
                print(f"  Warning: request failed: {result.error}", flush=True)

    return results


def _aggregate_phase(
    results: List[CacheRequestOutput],
    phase_index: int,
    phase_label: str,
    expected_min_cached: int,
) -> PhaseMetrics:
    """Aggregate per-request results into phase metrics."""
    total_prompt = sum(r.prompt_tokens for r in results if r.success)
    total_cached = sum(r.cached_tokens for r in results if r.success)
    ttfts = [r.ttft for r in results if r.success and r.ttft > 0]
    hit_rate = total_cached / total_prompt if total_prompt > 0 else 0.0

    return PhaseMetrics(
        phase_index=phase_index,
        phase_label=phase_label,
        request_count=len(results),
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        cache_hit_rate=hit_rate,
        expected_min_cached=expected_min_cached,
        mean_ttft_ms=float(np.mean(ttfts) * 1000) if ttfts else 0,
        p99_ttft_ms=float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
    )


# ---------------------------------------------------------------------------
# Scenario A: Same-image reuse
# ---------------------------------------------------------------------------


def _run_same_image_reuse(
    api_url: str,
    model: str,
    base_url: str,
    page_size: int,
    cache_cfg: dict,
    scenario_cfg: dict,
    width: int,
    height: int,
) -> ScenarioResult:
    """Phase 0: N requests with [Image_A] + text_0 (cold).
    Phase 1: N requests with [Image_A] + text_1 (warm — image cached).
    """
    num_images = scenario_cfg.get("num_images", 1)
    num_phases = scenario_cfg.get("num_phases", 2)
    requests_per_phase = scenario_cfg.get("requests_per_phase", 4)
    output_len = cache_cfg.get("output_len", 16)
    text_input_len = cache_cfg.get("text_input_len", 128)
    max_concurrency = cache_cfg.get("max_concurrency", 8)

    # Generate deterministic images (same across all phases)
    image_uris = [
        _gen_deterministic_image(width, height, image_id=i + 1) for i in range(num_images)
    ]

    phases: List[PhaseMetrics] = []
    start_time = time.time()

    for phase_idx in range(num_phases):
        _flush_cache(base_url)

        # Build requests: same images, unique text per phase
        text = f"Phase {phase_idx} question: " + "x" * text_input_len
        messages_list = [
            _build_image_messages(image_uris, text) for _ in range(requests_per_phase)
        ]

        print(f"  Phase {phase_idx}: sending {requests_per_phase} requests...", flush=True)
        results = asyncio.run(_send_phase(api_url, model, messages_list, output_len, max_concurrency))

        # Phase 0: cold. Phase 1+: same image & text as prior request in same phase.
        # Within a phase, 2nd+ request should hit cache for the identical request.
        if phase_idx == 0:
            expected = 0
            label = "cold"
        else:
            # After the first request in this phase, subsequent identical requests
            # should be fully cached (minus page alignment).
            # But since we flushed cache between phases, only intra-phase reuse counts.
            expected = 0
            label = "warm"

        phase_metrics = _aggregate_phase(results, phase_idx, label, expected)
        phases.append(phase_metrics)

    duration = time.time() - start_time
    total_prompt = sum(p.total_prompt_tokens for p in phases)
    total_cached = sum(p.total_cached_tokens for p in phases)

    return ScenarioResult(
        scenario_name="same_image_reuse",
        phases=phases,
        overall_cache_hit_rate=total_cached / total_prompt if total_prompt > 0 else 0.0,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        page_size=page_size,
        duration_s=round(duration, 1),
    )


# ---------------------------------------------------------------------------
# Scenario A2: Same-image reuse (no flush between phases)
# ---------------------------------------------------------------------------


def _run_same_image_reuse_no_flush(
    api_url: str,
    model: str,
    base_url: str,
    page_size: int,
    cache_cfg: dict,
    scenario_cfg: dict,
    width: int,
    height: int,
) -> ScenarioResult:
    """Phase 0: N requests with [Image_A] + text_0 (cold).
    Phase 1: N requests with [Image_A] + text_1 (warm — image prefix cached from phase 0).

    No flush between phases — tests cross-request prefix cache reuse.
    """
    num_images = scenario_cfg.get("num_images", 1)
    num_phases = scenario_cfg.get("num_phases", 2)
    requests_per_phase = scenario_cfg.get("requests_per_phase", 4)
    output_len = cache_cfg.get("output_len", 16)
    text_input_len = cache_cfg.get("text_input_len", 128)
    max_concurrency = cache_cfg.get("max_concurrency", 8)

    # Generate deterministic images (same across all phases)
    image_uris = [
        _gen_deterministic_image(width, height, image_id=i + 1) for i in range(num_images)
    ]

    # Flush once at the start only
    _flush_cache(base_url)

    phases: List[PhaseMetrics] = []
    start_time = time.time()

    for phase_idx in range(num_phases):
        # Different text per phase, same images
        text = f"Phase {phase_idx} question: " + "x" * text_input_len
        messages_list = [
            _build_image_messages(image_uris, text) for _ in range(requests_per_phase)
        ]

        print(f"  Phase {phase_idx}: sending {requests_per_phase} requests...", flush=True)
        results = asyncio.run(_send_phase(api_url, model, messages_list, output_len, max_concurrency))

        label = f"same imgs, text variant {phase_idx}"
        expected = 0

        phase_metrics = _aggregate_phase(results, phase_idx, label, expected)
        phases.append(phase_metrics)

    duration = time.time() - start_time
    total_prompt = sum(p.total_prompt_tokens for p in phases)
    total_cached = sum(p.total_cached_tokens for p in phases)

    return ScenarioResult(
        scenario_name="same_image_reuse",
        phases=phases,
        overall_cache_hit_rate=total_cached / total_prompt if total_prompt > 0 else 0.0,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        page_size=page_size,
        duration_s=round(duration, 1),
    )


# ---------------------------------------------------------------------------
# Scenario B: Partial prefix sharing (mm_split key test)
# ---------------------------------------------------------------------------


def _run_partial_prefix_sharing(
    api_url: str,
    model: str,
    base_url: str,
    page_size: int,
    cache_cfg: dict,
    scenario_cfg: dict,
    width: int,
    height: int,
) -> ScenarioResult:
    """Phase 0: N requests with [Image_A, Image_B] + text_0.
    Phase 1: N requests with [Image_A, Image_C] + text_1.

    With mm_split: Image_A is independently cached -> partial cache hit.
    Without mm_split: different bundle hash -> 0 cache hit.
    """
    shared_images = scenario_cfg.get("shared_images", 1)
    divergent_images = scenario_cfg.get("divergent_images", 1)
    num_phases = scenario_cfg.get("num_phases", 2)
    requests_per_phase = scenario_cfg.get("requests_per_phase", 4)
    output_len = cache_cfg.get("output_len", 16)
    text_input_len = cache_cfg.get("text_input_len", 128)
    max_concurrency = cache_cfg.get("max_concurrency", 8)

    # Shared images: image_id 1..shared_images
    shared_uris = [
        _gen_deterministic_image(width, height, image_id=i + 1)
        for i in range(shared_images)
    ]

    # Flush once at the start
    _flush_cache(base_url)

    phases: List[PhaseMetrics] = []
    start_time = time.time()

    for phase_idx in range(num_phases):
        # Divergent images differ per phase
        divergent_uris = [
            _gen_deterministic_image(
                width, height,
                image_id=1000 + phase_idx * divergent_images + i,
            )
            for i in range(divergent_images)
        ]
        all_uris = shared_uris + divergent_uris

        text = f"Phase {phase_idx} question: " + "y" * text_input_len
        messages_list = [
            _build_image_messages(all_uris, text) for _ in range(requests_per_phase)
        ]

        print(
            f"  Phase {phase_idx}: {shared_images} shared + {divergent_images} divergent images, "
            f"{requests_per_phase} requests...",
            flush=True,
        )
        results = asyncio.run(_send_phase(api_url, model, messages_list, output_len, max_concurrency))

        label = f"shared {shared_images} + new {divergent_images} imgs, text variant {phase_idx}"
        expected = 0

        phase_metrics = _aggregate_phase(results, phase_idx, label, expected)
        phases.append(phase_metrics)

    duration = time.time() - start_time
    total_prompt = sum(p.total_prompt_tokens for p in phases)
    total_cached = sum(p.total_cached_tokens for p in phases)

    return ScenarioResult(
        scenario_name="partial_prefix_sharing",
        phases=phases,
        overall_cache_hit_rate=total_cached / total_prompt if total_prompt > 0 else 0.0,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        page_size=page_size,
        duration_s=round(duration, 1),
    )


# ---------------------------------------------------------------------------
# Scenario C: Multi-turn conversation with images
# ---------------------------------------------------------------------------


def _run_multiturn_image(
    api_url: str,
    model: str,
    base_url: str,
    page_size: int,
    cache_cfg: dict,
    scenario_cfg: dict,
    width: int,
    height: int,
) -> ScenarioResult:
    """Round-barrier multi-turn with a new image per round.

    Round 0: [Image_A] + "Q1" -> R1
    Round 1: [Image_A] + "Q1" + R1 + [Image_B] + "Q2" -> R2
    Round 2: ... + R2 + [Image_C] + "Q3" -> R3

    Each round appends a NEW image + question. The entire prefix from
    previous rounds (including all previous images) should be cached.
    """
    num_rounds = scenario_cfg.get("num_rounds", 3)
    num_clients = scenario_cfg.get("num_clients", cache_cfg.get("num_clients", 4))
    output_len = cache_cfg.get("output_len", 32)
    text_input_len = cache_cfg.get("text_input_len", 128)
    sub_question_len = scenario_cfg.get("sub_question_len", 64)
    max_concurrency = cache_cfg.get("max_concurrency", 8)

    # Pre-generate all images needed (one per round)
    all_image_uris = [
        _gen_deterministic_image(width, height, image_id=round_idx + 1)
        for round_idx in range(num_rounds)
    ]

    # Flush cache for clean state
    _flush_cache(base_url)

    # Track per-client conversation state
    # Round 0: first image + first question
    initial_text = "Describe this image in detail: " + "z" * text_input_len
    client_histories: List[list] = [
        _build_image_messages([all_image_uris[0]], initial_text)
        for _ in range(num_clients)
    ]

    phases: List[PhaseMetrics] = []
    start_time = time.time()

    for round_idx in range(num_rounds):
        messages_list = [h.copy() for h in client_histories]

        n_images_so_far = round_idx + 1
        print(
            f"  Round {round_idx}: {num_clients} clients, "
            f"{n_images_so_far} image(s) in conversation...",
            flush=True,
        )
        results = asyncio.run(
            _send_phase(api_url, model, messages_list, output_len, max_concurrency)
        )

        # Update histories: append response + NEW image + new question
        if round_idx < num_rounds - 1:
            next_image_uri = all_image_uris[round_idx + 1]
            for i, result in enumerate(results):
                if result.success and result.generated_text:
                    sub_q = (
                        f"Now look at this new image and answer: "
                        + "w" * sub_question_len
                    )
                    client_histories[i] = _append_turn(
                        client_histories[i],
                        result.generated_text,
                        sub_q,
                        image_uris=[next_image_uri],
                    )

        label = f"round {round_idx}, {round_idx + 1} img(s) total"

        phase_metrics = _aggregate_phase(results, round_idx, label, expected_min_cached=0)
        phases.append(phase_metrics)

    duration = time.time() - start_time
    total_prompt = sum(p.total_prompt_tokens for p in phases)
    total_cached = sum(p.total_cached_tokens for p in phases)

    return ScenarioResult(
        scenario_name="multiturn_image",
        phases=phases,
        overall_cache_hit_rate=total_cached / total_prompt if total_prompt > 0 else 0.0,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        page_size=page_size,
        duration_s=round(duration, 1),
    )


# ---------------------------------------------------------------------------
# Scenario 0: Identical requests (basic sanity check)
# ---------------------------------------------------------------------------


def _run_identical_requests(
    api_url: str,
    model: str,
    base_url: str,
    page_size: int,
    cache_cfg: dict,
    scenario_cfg: dict,
    width: int,
    height: int,
) -> ScenarioResult:
    """Send N identical requests sequentially. The 2nd+ should fully hit cache.

    This is the most basic sanity check: if identical requests don't cache,
    nothing else will.
    """
    num_images = scenario_cfg.get("num_images", 1)
    num_requests = scenario_cfg.get("num_requests", 8)
    output_len = cache_cfg.get("output_len", 16)
    text_input_len = cache_cfg.get("text_input_len", 128)

    image_uris = [
        _gen_deterministic_image(width, height, image_id=i + 1) for i in range(num_images)
    ]
    text = "Describe what you see in detail: " + "x" * text_input_len
    messages = _build_image_messages(image_uris, text)

    _flush_cache(base_url)

    print(f"  Sending {num_requests} identical requests sequentially...", flush=True)
    start_time = time.time()

    # Send one by one, track per-request cache stats
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    results: List[CacheRequestOutput] = []
    for i in range(num_requests):
        result = asyncio.run(_send_single(api_url, model, messages, output_len, timeout))
        cached_pct = (
            result.cached_tokens / result.prompt_tokens * 100
            if result.prompt_tokens > 0
            else 0
        )
        print(
            f"    req {i}: prompt={result.prompt_tokens} cached={result.cached_tokens} "
            f"({cached_pct:.1f}%) ttft={result.ttft * 1000:.0f}ms",
            flush=True,
        )
        results.append(result)

    duration = time.time() - start_time

    # Split into "first request" (cold) and "rest" (should be cached)
    first = [results[0]]
    rest = results[1:]

    phases = [
        _aggregate_phase(first, 0, "1st request", expected_min_cached=0),
        _aggregate_phase(rest, 1, f"req 2-{num_requests}, identical", expected_min_cached=0),
    ]

    total_prompt = sum(p.total_prompt_tokens for p in phases)
    total_cached = sum(p.total_cached_tokens for p in phases)

    return ScenarioResult(
        scenario_name="identical_requests",
        phases=phases,
        overall_cache_hit_rate=total_cached / total_prompt if total_prompt > 0 else 0.0,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        page_size=page_size,
        duration_s=round(duration, 1),
    )


async def _send_single(
    api_url: str, model: str, messages: list, output_len: int, timeout: aiohttp.ClientTimeout
) -> CacheRequestOutput:
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=10 * 1024**2) as session:
        return await _send_cache_request(session, api_url, model, messages, output_len)


# ---------------------------------------------------------------------------
# Scenario dispatch
# ---------------------------------------------------------------------------

SCENARIO_RUNNERS = {
    "identical_requests": _run_identical_requests,
    "same_image_reuse": _run_same_image_reuse_no_flush,
    "partial_prefix_sharing": _run_partial_prefix_sharing,
    "multiturn_image": _run_multiturn_image,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_cache_test(
    base_url: str,
    model_id: str,
    cache_cfg: dict,
    output_file: Optional[str] = None,
) -> CacheReport:
    """Run a complete cache hit rate benchmark."""
    page_size = _get_page_size(base_url)
    resolution = cache_cfg.get("image_resolution", "720p")
    width, height = parse_image_resolution(resolution)
    scenarios_to_run = cache_cfg.get("scenarios", [])
    seed = cache_cfg.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Cache Hit Rate Benchmark", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"Page Size: {page_size}", flush=True)
    print(f"Image Resolution: {resolution} ({width}x{height})", flush=True)
    print(f"Scenarios: {', '.join(scenarios_to_run)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    start_time = time.time()
    scenario_results: List[ScenarioResult] = []

    for scenario_name in scenarios_to_run:
        runner = SCENARIO_RUNNERS.get(scenario_name)
        if not runner:
            print(f"Unknown scenario: {scenario_name}. Skipping.", flush=True)
            continue

        scenario_cfg = cache_cfg.get(scenario_name, {})
        print(f"--- Scenario: {scenario_name} ---", flush=True)

        result = runner(
            api_url=f"{base_url}/v1/chat/completions",
            model=model_id,
            base_url=base_url,
            page_size=page_size,
            cache_cfg=cache_cfg,
            scenario_cfg=scenario_cfg,
            width=width,
            height=height,
        )
        scenario_results.append(result)
        _print_scenario_result(result)

    duration = time.time() - start_time
    report = CacheReport(
        model=model_id,
        image_resolution=resolution,
        scenarios=scenario_results,
        duration_s=round(duration, 1),
    )

    _print_overall_summary(report)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {output_file}", flush=True)

    return report


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_scenario_result(result: ScenarioResult) -> None:
    for phase in result.phases:
        cached = phase.total_cached_tokens
        total = phase.total_prompt_tokens
        pct = phase.cache_hit_rate * 100
        print(
            f"  Phase {phase.phase_index} ({phase.phase_label}): "
            f"{cached:>6}/{total} cached ({pct:5.1f}%)  "
            f"TTFT: {phase.mean_ttft_ms:.0f}ms",
            flush=True,
        )
    print(
        f"  Overall: {result.total_cached_tokens}/{result.total_prompt_tokens} "
        f"({result.overall_cache_hit_rate * 100:.1f}%)  "
        f"Duration: {result.duration_s:.1f}s\n",
        flush=True,
    )


def _print_overall_summary(report: CacheReport) -> None:
    print(f"{'=' * 60}", flush=True)
    print(f"Summary", flush=True)
    print(f"{'=' * 60}", flush=True)
    for s in report.scenarios:
        print(
            f"  {s.scenario_name:<30} "
            f"cache_hit={s.overall_cache_hit_rate * 100:5.1f}%  "
            f"{s.total_cached_tokens}/{s.total_prompt_tokens} tokens  "
            f"{s.duration_s:.1f}s",
            flush=True,
        )
    print(f"Total duration: {report.duration_s:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)
