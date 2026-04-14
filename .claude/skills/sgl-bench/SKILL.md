---
name: sgl-bench
description: |
  Automated VLM benchmark tool for SGLang/vLLM servers. Use this skill whenever the user wants to:
  run performance benchmarks on vision-language models, stress test VLM servers for stability (OOM/hang detection),
  probe image count limits, measure prefix cache hit rates, run accuracy evaluations (OCRBench/MMMU/AIME),
  compare benchmark results, create or modify benchmark/server/stress/cache preset configs,
  or generate sgl-bench commands. Also trigger when the user mentions sgl-bench, bench_serving, TTFT testing,
  image throughput benchmarking, soak tests, RadixAttention cache efficiency, VLM accuracy evaluation,
  or VLM server capacity probing.
---

# sgl-bench: VLM Benchmark Tool

A config-driven CLI tool for benchmarking SGLang and vLLM vision-language model servers. It automates the full pipeline: server launch, warmup, benchmark execution, result collection, and git commit.

## Commands

### `sgl-bench run` - Performance Benchmark

Run benchmarks with Cartesian product of server x bench configs. Each combination gets its own server instance.

```
sgl-bench run -s <server_config> -b <bench_config> -d <description> [-g <gpus>]
```

| Flag | Description |
|------|-------------|
| `-s/--server` | Server config: file path, directory (all .toml inside, recursive), or preset name in `presets/server/` |
| `-b/--bench` | Bench config: file path, directory, or preset name in `presets/perf/` or `presets/acc/` |
| `-d/--description` | Short description for the experiment session |
| `-g/--gpus` | GPU IDs (e.g., `0,1,2,3`). Auto-detected from tp-size if omitted |

**Examples:**
```bash
# Single server x single bench
sgl-bench run -s presets/server/qwen3_vl_32b/tp4_cuda_ipc.toml \
  -b presets/perf/xiaohongshu/multi_img_1080p_c4.toml \
  -d "xiaohongshu single test"

# Cartesian product: 1 server x all bench configs in directory
sgl-bench run -s presets/server/qwen3_vl_32b/tp4_cuda_ipc.toml \
  -b presets/perf/xiaohongshu/ \
  -d "xiaohongshu TTFT test"

# Cartesian product: directory x directory (all combinations)
sgl-bench run -s presets/server/qwen3_vl_8b/ \
  -b presets/perf/default/ \
  -d "qwen3 8b full sweep"

# Run with accuracy config (auto-detects [accuracy] section)
sgl-bench run -s presets/server/qwen3_vl_8b/tp1.toml \
  -b presets/acc/ocr.toml \
  -d "OCRBench accuracy test"
```

When a directory is passed, all `.toml` files inside are expanded (recursive). Experiments are **grouped by server** — the server restarts only when switching to a new server config.

The `run` command auto-detects what to do based on config sections: if `[benchmark]` has `extra_args`, it runs perf benchmarks; if `[accuracy]` has `tasks`, it runs accuracy tests. Both can coexist in one config.

### `sgl-bench bench` - Benchmark Only (No Server Launch)

For when the server is already running externally. Same as `run` but skips server launch/shutdown.

```
sgl-bench bench -b <bench_config> -d <description> [--port 30000] [-s <server_config>] [-g <gpus>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-b/--bench` | (required) | Bench/accuracy config |
| `-d/--description` | (required) | Experiment description |
| `--port` | 30000 | Port of the running server |
| `-s/--server` | None | Server config (optional, for metadata only) |
| `-g/--gpus` | "external" | GPU IDs (metadata only) |

### `sgl-bench stress` - Stress/Soak Test

Continuously send randomized image requests to detect OOM, hangs, or performance degradation over time. Fresh random images are generated per request to avoid cache hits.

```
sgl-bench stress -s <server_config> -b <stress_config> -d <description> [-g <gpus>]
```

Requires exactly one server config and one stress config.

**Examples:**
```bash
# 1-hour soak test
sgl-bench stress -s presets/server/qwen3_vl_32b/tp4_cuda_ipc.toml \
  -b presets/stress/soak_1h.toml \
  -d "soak test w/ cuda_ipc"

# Quick 5-minute smoke test
sgl-bench stress -s presets/server/qwen3_vl_32b/tp4.toml \
  -b presets/stress/soak_5m_smoke.toml \
  -d "quick smoke test"
```

**Verdict**: PASSED if error rate < 5% and no abort; FAILED otherwise.

### `sgl-bench cache` - Prefix Cache Hit Rate Benchmark

Measure RadixAttention prefix cache efficiency with controlled multimodal request sequences. Automatically injects `--enable-cache-report` into server args so the server returns `cached_tokens` per request.

```
sgl-bench cache -s <server_config> -b <cache_config> -d <description> [-g <gpus>]
```

Requires exactly one server config and one cache config.

**Scenarios:**
- `identical_requests` — Send N identical requests; 2nd+ should fully cache (sanity check)
- `same_image_reuse` — Same images with different text per phase; tests image prefix reuse
- `partial_prefix_sharing` — Mix of shared + divergent images across phases; key test for mm-split optimization
- `multiturn_image` — Multi-turn conversation adding images each round; tests incremental caching

**Examples:**
```bash
# All cache scenarios
sgl-bench cache -s presets/server/qwen3_vl_8b/tp1.toml \
  -b presets/cache/mm_all.toml \
  -d "cache all scenarios"

# Just the partial prefix test
sgl-bench cache -s presets/server/qwen3_vl_8b/tp1.toml \
  -b presets/cache/mm_partial_prefix.toml \
  -d "mm-split partial prefix test"
```

Output: `cache_report.json` with per-scenario, per-phase metrics (prompt_tokens, cached_tokens, cache_hit_rate, TTFT).

### `sgl-bench probe` - Image Limit Probe

Find the maximum number of images a server can handle per request. Sends single requests with incrementing image counts until failure. Server is **restarted between resolutions** so a crash on one doesn't block the next.

```
sgl-bench probe -s <server_config> [-d <description>] [--resolutions <res>] [--max-images N] [--timeout S]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-s/--server` | (required) | Server config |
| `-d/--description` | "image limit probe" | Experiment description |
| `--resolutions` | `720p,1080p,1440x2560` | Comma-separated resolutions to test |
| `--min-images` | 1 | Start probing from this image count |
| `--max-images` | 500 | Upper bound of images to try per resolution |
| `--input-len` | 256 | Text input token length |
| `--output-len` | 32 | Output token length |
| `--timeout` | 300 | Per-request timeout in seconds |

### `sgl-bench compare` - Compare Results

```bash
sgl-bench compare <experiment_dir_a> <experiment_dir_b>
```

Compares metrics side-by-side: throughput, latency percentiles (TTFT, ITL, E2E).

### `sgl-bench tasks` - List Presets

```bash
sgl-bench tasks
```

Lists all available presets grouped by category (server, perf, stress, acc, cache).

---

## Preset Config Format

All configs are TOML. Presets live under `presets/` in the project root:

```
presets/
  server/             # Server launch configs
    qwen3_vl_32b/     # Grouped by model
    qwen3_vl_8b/
    all_vlm_models/   # Many models organized by vendor
      qwen/
      meta_mistral/
      google/
      nvidia/
      ...
    mm_split_test/    # Branch-specific test configs
  perf/               # Performance benchmark configs
    default/          # Standard single-image sweeps
    multi_img/        # Multi-image benchmarks
    xiaohongshu/      # Workload-specific
    backend_cmp/      # Backend comparison
  stress/             # Stress test configs
    soak_1h.toml
    soak_5m_smoke.toml
  acc/                # Accuracy tests
    ocr.toml          # OCRBench
    mmmu.toml         # MMMU Pro Vision
    aime25.toml       # AIME 2025 math
  cache/              # Cache hit rate tests
    mm_all.toml       # All 4 scenarios combined
    mm_same_image.toml
    mm_partial_prefix.toml
```

### Server Config (`[server]`)

```toml
[server]
model_path = "Qwen/Qwen3-VL-32B-Instruct"
extra_args = "--port 30000 --tp-size 4 --enable-multimodal"
env = { SGLANG_USE_CUDA_IPC_TRANSPORT = "1" }    # optional
backend = "sglang"                                 # or "vllm", default: sglang
startup_timeout = 600                              # optional, seconds
```

- `model_path`: HuggingFace model ID
- `extra_args`: CLI args passed to the server launch command
- `env`: Environment variables (e.g., CUDA IPC transport)
- `backend`: `"sglang"` (default) or `"vllm"`. vLLM auto-converts `--tp-size` to `--tensor-parallel-size` and strips `--enable-multimodal`

### Benchmark Config (`[benchmark]`)

```toml
[benchmark]
extra_args = """
    --backend sglang-oai-chat
    --dataset-name image
    --num-prompts 128
    --request-rate inf
    --max-concurrency 16
    --random-input-len 512
    --random-output-len 256
    --image-count 5
    --image-resolution 1080p
"""
```

Key `--backend` values:
- `sglang-oai-chat`: OpenAI-compatible chat API (recommended for multi-image)
- `sglang`: Native /generate API (single image only)
- `vllm`: For vLLM servers

For vLLM backends, sgl-bench uses a built-in native runner instead of the sglang bench_serving subprocess.

### Accuracy Config (`[accuracy]`)

```toml
[accuracy]
tasks = ["ocrbench"]           # Valid: ocrbench, mmmu, aime2025
extra_args = "--max-tokens 8192 --stream"

# Per-task overrides (optional)
[accuracy.ocrbench]
extra_args = "--max-tokens 8192"
```

Accuracy tests use the [Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier) framework (auto-cloned to `~/.sgl-bench/Kimi-Vendor-Verifier` on first use). Tests run via `inspect-ai` under `uv run`.

### Stress Config (`[stress]`)

```toml
[stress]
duration_minutes = 60
max_concurrency = 32
request_rate = "inf"          # or numeric (e.g., 10.0)
pool_size = 10                # number of recipe templates

image_count_range = [1, 5]
image_resolutions = ["720p", "1080p"]
input_len_range = [256, 4096]
output_len_range = [64, 300]

health_check_interval_s = 30
window_minutes = 5
```

All parameters are randomized within their ranges per request. Fresh images are generated each time (not reused) to avoid sglang's RadixAttention cache.

### Cache Config (`[cache]`)

```toml
[cache]
scenarios = ["identical_requests", "same_image_reuse", "partial_prefix_sharing", "multiturn_image"]
output_len = 16
image_resolution = "720p"
text_input_len = 64
max_concurrency = 1
seed = 42

# Per-scenario parameters
[cache.identical_requests]
num_images = 1
num_requests = 4

[cache.same_image_reuse]
num_images = 3
num_phases = 4
requests_per_phase = 1

[cache.partial_prefix_sharing]
shared_images = 4
divergent_images = 4
num_phases = 6
requests_per_phase = 1

[cache.multiturn_image]
num_clients = 1
num_rounds = 8
sub_question_len = 32
```

### Shared Sections (all configs can include)

```toml
[warmup]
enabled = true
num_prompts = 3
seed = 8413927

[output]
dir = "./records"
auto_commit = true
auto_push = false

[run]
runs = 1                      # Number of benchmark runs
```

---

## Results Organization

```
records/
  YYYYMMDD/
    YYYYMMDD_HHMMSS_<description>/       # Session directory
      summary.json                        # All experiments summary
      probe.log                           # (probe only) CLI log
      probe_report.json                   # (probe only) Combined report
      <server>_x_<bench>/                 # Experiment directory
        experiment.json                   # Full metadata + results
        server_config.toml
        bench_config.toml
        server.log
        bench_run_0.json                  # (perf) Benchmark results
        bench_warmup.jsonl                # (perf) Warmup run
        stress_report.json                # (stress) Stress test report
        cache_report.json                 # (cache) Cache hit rate results
        cache.log                         # (cache) CLI log
        accuracy_logs/                    # (accuracy) Test logs
          inspect_eval.log
```

- Each `sgl-bench run/stress/probe/cache` invocation creates one **session**
- Each server x bench combination creates one **experiment** within the session
- `experiment.json` captures: GPU info, CUDA/driver version, sglang git state (branch, commit, dirty status), server command, merged config, and all benchmark/accuracy/stress/cache results
- Auto git commit per session (configurable via `[output]`)

---

## Key Technical Notes

- **CUDA IPC**: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` in server env dramatically improves multi-image TTFT on sglang (from ~3s to ~1s for 5x1080p)
- **vLLM compatibility**: Server configs with `backend = "vllm"` auto-handle arg differences (tp-size -> tensor-parallel-size, no --enable-multimodal). vLLM uses the built-in native runner instead of bench_serving subprocess
- **GPU auto-detection**: Free GPUs are selected based on tp-size from extra_args, skipping GPUs with running compute processes. Use `-g` to override
- **Port conflict**: If the configured port is in use, sgl-bench auto-finds an available port
- **Server lifecycle**: Each experiment (or each resolution in probe) gets a fresh server instance
- **Cache auto-config**: The `cache` command automatically injects `--enable-cache-report` into server args
- **Environment capture**: Every experiment records sglang install type (editable/pip), git branch/commit/dirty status, GPU info, and CUDA version for reproducibility

## Creating New Presets

When the user wants a new benchmark config, create a `.toml` file following the patterns above. Place it in the appropriate `presets/` subdirectory. For multi-image VLM workloads, use `--backend sglang-oai-chat` (not `sglang`) because the native API doesn't handle multi-image properly.
