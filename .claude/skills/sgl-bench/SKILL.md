---
name: sgl-bench
description: |
  Automated VLM benchmark tool for SGLang/vLLM servers. Use this skill whenever the user wants to:
  run performance benchmarks on vision-language models, stress test VLM servers for stability (OOM/hang detection),
  probe image count limits, compare benchmark results, create or modify benchmark/server/stress preset configs,
  or generate sgl-bench commands. Also trigger when the user mentions sgl-bench, bench_serving, TTFT testing,
  image throughput benchmarking, soak tests, or VLM server capacity probing.
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
| `-s/--server` | Server config: file path, directory (all .toml inside), or preset name in `presets/server/` |
| `-b/--bench` | Bench config: file path, directory, or preset name in `presets/perf/` |
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
```

When a directory is passed, all `.toml` files inside are expanded. Experiments are **grouped by server** — the server restarts only when switching to a new server config.

### `sgl-bench stress` - Stress/Soak Test

Continuously send randomized image requests to detect OOM, hangs, or performance degradation over time. Fresh random images are generated per request to avoid cache hits.

```
sgl-bench stress -s <server_config> -b <stress_config> -d <description> [-g <gpus>]
```

| Flag | Description |
|------|-------------|
| `-s/--server` | Exactly one server config |
| `-b/--bench` | Stress config: file or preset name in `presets/stress/` |
| `-d/--description` | Experiment description |

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

### `sgl-bench probe` - Image Limit Probe

Find the maximum number of images a server can handle per request. Sends single requests with incrementing image counts (1, 2, 3, ...) until the server fails. Server is **restarted between resolutions** so a crash on one doesn't block the next.

```
sgl-bench probe -s <server_config> [-d <description>] [--resolutions <res>] [--max-images N] [--timeout S]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-s/--server` | (required) | Server config |
| `-d/--description` | "image limit probe" | Experiment description |
| `--resolutions` | `720p,1080p,1440x2560` | Comma-separated resolutions to test |
| `--max-images` | 500 | Upper bound of images to try per resolution |
| `--timeout` | 300 | Per-request timeout in seconds |
| `--input-len` | 256 | Text input token length |
| `--output-len` | 32 | Output token length |

**Examples:**
```bash
# Default: test 720p, 1080p, 2k
sgl-bench probe -s presets/server/qwen3_vl_32b/tp4_cuda_ipc.toml \
  -d "max image count probe"

# Compare sglang vs vllm capacity
sgl-bench probe -s presets/server/qwen3_vl_32b/tp4.toml \
  -d "max image count probe sglang"
sgl-bench probe -s presets/server/qwen3_vl_32b/tp4_vllm.toml \
  -d "max image count probe vllm"
```

### `sgl-bench compare` - Compare Results

```bash
sgl-bench compare <experiment_dir_a> <experiment_dir_b>
```

### `sgl-bench tasks` - List Presets

```bash
sgl-bench tasks
```

### `sgl-bench bench` - Benchmark Only (No Server Launch)

For when the server is already running externally.

---

## Preset Config Format

All configs are TOML. Presets live under `presets/` in the project root:

```
presets/
  server/           # Server launch configs
    qwen3_vl_32b/   # Grouped by model
      tp4.toml
      tp4_cuda_ipc.toml
      tp4_vllm.toml
      tp1.toml
    qwen3_vl_8b/
      tp1.toml
      tp2.toml
  perf/             # Performance benchmark configs
    default/        # Standard single-image sweeps
    multi_img/      # Multi-image benchmarks
    xiaohongshu/    # Workload-specific (5 images, ~10k image tokens)
    backend_cmp/    # Backend comparison
  stress/           # Stress test configs
    soak_1h.toml
    soak_5m_smoke.toml
  acc/              # Accuracy tests
    mmmu.toml
    ocr.toml
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

---

## Results Organization

```
records/
  YYYYMMDD/
    YYYYMMDD_HHMMSS_<description>/       # Session directory
      summary.json                        # All experiments summary
      <server>_x_<bench>/                 # Experiment directory
        experiment.json
        server_config.toml
        bench_config.toml
        server.log
        result.json / stress_report.json / probe_report.json
```

- Each `sgl-bench run/stress/probe` invocation creates one **session**
- Each server x bench combination creates one **experiment** within the session
- Auto git commit per session (configurable)

---

## Key Technical Notes

- **CUDA IPC**: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` in server env dramatically improves multi-image TTFT on sglang (from ~3s to ~1s for 5x1080p)
- **vLLM compatibility**: Server configs with `backend = "vllm"` auto-handle arg differences (tp-size -> tensor-parallel-size, no --enable-multimodal)
- **GPU auto-detection**: GPUs are selected based on tp-size from extra_args. Use `-g` to override
- **Port conflict**: If the configured port is in use, sgl-bench auto-finds an available port
- **Server lifecycle**: Each experiment (or each resolution in probe) gets a fresh server instance

## Creating New Presets

When the user wants a new benchmark config, create a `.toml` file following the patterns above. Place it in the appropriate `presets/` subdirectory. For multi-image VLM workloads, use `--backend sglang-oai-chat` (not `sglang`) because the native API doesn't handle multi-image properly.
