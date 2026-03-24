# sgl-bench

Automated benchmark tool for SGLang VLM models.

One command to: launch server → warmup → benchmark → record environment/args/results → git commit.

Supports both **performance benchmarks** (via `bench_serving`) and **accuracy tests** (via [Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier)).

## Install

```bash
pip install -e .
```

Kimi-Vendor-Verifier (for accuracy tests) is auto-cloned and installed on first run. Requires `uv` and `git`.

## Usage

Use `-t` to run a preset task, or `-c` to specify a custom config file:

```bash
# Preset tasks
sgl-bench run -t perf_image -d "image throughput test" -g 0,1,2,3
sgl-bench run -t ocr -d "OCRBench eval" -g 0,1,2,3
sgl-bench run -t mmmu -d "MMMU eval" -g 0,1,2,3
sgl-bench run -t aime25 -d "AIME 2025 eval" -g 0,1,2,3

# Custom config
sgl-bench run -c my_config.toml -d "custom test" -g 0,1,2,3

# Benchmark only (server already running)
sgl-bench bench -t perf_image -d "rerun" -g 0,1,2,3

# Compare two experiments
sgl-bench compare records/exp_a records/exp_b

# List available tasks
sgl-bench tasks
```

## Preset Tasks

Tasks are TOML files in `presets/`. Add or remove files to customize:

| Task | Type | Description |
|------|------|-------------|
| `ocr` | accuracy | OCRBench evaluation |
| `mmmu` | accuracy | MMMU Pro Vision evaluation |
| `aime25` | accuracy | AIME 2025 math evaluation |
| `perf_image` | performance | Image throughput benchmark |

Edit preset files to change `model_path` or other parameters.

## Config

All parameters are defined in a TOML config file. Multiline `extra_args` supported:

```toml
[server]
model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
startup_timeout = 600
extra_args = """
    --port 30000
    --tp-size 4
    --enable-multimodal
"""

[benchmark]
extra_args = """
    --backend sglang
    --port 30000
    --dataset-name image
    --num-prompts 100
"""

[warmup]
enabled = true
num_prompts = 3
seed = 8413927

[accuracy]
model = "Qwen/Qwen2.5-VL-72B-Instruct"
api_key = "empty"
tasks = ["ocrbench"]

[accuracy.ocrbench]
extra_args = "--max-tokens 8192 --stream"

[output]
dir = "./records"
auto_commit = true

[run]
runs = 1
```

## Experiment Output

Each experiment creates a directory under `records/`:

```
records/20260323_143022_test_throughput/
  experiment.json    # environment (GPU/CUDA/sglang version/commit) + commands + results
  config.toml        # config snapshot
  server.log         # server output
  bench_warmup.jsonl
  bench_run_0.jsonl
  accuracy_logs/     # inspect-ai eval logs (if accuracy tests enabled)
```
