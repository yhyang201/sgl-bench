# sgl-bench

Automated benchmark tool for SGLang VLM models.

One command to: launch server → warmup → benchmark → record environment/args/results → git commit.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Full pipeline
sgl-bench run -c presets/qwen_vl_72b_image.toml -d "test throughput on 4xA100" -g 0,1,2,3

# Benchmark only (server already running)
sgl-bench bench -c presets/qwen_vl_72b_image.toml -d "rerun with different prompts" -g 0,1,2,3

# Compare two experiments
sgl-bench compare records/exp_a records/exp_b
```

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
    --image-count 1
    --image-resolution 1080p
"""

[warmup]
enabled = true
num_prompts = 3
seed = 8413927

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
  experiment.json   # environment (GPU/CUDA/sglang version/commit) + commands + results
  config.toml       # config snapshot
  server.log        # server output
  bench_warmup.jsonl
  bench_run_0.jsonl
```
