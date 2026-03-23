"""Warmup and benchmark execution via bench_serving subprocess."""

import json
import random
import re
import shlex
import subprocess
import time


def _override_args(extra_args: str, overrides: dict[str, str]) -> str:
    """Override specific flags in an extra_args string.

    For each key in overrides, replace its value if present, or append it.
    Example: _override_args("--num-prompts 100 --port 30000", {"--num-prompts": "3"})
             -> "--num-prompts 3 --port 30000"
    """
    result = extra_args
    for flag, value in overrides.items():
        # Match --flag followed by its value (handles both --flag value and --flag=value)
        pattern = rf"({re.escape(flag)})\s+\S+"
        if re.search(pattern, result):
            result = re.sub(pattern, rf"\1 {value}", result)
        else:
            result = f"{result} {flag} {value}"
    return result


def build_bench_command(
    extra_args: str,
    seed: int,
    output_file: str,
) -> list[str]:
    """Build the full bench_serving command.

    Auto-injects: --seed, --output-file, --output-details.
    Everything else comes from extra_args.
    """
    # Override --seed in extra_args (in case user left one in), and inject output flags
    args = _override_args(extra_args, {"--seed": str(seed)})
    # Remove --output-file and --output-details from extra_args if present
    args = re.sub(r"--output-file\s+\S+", "", args)
    args = re.sub(r"--output-details", "", args)

    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--output-file", output_file,
        "--output-details",
    ]
    cmd.extend(shlex.split(args))
    return cmd


def run_warmup(config: dict, experiment_dir: str) -> str | None:
    """Run warmup bench_serving with special seed.

    Completely follows benchmark extra_args, only overrides --seed and --num-prompts.
    Returns the warmup command string, or None if warmup is disabled.
    """
    warmup_cfg = config.get("warmup", {})
    if not warmup_cfg.get("enabled", True):
        return None

    seed = warmup_cfg.get("seed", 8413927)
    num_prompts = warmup_cfg.get("num_prompts", 3)
    output_file = f"{experiment_dir}/bench_warmup.jsonl"

    # Start from benchmark extra_args, override seed and num-prompts
    extra_args = config["benchmark"].get("extra_args", "")
    extra_args = _override_args(extra_args, {"--num-prompts": str(num_prompts)})

    cmd = build_bench_command(extra_args, seed, output_file)
    cmd_str = " ".join(cmd)

    print(f"Running warmup (seed={seed}, num_prompts={num_prompts})...", flush=True)

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"Warning: warmup exited with code {result.returncode}", flush=True)
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}", flush=True)
    else:
        print("Warmup completed.", flush=True)

    time.sleep(2)
    return cmd_str


def run_benchmark(config: dict, run_index: int, experiment_dir: str) -> dict:
    """Run a single benchmark with a random seed.

    Returns a dict with: run_index, seed, command, results.
    """
    seed = random.randint(1, 999999)
    output_file = f"{experiment_dir}/bench_run_{run_index}.jsonl"

    extra_args = config["benchmark"].get("extra_args", "")
    cmd = build_bench_command(extra_args, seed, output_file)
    cmd_str = " ".join(cmd)

    print(f"Running benchmark run {run_index} (seed={seed})...", flush=True)

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=3600,
    )

    run_data = {
        "run_index": run_index,
        "seed": seed,
        "command": cmd_str,
        "stdout": result.stdout,
        "results": {},
    }

    if result.returncode != 0:
        run_data["error"] = result.stderr[:2000] if result.stderr else f"exit code {result.returncode}"
        print(f"Warning: benchmark run {run_index} exited with code {result.returncode}", flush=True)
    else:
        run_data["results"] = parse_bench_results(output_file)
        print(f"Benchmark run {run_index} completed.", flush=True)

    return run_data


def parse_bench_results(jsonl_path: str) -> dict:
    """Parse the last line of a bench_serving JSONL output file."""
    try:
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        if not lines:
            return {"error": "empty output file"}
        return json.loads(lines[-1])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": str(e)}
