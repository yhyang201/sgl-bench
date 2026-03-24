"""Accuracy evaluation via Kimi-Vendor-Verifier."""

import os
import shlex
import subprocess
from pathlib import Path


VALID_TASKS = {"ocrbench", "mmmu", "aime2025"}
REPO_URL = "https://github.com/MoonshotAI/Kimi-Vendor-Verifier.git"
DEFAULT_REPO_DIR = Path.home() / ".sgl-bench" / "Kimi-Vendor-Verifier"


def ensure_repo(repo_path: str | None) -> str:
    """Ensure Kimi-Vendor-Verifier repo is available.

    If repo_path is set and exists, use it.
    Otherwise, auto-clone to ~/.sgl-bench/Kimi-Vendor-Verifier and run uv sync.
    """
    if repo_path and os.path.isdir(repo_path):
        return repo_path

    target = str(DEFAULT_REPO_DIR)

    if os.path.isdir(target):
        print(f"Using cached repo: {target}", flush=True)
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=target, capture_output=True, timeout=60,
        )
        return target

    print(f"Cloning Kimi-Vendor-Verifier to {target}...", flush=True)
    os.makedirs(os.path.dirname(target), exist_ok=True)

    result = subprocess.run(
        ["git", "clone", REPO_URL, target],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repo: {result.stderr.strip()}")

    print("Installing dependencies (uv sync)...", flush=True)
    result = subprocess.run(
        ["uv", "sync"],
        cwd=target, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv sync failed: {result.stderr.strip()}")

    result = subprocess.run(
        ["uv", "pip", "install", "-e", "."],
        cwd=target, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv pip install -e . failed: {result.stderr.strip()}")

    print("Kimi-Vendor-Verifier ready.", flush=True)
    return target


def validate_accuracy_config(config: dict) -> None:
    """Validate accuracy section of config."""
    acc = config.get("accuracy", {})
    if not acc:
        raise ValueError("Missing [accuracy] section in config.")

    tasks = acc.get("tasks", [])
    if not tasks:
        raise ValueError("accuracy.tasks must list at least one task (ocrbench, mmmu, aime2025).")
    for t in tasks:
        if t not in VALID_TASKS:
            raise ValueError(f"Unknown accuracy task: {t}. Valid tasks: {sorted(VALID_TASKS)}")


def run_accuracy_task(
    task: str,
    config: dict,
    repo_path: str,
    base_url: str,
    log_dir: str,
) -> dict:
    """Run a single accuracy evaluation task."""
    acc = config["accuracy"]
    # Kimi-Vendor-Verifier's kimi provider reads KIMI_API_KEY and KIMI_BASE_URL
    model = "kimi/" + config["server"]["model_path"]
    api_key = acc.get("api_key", "empty")

    cmd = [
        "uv", "run", "python", "eval.py", task,
        "--model", model,
    ]

    # Per-task extra_args override, fallback to common extra_args
    task_cfg = acc.get(task, {})
    extra = task_cfg.get("extra_args", "").strip()
    if not extra:
        extra = acc.get("extra_args", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))

    cmd_str = " ".join(cmd)
    print(f"Running accuracy task: {task}", flush=True)
    print(f"  command: {cmd_str}", flush=True)

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # avoid conflict with uv's own .venv
    env["KIMI_API_KEY"] = api_key
    env["KIMI_BASE_URL"] = base_url

    log_file_path = os.path.join(log_dir, f"{task}.log")
    with open(log_file_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=repo_path, env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=86400,
        )

    run_data = {
        "task": task,
        "command": cmd_str,
        "base_url": base_url,
        "returncode": result.returncode,
        "log_file": log_file_path,
    }

    if result.returncode != 0:
        print(f"  Warning: {task} exited with code {result.returncode}", flush=True)
        # Print tail of log
        with open(log_file_path, "r") as f:
            lines = f.readlines()
        for line in lines[-10:]:
            print(f"  {line.rstrip()}", flush=True)
    else:
        print(f"  {task} completed. Log: {log_file_path}", flush=True)

    return run_data


def run_accuracy_tests(config: dict, base_url: str, experiment_dir: str) -> list[dict]:
    """Run all configured accuracy tasks."""
    acc = config["accuracy"]
    repo_path = ensure_repo(acc.get("repo_path"))
    tasks = acc["tasks"]
    log_dir = os.path.abspath(os.path.join(experiment_dir, "accuracy_logs"))
    os.makedirs(log_dir, exist_ok=True)

    results = []
    for task in tasks:
        run_data = run_accuracy_task(task, config, repo_path, base_url, log_dir)
        results.append(run_data)

    return results
