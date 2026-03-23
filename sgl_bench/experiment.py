"""Experiment data model, directory management, and JSON persistence."""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Experiment:
    experiment_id: str
    description: str
    gpu_ids: str
    output_dir: Path
    config: dict = field(default_factory=dict)
    sglang_info: dict = field(default_factory=dict)
    gpu_info: dict = field(default_factory=dict)
    server_cmd: str = ""
    warmup_cmd: str | None = None
    benchmark_runs: list = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"
    error: str | None = None

    @staticmethod
    def create(description: str, gpu_ids: str, base_dir: str, config: dict) -> "Experiment":
        """Create a new experiment with timestamp-based ID."""
        now = datetime.now()
        experiment_id = now.strftime("%Y%m%d_%H%M%S")

        # Create short directory name from description
        short_desc = _slugify(description)[:40]
        dir_name = f"{experiment_id}_{short_desc}" if short_desc else experiment_id
        output_dir = Path(base_dir) / dir_name

        return Experiment(
            experiment_id=experiment_id,
            description=description,
            gpu_ids=gpu_ids,
            output_dir=output_dir,
            config=config,
            started_at=now.isoformat(),
            status="running",
        )

    def create_directory(self) -> None:
        """Create the experiment output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def copy_config(self, config_path: str) -> None:
        """Copy the original config file into the experiment directory."""
        dest = self.output_dir / "config.toml"
        shutil.copy2(config_path, dest)

    def save_partial(self) -> None:
        """Save current state (for crash recovery)."""
        self._write_json()

    def save(self) -> None:
        """Final save with finished timestamp."""
        self.finished_at = datetime.now().isoformat()
        if self.status == "running":
            self.status = "completed"
        self._write_json()

    def mark_failed(self, error: str) -> None:
        """Mark experiment as failed with error message."""
        self.status = "failed"
        self.error = error
        self.finished_at = datetime.now().isoformat()
        self._write_json()

    def _write_json(self) -> None:
        """Write experiment.json to the output directory."""
        data = {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "gpu_ids": self.gpu_ids,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "error": self.error,
            "config": self.config,
            "platform": self.gpu_info,
            "sglang_info": self.sglang_info,
            "server": {"command": self.server_cmd},
            "warmup": {"command": self.warmup_cmd, "seed": self.config.get("warmup", {}).get("seed", 8413927)},
            "benchmark_runs": [
                {
                    "run_index": r.get("run_index"),
                    "seed": r.get("seed"),
                    "command": r.get("command"),
                    "results": r.get("results", {}),
                    "error": r.get("error"),
                }
                for r in self.benchmark_runs
            ],
        }
        path = self.output_dir / "experiment.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def git_commit(self, auto_push: bool = False) -> None:
        """Auto-commit experiment results to git."""
        experiments_dir = self.output_dir.parent

        # Initialize git repo if needed
        git_dir = experiments_dir / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"],
                cwd=str(experiments_dir),
                capture_output=True,
            )
            # Create .gitignore
            gitignore = experiments_dir / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*.pyc\n__pycache__/\n")

        # Add and commit
        subprocess.run(
            ["git", "add", str(self.output_dir.name)],
            cwd=str(experiments_dir),
            capture_output=True,
        )
        msg = f"experiment: {self.experiment_id} - {self.description}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(experiments_dir),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Git commit: {msg}", flush=True)
        else:
            print(f"Git commit skipped (no changes or error): {result.stderr.strip()}", flush=True)

        # Optional push
        if auto_push:
            push_result = subprocess.run(
                ["git", "push"],
                cwd=str(experiments_dir),
                capture_output=True, text=True,
            )
            if push_result.returncode == 0:
                print("Git push completed.", flush=True)
            else:
                print(f"Git push failed: {push_result.stderr.strip()}", flush=True)

    def print_summary(self) -> None:
        """Print a summary table of benchmark results."""
        runs = self.benchmark_runs
        if not runs:
            print("No benchmark runs completed.", flush=True)
            return

        # Collect key metrics from all runs
        metrics_keys = [
            "request_throughput",
            "output_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p99_itl_ms",
            "mean_e2e_latency_ms",
            "p99_e2e_latency_ms",
        ]

        print("\n" + "=" * 60, flush=True)
        print(f"Experiment: {self.experiment_id}", flush=True)
        print(f"Description: {self.description}", flush=True)
        print(f"Status: {self.status}", flush=True)
        print("=" * 60, flush=True)

        if len(runs) == 1:
            # Single run: just print the metrics
            results = runs[0].get("results", {})
            print(f"  seed: {runs[0].get('seed')}", flush=True)
            for key in metrics_keys:
                if key in results:
                    print(f"  {key}: {results[key]:.2f}", flush=True)
        else:
            # Multiple runs: print each run + mean/std
            import statistics

            print(f"  Runs: {len(runs)}", flush=True)
            print(f"  Seeds: {[r.get('seed') for r in runs]}", flush=True)
            print("-" * 60, flush=True)

            header = f"{'metric':<28}"
            for r in runs:
                header += f"  {'run_' + str(r['run_index']):>10}"
            header += f"  {'mean':>10}  {'std':>10}"
            print(header, flush=True)
            print("-" * 60, flush=True)

            for key in metrics_keys:
                values = [r.get("results", {}).get(key) for r in runs]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                row = f"{key:<28}"
                for r in runs:
                    v = r.get("results", {}).get(key)
                    row += f"  {v:>10.2f}" if v is not None else f"  {'N/A':>10}"
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row += f"  {mean:>10.2f}  {std:>10.2f}"
                print(row, flush=True)

        print("=" * 60, flush=True)
        print(f"Results saved to: {self.output_dir}", flush=True)


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    # Keep Chinese characters, alphanumeric, and basic separators
    text = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")
