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
class Session:
    """A session groups multiple experiments from a single `sgl-bench run` invocation."""

    session_id: str
    description: str
    session_dir: Path
    experiments: list["Experiment"] = field(default_factory=list)
    started_at: str = ""

    @staticmethod
    def create(description: str, base_dir: str) -> "Session":
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        date_dir = now.strftime("%Y%m%d")
        short_desc = _slugify(description)[:40]
        dir_name = f"{session_id}_{short_desc}" if short_desc else session_id
        session_dir = Path(base_dir) / date_dir / dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return Session(
            session_id=session_id,
            description=description,
            session_dir=session_dir,
            started_at=now.isoformat(),
        )

    def create_experiment(
        self,
        config: dict,
        gpu_ids: str,
        server_config_name: str,
        bench_config_name: str,
    ) -> "Experiment":
        """Create an experiment inside this session."""
        sub_dir = f"{server_config_name}_x_{bench_config_name}"
        output_dir = self.session_dir / sub_dir
        now = datetime.now()
        exp = Experiment(
            experiment_id=now.strftime("%Y%m%d_%H%M%S"),
            description=self.description,
            gpu_ids=gpu_ids,
            output_dir=output_dir,
            config=config,
            server_config_name=server_config_name,
            bench_config_name=bench_config_name,
            started_at=now.isoformat(),
            status="running",
        )
        self.experiments.append(exp)
        return exp

    def save_summary(self) -> None:
        """Write summary.json aggregating all experiment results."""
        experiments = []
        for exp in self.experiments:
            entry = {
                "server": exp.server_config_name,
                "bench": exp.bench_config_name,
                "status": exp.status,
                "dir": str(exp.output_dir.name),
            }
            # Extract key benchmark metrics
            if exp.benchmark_runs:
                results = exp.benchmark_runs[0].get("results", {})
                for key in [
                    "request_throughput", "output_throughput",
                    "mean_ttft_ms", "p99_ttft_ms",
                    "mean_itl_ms", "p99_itl_ms",
                    "mean_e2e_latency_ms", "p99_e2e_latency_ms",
                ]:
                    if key in results:
                        entry[key] = results[key]
            # Extract accuracy results
            if exp.accuracy_results:
                entry["accuracy"] = {
                    r.get("task"): "OK" if r.get("returncode") == 0 else f"FAILED({r.get('returncode')})"
                    for r in exp.accuracy_results
                }
            # Extract cache results
            if exp.cache_report:
                entry["cache"] = {
                    s["scenario_name"]: f"{s['overall_cache_hit_rate'] * 100:.1f}%"
                    for s in exp.cache_report.get("scenarios", [])
                }
            if exp.error:
                entry["error"] = exp.error
            experiments.append(entry)

        summary = {
            "session_id": self.session_id,
            "description": self.description,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(),
            "total": len(self.experiments),
            "completed": sum(1 for e in self.experiments if e.status == "completed"),
            "failed": sum(1 for e in self.experiments if e.status == "failed"),
            "experiments": experiments,
        }
        path = self.session_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def git_commit(self, auto_push: bool = False) -> None:
        """Auto-commit the entire session directory to the project git repo."""
        repo_root = _find_git_root(self.session_dir)
        if not repo_root:
            print("Git commit skipped: not inside a git repo.", flush=True)
            return

        try:
            rel_path = self.session_dir.resolve().relative_to(repo_root.resolve())
        except ValueError:
            print("Git commit skipped: session dir is outside the git repo.", flush=True)
            return

        subprocess.run(
            ["git", "add", str(rel_path)],
            cwd=str(repo_root),
            capture_output=True,
        )
        n_ok = sum(1 for e in self.experiments if e.status == "completed")
        n_fail = sum(1 for e in self.experiments if e.status == "failed")
        msg = f"session: {self.session_id} - {self.description} ({n_ok} ok, {n_fail} fail)"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(repo_root),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Git commit: {msg}", flush=True)
        else:
            print(f"Git commit skipped (no changes or error): {result.stderr.strip()}", flush=True)

        if auto_push:
            push_result = subprocess.run(
                ["git", "push"],
                cwd=str(repo_root),
                capture_output=True, text=True,
            )
            if push_result.returncode == 0:
                print("Git push completed.", flush=True)
            else:
                print(f"Git push failed: {push_result.stderr.strip()}", flush=True)

    def print_final_summary(self) -> None:
        """Print a table summarizing all experiments in this session."""
        print(f"\n{'='*70}", flush=True)
        print(f"Session: {self.session_id} - {self.description}", flush=True)
        n_ok = sum(1 for e in self.experiments if e.status == "completed")
        n_fail = sum(1 for e in self.experiments if e.status == "failed")
        print(f"Total: {len(self.experiments)}  Completed: {n_ok}  Failed: {n_fail}", flush=True)
        print(f"{'='*70}", flush=True)

        # Print benchmark results table if any
        bench_exps = [e for e in self.experiments if e.benchmark_runs]
        if bench_exps:
            print(f"\n{'server':<20} {'bench':<30} {'req/s':>8} {'out tok/s':>10} {'ttft ms':>8} {'itl ms':>8} {'status':>8}", flush=True)
            print("-" * 96, flush=True)
            for exp in bench_exps:
                r = exp.benchmark_runs[0].get("results", {}) if exp.benchmark_runs else {}
                print(
                    f"{exp.server_config_name:<20} "
                    f"{exp.bench_config_name:<30} "
                    f"{r.get('request_throughput', 0):>8.2f} "
                    f"{r.get('output_throughput', 0):>10.1f} "
                    f"{r.get('mean_ttft_ms', 0):>8.1f} "
                    f"{r.get('mean_itl_ms', 0):>8.1f} "
                    f"{exp.status:>8}",
                    flush=True,
                )

        # Print accuracy results if any
        acc_exps = [e for e in self.experiments if e.accuracy_results]
        if acc_exps:
            print(f"\n{'server':<20} {'bench':<20} {'task':<15} {'status':>8}", flush=True)
            print("-" * 66, flush=True)
            for exp in acc_exps:
                for r in exp.accuracy_results:
                    status = "OK" if r.get("returncode") == 0 else "FAILED"
                    print(
                        f"{exp.server_config_name:<20} "
                        f"{exp.bench_config_name:<20} "
                        f"{r.get('task', ''):<15} "
                        f"{status:>8}",
                        flush=True,
                    )

        # Print cache results if any
        cache_exps = [e for e in self.experiments if e.cache_report]
        if cache_exps:
            print(f"\n{'server':<20} {'bench':<20} {'scenario':<25} {'cache_hit':>10}", flush=True)
            print("-" * 78, flush=True)
            for exp in cache_exps:
                for s in exp.cache_report.get("scenarios", []):
                    hit_pct = f"{s['overall_cache_hit_rate'] * 100:.1f}%"
                    print(
                        f"{exp.server_config_name:<20} "
                        f"{exp.bench_config_name:<20} "
                        f"{s['scenario_name']:<25} "
                        f"{hit_pct:>10}",
                        flush=True,
                    )

        print(f"\n{'='*70}", flush=True)
        print(f"Results: {self.session_dir}", flush=True)
        print(f"Summary: {self.session_dir / 'summary.json'}", flush=True)


@dataclass
class Experiment:
    experiment_id: str
    description: str
    gpu_ids: str
    output_dir: Path
    config: dict = field(default_factory=dict)
    server_config_name: str = ""
    bench_config_name: str = ""
    sglang_info: dict = field(default_factory=dict)
    gpu_info: dict = field(default_factory=dict)
    server_cmd: str = ""
    warmup_cmd: str | None = None
    benchmark_runs: list = field(default_factory=list)
    accuracy_results: list = field(default_factory=list)
    stress_report: dict | None = None
    cache_report: dict | None = None
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"
    error: str | None = None

    @staticmethod
    def create(
        description: str,
        gpu_ids: str,
        base_dir: str,
        config: dict,
        server_config_name: str = "",
        bench_config_name: str = "",
    ) -> "Experiment":
        """Create a standalone experiment with timestamp-based ID."""
        now = datetime.now()
        experiment_id = now.strftime("%Y%m%d_%H%M%S")
        short_desc = _slugify(description)[:40]
        dir_name = f"{experiment_id}_{short_desc}" if short_desc else experiment_id
        output_dir = Path(base_dir) / dir_name

        return Experiment(
            experiment_id=experiment_id,
            description=description,
            gpu_ids=gpu_ids,
            output_dir=output_dir,
            config=config,
            server_config_name=server_config_name,
            bench_config_name=bench_config_name,
            started_at=now.isoformat(),
            status="running",
        )

    def create_directory(self) -> None:
        """Create the experiment output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def copy_configs(self, server_path: str, bench_path: str) -> None:
        """Copy the server and bench config files into the experiment directory."""
        shutil.copy2(server_path, self.output_dir / "server_config.toml")
        shutil.copy2(bench_path, self.output_dir / "bench_config.toml")

    def copy_config(self, config_path: str) -> None:
        """Copy a single combined config file into the experiment directory."""
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
            "server_config": self.server_config_name,
            "bench_config": self.bench_config_name,
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
            "accuracy_results": [
                {
                    "task": r.get("task"),
                    "command": r.get("command"),
                    "returncode": r.get("returncode"),
                }
                for r in self.accuracy_results
            ],
            "stress_report": self.stress_report,
            "cache_report": self.cache_report,
        }
        path = self.output_dir / "experiment.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def git_commit(self, auto_push: bool = False) -> None:
        """Auto-commit experiment results to the project git repo."""
        repo_root = _find_git_root(self.output_dir)
        if not repo_root:
            print("Git commit skipped: not inside a git repo.", flush=True)
            return

        try:
            rel_path = self.output_dir.resolve().relative_to(repo_root.resolve())
        except ValueError:
            print("Git commit skipped: experiment dir is outside the git repo.", flush=True)
            return

        subprocess.run(
            ["git", "add", str(rel_path)],
            cwd=str(repo_root),
            capture_output=True,
        )
        msg = f"experiment: {self.experiment_id} - {self.description}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(repo_root),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Git commit: {msg}", flush=True)
        else:
            print(f"Git commit skipped (no changes or error): {result.stderr.strip()}", flush=True)

        if auto_push:
            push_result = subprocess.run(
                ["git", "push"],
                cwd=str(repo_root),
                capture_output=True, text=True,
            )
            if push_result.returncode == 0:
                print("Git push completed.", flush=True)
            else:
                print(f"Git push failed: {push_result.stderr.strip()}", flush=True)

    def print_summary(self) -> None:
        """Print a summary of benchmark and accuracy results."""
        print("\n" + "=" * 60, flush=True)
        print(f"Experiment: {self.experiment_id}", flush=True)
        print(f"Description: {self.description}", flush=True)
        if self.server_config_name or self.bench_config_name:
            print(f"Server: {self.server_config_name}  Bench: {self.bench_config_name}", flush=True)
        print(f"Status: {self.status}", flush=True)
        print("=" * 60, flush=True)

        self._print_benchmark_summary()
        self._print_accuracy_summary()

        print("=" * 60, flush=True)
        print(f"Results saved to: {self.output_dir}", flush=True)

    def _print_accuracy_summary(self) -> None:
        """Print accuracy test results."""
        if not self.accuracy_results:
            return

        print("\n--- Accuracy Tests ---", flush=True)
        for r in self.accuracy_results:
            status = "OK" if r.get("returncode") == 0 else f"FAILED (exit {r.get('returncode')})"
            print(f"  {r.get('task')}: {status}", flush=True)
            print(f"    command: {r.get('command')}", flush=True)
        print("  Detailed logs in: accuracy_logs/", flush=True)

    def _print_benchmark_summary(self) -> None:
        """Print performance benchmark results."""
        runs = self.benchmark_runs
        if not runs:
            return

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

        print("\n--- Performance Benchmark ---", flush=True)
        if len(runs) == 1:
            results = runs[0].get("results", {})
            print(f"  seed: {runs[0].get('seed')}", flush=True)
            for key in metrics_keys:
                if key in results:
                    print(f"  {key}: {results[key]:.2f}", flush=True)
        else:
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


def _find_git_root(start_dir: Path) -> Path | None:
    """Find the nearest git repo root from start_dir upwards."""
    current = start_dir.resolve()
    for _ in range(20):
        if (current / ".git").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")
