"""Compare two experiment results side-by-side."""

import json
from pathlib import Path


COMPARE_METRICS = [
    "request_throughput",
    "output_throughput",
    "input_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
]


def load_experiment(path: str) -> dict:
    """Load experiment.json from a directory."""
    exp_path = Path(path)
    json_path = exp_path / "experiment.json" if exp_path.is_dir() else exp_path
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _avg_results(experiment: dict) -> dict:
    """Average results across all runs in an experiment."""
    runs = experiment.get("benchmark_runs", [])
    if not runs:
        return {}

    all_keys = set()
    for r in runs:
        all_keys.update(r.get("results", {}).keys())

    avg = {}
    for key in all_keys:
        values = []
        for r in runs:
            v = r.get("results", {}).get(key)
            if isinstance(v, (int, float)):
                values.append(v)
        if values:
            avg[key] = sum(values) / len(values)
    return avg


def compare_experiments(path_a: str, path_b: str) -> None:
    """Print a side-by-side comparison of two experiments."""
    exp_a = load_experiment(path_a)
    exp_b = load_experiment(path_b)
    results_a = _avg_results(exp_a)
    results_b = _avg_results(exp_b)

    name_a = exp_a.get("experiment_id", "exp_a")
    name_b = exp_b.get("experiment_id", "exp_b")

    print(f"\nComparing: {name_a} vs {name_b}")
    print(f"  A: {exp_a.get('description', '')}")
    print(f"  B: {exp_b.get('description', '')}")
    print()

    header = f"{'metric':<28} {name_a:>12} {name_b:>12} {'diff':>10}"
    print(header)
    print("-" * len(header))

    for key in COMPARE_METRICS:
        va = results_a.get(key)
        vb = results_b.get(key)
        if va is None and vb is None:
            continue

        sa = f"{va:.2f}" if va is not None else "N/A"
        sb = f"{vb:.2f}" if vb is not None else "N/A"

        if va is not None and vb is not None and va != 0:
            pct = (vb - va) / abs(va) * 100
            sdiff = f"{pct:+.1f}%"
        else:
            sdiff = "-"

        print(f"{key:<28} {sa:>12} {sb:>12} {sdiff:>10}")

    print()
