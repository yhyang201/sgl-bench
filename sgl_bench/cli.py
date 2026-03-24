"""Click CLI entry point for sgl-bench."""

import shutil
import sys
import tempfile
from pathlib import Path

import click

from .accuracy import run_accuracy_tests, validate_accuracy_config
from .benchmark import run_benchmark, run_warmup
from .compare import compare_experiments
from .config import load_toml, merge_configs
from .detect import detect_gpu_info, detect_sglang_install
from .experiment import Experiment, Session
from .server import (
    build_server_command,
    check_gpu_available,
    find_available_port,
    get_server_port,
    launch_server,
    shutdown_server,
    wait_for_server,
)

# Presets directory shipped with the package
PRESETS_DIR = Path(__file__).parent.parent / "presets"


def _resolve_paths(path_str: str, preset_subdirs: list[str]) -> list[str]:
    """Resolve a CLI path argument to a list of TOML file paths.

    Accepts:
      - A .toml file path (direct)
      - A directory path (all *.toml inside, recursive)
      - A preset name (lookup in preset_subdirs)
    """
    p = Path(path_str)

    # Direct file
    if p.is_file():
        return [str(p)]

    # Directory — collect all .toml files (recursive for subdirectories)
    if p.is_dir():
        files = sorted(p.rglob("*.toml"))
        if not files:
            raise click.UsageError(f"No .toml files found in directory: {path_str}")
        return [str(f) for f in files]

    # Preset name lookup
    for subdir in preset_subdirs:
        preset = PRESETS_DIR / subdir / f"{path_str}.toml"
        if preset.exists():
            return [str(preset)]

    # Preset subdirectory name (e.g. "perf" -> presets/perf/)
    for subdir in preset_subdirs:
        preset_dir = PRESETS_DIR / subdir
        if path_str == subdir and preset_dir.is_dir():
            files = sorted(preset_dir.rglob("*.toml"))
            if files:
                return [str(f) for f in files]

    raise click.UsageError(
        f"Cannot resolve '{path_str}': not a file, directory, or known preset.\n"
        f"Preset dirs searched: {[str(PRESETS_DIR / s) for s in preset_subdirs]}"
    )


@click.group()
def cli():
    """sgl-bench: Automated benchmark tool for SGLang VLM models."""
    pass


@cli.command()
@click.option("-s", "--server", "server_path", required=True,
              help="Server config: file, directory, or preset name.")
@click.option("-b", "--bench", "bench_path", required=True,
              help="Bench/accuracy config: file, directory, or preset name.")
@click.option("-d", "--description", required=True, help="Experiment description/purpose.")
@click.option("-g", "--gpus", default=None, help="GPU IDs to use, e.g. '0,1,2,3'. Default: auto.")
def run(server_path: str, bench_path: str, description: str, gpus: str | None):
    """Full pipeline: launch server -> warmup -> benchmark -> save.

    Supports Cartesian product: if -s and -b resolve to multiple files,
    all combinations are tested. Grouped by server to avoid restarts.
    """
    server_paths = _resolve_paths(server_path, ["server"])
    bench_paths = _resolve_paths(bench_path, ["perf", "acc"])

    total = len(server_paths) * len(bench_paths)
    print(f"Resolved {len(server_paths)} server(s) x {len(bench_paths)} bench(es) = {total} experiment(s)",
          flush=True)

    # Detect environment once
    print("Detecting environment...", flush=True)
    sglang_info = detect_sglang_install()
    gpu_info = detect_gpu_info()

    # Determine output dir from first config merge
    first_s = load_toml(server_paths[0])
    first_b = load_toml(bench_paths[0])
    base_dir = merge_configs(first_s, first_b).get("output", {}).get("dir", "./records")

    # Create session
    session = Session.create(description, base_dir)
    print(f"Session: {session.session_dir}", flush=True)

    for si, s_path in enumerate(server_paths):
        s_cfg_orig = load_toml(s_path)
        s_name = Path(s_path).stem

        for bi, b_path in enumerate(bench_paths):
            s_cfg = load_toml(s_path)  # reload fresh each time
            b_cfg = load_toml(b_path)
            b_name = Path(b_path).stem

            # Resolve GPUs
            dummy_config = merge_configs(s_cfg, {})
            gpu_ids, gpu_str = _resolve_gpus(gpus, dummy_config)

            # Find port
            desired_port = get_server_port(dummy_config)
            port = find_available_port(desired_port)
            if port != desired_port:
                print(f"Port {desired_port} is in use, switching to {port}.", flush=True)
                _override_port_in_cfg(s_cfg, port)

            config = merge_configs(s_cfg, b_cfg)
            if port != desired_port:
                config = _override_port(config, port)

            exp = session.create_experiment(config, gpu_str, s_name, b_name)
            exp.create_directory()
            exp.copy_configs(s_path, b_path)

            exp.sglang_info = sglang_info
            exp.gpu_info = gpu_info

            done = sum(1 for e in session.experiments if e.status != "running")
            print(f"\n{'='*60}", flush=True)
            print(f"[{done+1}/{total}] {s_name} x {b_name}", flush=True)
            print(f"  GPUs: {gpu_str}", flush=True)

            # Launch server per experiment, log directly into experiment dir
            check_gpu_available(gpu_ids)
            server_cmd = build_server_command(config)
            server_cmd_str = " ".join(server_cmd)
            exp.server_cmd = server_cmd_str
            exp.save_partial()
            print(f"  Command: {server_cmd_str}", flush=True)

            log_path = str(exp.output_dir / "server.log")
            server_process = launch_server(server_cmd, log_path, gpu_ids)
            try:
                timeout = s_cfg.get("server", {}).get("startup_timeout", 600)
                wait_for_server(port, timeout, server_process, log_path)

                _run_experiment(config, exp, port)
                exp.save()
            except Exception as e:
                exp.mark_failed(str(e))
                print(f"  Error: {e}", file=sys.stderr, flush=True)
                _print_server_log_tail(log_path)
            finally:
                shutdown_server(server_process)

            exp.print_summary()

    # Session-level wrap-up
    session.save_summary()

    output_cfg = merge_configs(first_s, first_b).get("output", {})
    if output_cfg.get("auto_commit", True):
        session.git_commit(auto_push=output_cfg.get("auto_push", False))

    session.print_final_summary()


def _run_experiment(config: dict, exp: Experiment, port: int) -> None:
    """Run benchmark and/or accuracy tests for a single experiment."""
    if config.get("benchmark", {}).get("extra_args", "").strip():
        warmup_cmd = run_warmup(config, str(exp.output_dir))
        exp.warmup_cmd = warmup_cmd
        exp.save_partial()

        num_runs = config.get("run", {}).get("runs", 1)
        for i in range(num_runs):
            run_data = run_benchmark(config, i, str(exp.output_dir))
            exp.benchmark_runs.append(run_data)
            exp.save_partial()

    if config.get("accuracy", {}).get("tasks"):
        validate_accuracy_config(config)
        base_url = f"http://127.0.0.1:{port}/v1"
        acc_results = run_accuracy_tests(config, base_url, str(exp.output_dir))
        exp.accuracy_results = acc_results
        exp.save_partial()


@cli.command()
@click.option("-s", "--server", "server_path", default=None,
              help="Server config (for metadata). Optional if server is external.")
@click.option("-b", "--bench", "bench_path", required=True,
              help="Bench/accuracy config: file, directory, or preset name.")
@click.option("-d", "--description", required=True, help="Experiment description/purpose.")
@click.option("-g", "--gpus", default=None, help="GPU IDs used (for metadata only). Default: all GPUs.")
@click.option("--port", default=30000, help="Port of the running server. Default: 30000.")
def bench(server_path: str | None, bench_path: str, description: str, gpus: str | None, port: int):
    """Benchmark only (server already running). No server launch/shutdown."""
    bench_paths = _resolve_paths(bench_path, ["perf", "acc"])

    s_cfg = load_toml(server_path) if server_path else {}
    _, gpu_str = _resolve_gpus(gpus, merge_configs(s_cfg, {})) if gpus else ([], "external")

    print("Detecting environment...", flush=True)
    sglang_info = detect_sglang_install()
    gpu_info = detect_gpu_info()

    base_dir = merge_configs(s_cfg, load_toml(bench_paths[0])).get("output", {}).get("dir", "./records")
    session = Session.create(description, base_dir)

    for b_path in bench_paths:
        b_cfg = load_toml(b_path)
        b_name = Path(b_path).stem
        config = merge_configs(s_cfg, b_cfg)
        config = _override_port(config, port)

        s_name = Path(server_path).stem if server_path else "external"
        exp = session.create_experiment(config, gpu_str, s_name, b_name)
        exp.create_directory()
        if server_path:
            exp.copy_configs(server_path, b_path)
        else:
            exp.copy_config(b_path)

        exp.sglang_info = sglang_info
        exp.gpu_info = gpu_info
        exp.server_cmd = "(external server)"
        exp.save_partial()
        _print_env_info(exp)

        try:
            _run_experiment(config, exp, port)
            exp.save()
        except Exception as e:
            exp.mark_failed(str(e))
            print(f"\nError: {e}", file=sys.stderr, flush=True)

        exp.print_summary()

    session.save_summary()

    output_cfg = config.get("output", {})
    if output_cfg.get("auto_commit", True):
        session.git_commit(auto_push=output_cfg.get("auto_push", False))

    if len(bench_paths) > 1:
        session.print_final_summary()


@cli.command("compare")
@click.argument("exp_a")
@click.argument("exp_b")
def compare_cmd(exp_a: str, exp_b: str):
    """Compare two experiment directories."""
    compare_experiments(exp_a, exp_b)


@cli.command("tasks")
def list_tasks():
    """List available preset tasks."""
    if not PRESETS_DIR.exists():
        print(f"Presets directory not found: {PRESETS_DIR}", flush=True)
        return

    for subdir in sorted(PRESETS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        presets = sorted(subdir.glob("*.toml"))
        if not presets:
            continue
        print(f"[{subdir.name}]", flush=True)
        for p in presets:
            desc = ""
            with open(p) as f:
                first_line = f.readline().strip()
                if first_line.startswith("#"):
                    desc = first_line.lstrip("# ")
            print(f"  {p.stem:<30} {desc}", flush=True)
        print(flush=True)


def _override_port_in_cfg(server_cfg: dict, new_port: int) -> None:
    """Replace --port in server_cfg's extra_args in-place."""
    import re
    args = server_cfg.get("server", {}).get("extra_args", "")
    if "--port" in args:
        server_cfg["server"]["extra_args"] = re.sub(
            r"--port\s+\d+", f"--port {new_port}", args
        )
    else:
        server_cfg.setdefault("server", {})
        server_cfg["server"]["extra_args"] = f"{args} --port {new_port}"


def _override_port(config: dict, new_port: int) -> dict:
    """Replace --port in server and benchmark extra_args with new_port."""
    import copy
    import re
    config = copy.deepcopy(config)
    for section in ["server", "benchmark"]:
        args = config.get(section, {}).get("extra_args", "")
        if "--port" in args:
            config[section]["extra_args"] = re.sub(
                r"--port\s+\d+", f"--port {new_port}", args
            )
    return config


def _resolve_gpus(gpus: str | None, config: dict) -> tuple[list[int], str]:
    """Resolve GPU IDs."""
    if gpus:
        try:
            ids = [int(g.strip()) for g in gpus.split(",")]
        except ValueError:
            raise click.BadParameter(f"Invalid GPU IDs: {gpus}. Expected comma-separated integers like '0,1,2,3'.")
        return ids, gpus

    from .config import extract_tp_size
    needed = extract_tp_size(config.get("server", {}).get("extra_args", ""))

    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise click.UsageError("nvidia-smi failed. Please specify -g / --gpus.")
        all_gpus = [int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip()]
    except FileNotFoundError:
        raise click.UsageError("nvidia-smi not found. Please specify -g / --gpus.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        busy_buses = set(line.strip() for line in result.stdout.strip().splitlines() if line.strip())

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        busy_ids = set()
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1] in busy_buses:
                busy_ids.add(int(parts[0]))

        free_gpus = [g for g in all_gpus if g not in busy_ids]
    except Exception:
        free_gpus = all_gpus

    if len(free_gpus) < needed:
        raise click.UsageError(
            f"Need {needed} free GPUs (from --tp-size), but only {len(free_gpus)} available: {free_gpus}.\n"
            f"Please specify -g / --gpus or free up GPUs."
        )

    picked = free_gpus[:needed]
    gpu_str = ",".join(str(i) for i in picked)
    print(f"Auto-selected GPUs: {gpu_str} (tp-size={needed})", flush=True)
    return picked, gpu_str


def _print_env_info(exp: Experiment) -> None:
    si = exp.sglang_info
    gi = exp.gpu_info

    print(f"  sglang: {si.get('install_type', 'unknown')} install, "
          f"version={si.get('version', '?')}", flush=True)
    if si.get("git_commit"):
        dirty_mark = " (DIRTY)" if si.get("git_dirty") else ""
        branch = si.get("git_branch", "?")
        print(f"  branch: {branch}, commit: {si['git_commit'][:12]}{dirty_mark}", flush=True)
        if si.get("git_diff_summary"):
            print(f"  changes: {si['git_diff_summary']}", flush=True)
        if si.get("git_dirty_warning"):
            print(f"  WARNING: {si['git_dirty_warning']}", flush=True)

    devices = gi.get("devices", [])
    gpu_name = devices[0] if devices else "unknown"
    print(f"  GPUs: {gi.get('device_count', 0)}x {gpu_name}", flush=True)
    print(f"  CUDA: {gi.get('cuda_version', '?')}, Driver: {gi.get('driver_version', '?')}", flush=True)


def _print_server_log_tail(log_path, lines=20):
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        if tail:
            print(f"\n--- Last {len(tail)} lines of server.log ---", file=sys.stderr, flush=True)
            for line in tail:
                print(line.rstrip(), file=sys.stderr, flush=True)
            print("--- End of server.log ---", file=sys.stderr, flush=True)
    except FileNotFoundError:
        pass
