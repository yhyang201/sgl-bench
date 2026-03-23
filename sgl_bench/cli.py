"""Click CLI entry point for sgl-bench."""

import sys

import click

from .benchmark import run_benchmark, run_warmup
from .compare import compare_experiments
from .config import load_config
from .detect import detect_gpu_info, detect_sglang_install
from .experiment import Experiment
from .server import (
    build_server_command,
    check_gpu_available,
    get_server_port,
    launch_server,
    shutdown_server,
    wait_for_server,
)


@click.group()
def cli():
    """sgl-bench: Automated benchmark tool for SGLang VLM models."""
    pass


@cli.command()
@click.option("-c", "--config", "config_path", required=True, help="TOML config file path.")
@click.option("-d", "--description", required=True, help="Experiment description/purpose.")
@click.option("-g", "--gpus", required=True, help="GPU IDs to use, e.g. '0,1,2,3'.")
def run(config_path: str, description: str, gpus: str):
    """Full pipeline: launch server -> warmup -> benchmark -> save."""
    config = load_config(config_path)
    gpu_ids = _parse_gpu_ids(gpus)

    output_dir = config.get("output", {}).get("dir", "./records")
    exp = Experiment.create(description, gpus, output_dir, config)
    exp.create_directory()
    exp.copy_config(config_path)

    server_process = None
    try:
        print("Detecting environment...", flush=True)
        exp.sglang_info = detect_sglang_install()
        exp.gpu_info = detect_gpu_info()
        exp.save_partial()
        _print_env_info(exp)

        check_gpu_available(gpu_ids)
        print(f"GPUs {gpus} are available.", flush=True)

        server_cmd = build_server_command(config)
        exp.server_cmd = " ".join(server_cmd)
        exp.save_partial()

        print(f"Launching server: {exp.server_cmd}", flush=True)
        print(f"  CUDA_VISIBLE_DEVICES={gpus}", flush=True)

        log_path = str(exp.output_dir / "server.log")
        server_process = launch_server(server_cmd, log_path, gpu_ids)

        port = get_server_port(config)
        timeout = config["server"].get("startup_timeout", 600)
        wait_for_server(port, timeout, server_process)

        warmup_cmd = run_warmup(config, str(exp.output_dir))
        exp.warmup_cmd = warmup_cmd
        exp.save_partial()

        num_runs = config.get("run", {}).get("runs", 1)
        for i in range(num_runs):
            run_data = run_benchmark(config, i, str(exp.output_dir))
            exp.benchmark_runs.append(run_data)
            exp.save_partial()

        exp.save()

    except Exception as e:
        exp.mark_failed(str(e))
        print(f"\nError: {e}", file=sys.stderr, flush=True)
        _print_server_log_tail(exp.output_dir / "server.log")
        sys.exit(1)

    finally:
        if server_process is not None:
            shutdown_server(server_process)

    output_cfg = config.get("output", {})
    if output_cfg.get("auto_commit", True):
        exp.git_commit(auto_push=output_cfg.get("auto_push", False))

    exp.print_summary()


@cli.command()
@click.option("-c", "--config", "config_path", required=True, help="TOML config file path.")
@click.option("-d", "--description", required=True, help="Experiment description/purpose.")
@click.option("-g", "--gpus", required=True, help="GPU IDs used (for metadata only, server already running).")
def bench(config_path: str, description: str, gpus: str):
    """Benchmark only (server already running). No server launch/shutdown."""
    config = load_config(config_path)

    output_dir = config.get("output", {}).get("dir", "./records")
    exp = Experiment.create(description, gpus, output_dir, config)
    exp.create_directory()
    exp.copy_config(config_path)

    try:
        print("Detecting environment...", flush=True)
        exp.sglang_info = detect_sglang_install()
        exp.gpu_info = detect_gpu_info()
        exp.server_cmd = "(external server)"
        exp.save_partial()
        _print_env_info(exp)

        warmup_cmd = run_warmup(config, str(exp.output_dir))
        exp.warmup_cmd = warmup_cmd
        exp.save_partial()

        num_runs = config.get("run", {}).get("runs", 1)
        for i in range(num_runs):
            run_data = run_benchmark(config, i, str(exp.output_dir))
            exp.benchmark_runs.append(run_data)
            exp.save_partial()

        exp.save()

    except Exception as e:
        exp.mark_failed(str(e))
        print(f"\nError: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    output_cfg = config.get("output", {})
    if output_cfg.get("auto_commit", True):
        exp.git_commit(auto_push=output_cfg.get("auto_push", False))

    exp.print_summary()


@cli.command("compare")
@click.argument("exp_a")
@click.argument("exp_b")
def compare_cmd(exp_a: str, exp_b: str):
    """Compare two experiment directories."""
    compare_experiments(exp_a, exp_b)


def _parse_gpu_ids(gpus: str) -> list[int]:
    try:
        return [int(g.strip()) for g in gpus.split(",")]
    except ValueError:
        raise click.BadParameter(f"Invalid GPU IDs: {gpus}. Expected comma-separated integers like '0,1,2,3'.")


def _print_env_info(exp: Experiment) -> None:
    si = exp.sglang_info
    gi = exp.gpu_info

    print(f"  sglang: {si.get('install_type', 'unknown')} install, "
          f"version={si.get('version', '?')}", flush=True)
    if si.get("git_commit"):
        dirty_mark = " (DIRTY)" if si.get("git_dirty") else ""
        print(f"  commit: {si['git_commit'][:12]}{dirty_mark}", flush=True)
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
