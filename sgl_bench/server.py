"""Server lifecycle: GPU check, launch, health check, shutdown."""

import os
import signal
import shlex
import subprocess
import time
import urllib.request
import urllib.error

from .config import extract_port


def check_gpu_available(gpu_ids: list[int]) -> None:
    """Check that specified GPUs have no other compute processes.

    Raises RuntimeError with details if any GPU is occupied.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_bus_id,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Cannot check GPU availability.")

    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    # Get GPU bus IDs for specified gpu_ids
    try:
        idx_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        raise RuntimeError("Failed to query GPU indices.")

    gpu_bus_ids = {}
    for line in idx_result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpu_bus_ids[int(parts[0])] = parts[1]

    target_bus_ids = set()
    for gid in gpu_ids:
        if gid not in gpu_bus_ids:
            raise RuntimeError(f"GPU {gid} not found. Available GPUs: {list(gpu_bus_ids.keys())}")
        target_bus_ids.add(gpu_bus_ids[gid])

    # Check if any compute process is running on target GPUs
    conflicts = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            pid, bus_id, proc_name, mem = parts[0], parts[1], parts[2], parts[3]
            if bus_id in target_bus_ids:
                conflicts.append(f"  GPU {bus_id}: PID={pid}, process={proc_name}, memory={mem}")

    if conflicts:
        detail = "\n".join(conflicts)
        raise RuntimeError(
            f"The following GPUs have active compute processes:\n{detail}\n"
            f"Please free these GPUs before launching the server."
        )


def build_server_command(config: dict) -> list[str]:
    """Build the full sglang launch_server command from config.

    Command: python -m sglang.launch_server --model-path {model_path} {extra_args}
    """
    server = config["server"]
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", server["model_path"],
    ]
    extra = server.get("extra_args", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def launch_server(cmd: list[str], log_path: str, gpu_ids: list[int]) -> subprocess.Popen:
    """Launch the sglang server as a subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    log_file = open(log_path, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    process._log_file = log_file
    return process


def get_server_port(config: dict) -> int:
    """Get port from server extra_args, default 30000."""
    return extract_port(config["server"].get("extra_args", ""))


def wait_for_server(port: int, timeout: int, process: subprocess.Popen) -> None:
    """Poll server health endpoint until ready or timeout."""
    url = f"http://127.0.0.1:{port}/health_generate"
    start = time.time()
    print(f"Waiting for server on port {port} (timeout={timeout}s)...", flush=True)

    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server process exited with code {process.returncode} before becoming ready. "
                f"Check server.log for details."
            )
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"Server ready! (took {elapsed}s)", flush=True)
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            pass
        time.sleep(5)

    raise RuntimeError(f"Server did not become ready within {timeout}s. Check server.log for details.")


def shutdown_server(process: subprocess.Popen) -> None:
    """Gracefully shutdown the server process group."""
    if process.poll() is not None:
        _close_log(process)
        return

    pgid = os.getpgid(process.pid)
    print("Shutting down server...", flush=True)

    try:
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server did not stop gracefully, sending SIGKILL...", flush=True)
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=5)
    except ProcessLookupError:
        pass

    _close_log(process)
    print("Server stopped.", flush=True)


def _close_log(process: subprocess.Popen) -> None:
    log_file = getattr(process, "_log_file", None)
    if log_file and not log_file.closed:
        log_file.close()
