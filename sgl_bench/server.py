"""Server lifecycle: GPU check, launch, health check, shutdown."""

import os
import re
import signal
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error

from .config import extract_port


def get_server_backend(config: dict) -> str:
    """Get server backend from config. Default: sglang."""
    return config.get("server", {}).get("backend", "sglang")


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


def find_available_port(start_port: int) -> int:
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + 99}.")


def build_server_command(config: dict) -> list[str]:
    """Build the server launch command based on backend.

    sglang: python -m sglang.launch_server --model-path {model} {extra_args}
    vllm:   python -m vllm.entrypoints.openai.api_server --model {model} {extra_args}
    """
    server = config["server"]
    backend = get_server_backend(config)
    model = server["model_path"]
    extra = server.get("extra_args", "").strip()

    if backend == "vllm":
        # vLLM uses --tensor-parallel-size instead of --tp-size
        extra = re.sub(r"--tp-size\s+(\d+)", r"--tensor-parallel-size \1", extra)
        # vLLM auto-detects multimodal, remove sglang-specific flag
        extra = re.sub(r"--enable-multimodal\s*", "", extra)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
        ]
    else:
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
        ]

    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def launch_server(cmd: list[str], log_path: str, gpu_ids: list[int],
                   extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    """Launch the server as a subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    if extra_env:
        env.update(extra_env)

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


def get_health_url(port: int, backend: str) -> str:
    """Get the health check URL for the given backend."""
    if backend == "vllm":
        return f"http://127.0.0.1:{port}/health"
    return f"http://127.0.0.1:{port}/health_generate"


def wait_for_server(port: int, timeout: int, process: subprocess.Popen, log_path: str,
                    backend: str = "sglang") -> None:
    """Poll server health endpoint until ready or timeout."""
    url = get_health_url(port, backend)
    start = time.time()
    print(f"Waiting for {backend} server on port {port} (timeout={timeout}s)...", flush=True)

    while time.time() - start < timeout:
        # Check if process died
        if process.poll() is not None:
            error_msg = _extract_log_error(log_path)
            raise RuntimeError(
                f"Server process exited with code {process.returncode}.\n"
                f"{error_msg or 'Check server.log for details.'}"
            )

        # Check server.log for ERROR lines
        error_msg = _extract_log_error(log_path)
        if error_msg:
            raise RuntimeError(f"Server error detected in log:\n{error_msg}")

        # Try health check
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


def _extract_log_error(log_path: str) -> str | None:
    """Scan server.log for ERROR lines. Returns the error message or None."""
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "ERROR:" in line:
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


def shutdown_server(process: subprocess.Popen) -> None:
    """Gracefully shutdown the server process group."""
    print("Shutting down server...", flush=True)

    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        pgid = None

    if process.poll() is not None:
        # Main process already dead but children may linger — kill the group
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _close_log(process)
        _wait_gpu_release()
        print("Server stopped (cleaned up orphan children).", flush=True)
        return

    if pgid is None:
        _close_log(process)
        print("Server stopped.", flush=True)
        return

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
    _wait_gpu_release()
    print("Server stopped.", flush=True)


def _wait_gpu_release(timeout: int = 30) -> None:
    """Wait for GPU memory to be released after server shutdown."""
    import time
    for i in range(timeout):
        time.sleep(1)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if not result.stdout.strip():
                return
        except Exception:
            return
    print(f"Warning: GPU processes still active after {timeout}s wait.", flush=True)


def _close_log(process: subprocess.Popen) -> None:
    log_file = getattr(process, "_log_file", None)
    if log_file and not log_file.closed:
        log_file.close()
