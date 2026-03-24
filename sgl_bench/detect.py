"""Detect sglang installation type, version, git state, and GPU info."""

import json
import os
import subprocess
import sys


def detect_sglang_install() -> dict:
    """Detect how sglang is installed and gather version/git info.

    Returns a dict with keys:
      install_type: "editable" | "pip" | "not_found"
      version, location, git_commit, git_dirty, git_diff_summary
    """
    info = {"install_type": "not_found"}

    # Step 1: pip show sglang
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "sglang"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return info

    if result.returncode != 0:
        return info

    fields = {}
    for line in result.stdout.strip().splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            fields[key.strip().lower()] = value.strip()

    info["version"] = fields.get("version", "unknown")

    # Step 2: Determine actual source location via sglang.__file__
    # pip show location can report site-packages even for editable installs
    try:
        file_result = subprocess.run(
            [sys.executable, "-c", "import sglang; print(sglang.__file__)"],
            capture_output=True, text=True, timeout=10,
        )
        if file_result.returncode == 0:
            source_file = file_result.stdout.strip()
            location = os.path.dirname(os.path.dirname(source_file))  # up from sglang/__init__.py
        else:
            location = fields.get("location", "")
    except Exception:
        location = fields.get("location", "")

    info["location"] = location

    # Step 3: Check if source is in a git repo (editable) or plain pip install
    repo_root = _find_git_root(location)
    if not repo_root:
        info["install_type"] = "pip"
        return info

    info["install_type"] = "editable"
    if not repo_root:
        return info

    # Step 4: Get git branch and commit hash
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info["git_branch"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    # Step 5: Check for uncommitted changes
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            dirty = bool(result.stdout.strip())
            info["git_dirty"] = dirty
            if dirty:
                # Get diff summary
                diff_result = subprocess.run(
                    ["git", "-C", repo_root, "diff", "--stat"],
                    capture_output=True, text=True, timeout=10,
                )
                if diff_result.returncode == 0 and diff_result.stdout.strip():
                    # Last line of git diff --stat is the summary
                    lines = diff_result.stdout.strip().splitlines()
                    info["git_diff_summary"] = lines[-1].strip() if lines else ""
                    info["git_dirty_warning"] = (
                        "Working tree has uncommitted changes, not exactly this commit"
                    )
    except Exception:
        pass

    return info


def _find_git_root(path: str) -> str | None:
    """Walk up from path to find a .git directory."""
    current = os.path.abspath(path)
    for _ in range(20):  # safety limit
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def detect_gpu_info() -> dict:
    """Detect GPU information via a subprocess (avoids importing torch here).

    Returns a dict with keys:
      device_count, devices, driver_version, cuda_version
    """
    # Use a small Python script to gather GPU info via torch
    script = """
import json, sys
try:
    import torch
    info = {
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_version": torch.version.cuda or "unknown",
    }
except Exception as e:
    info = {"error": str(e), "device_count": 0, "devices": [], "cuda_version": "unknown"}
print(json.dumps(info))
"""
    gpu_info = {"device_count": 0, "devices": [], "cuda_version": "unknown"}

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info.update(json.loads(result.stdout.strip()))
    except Exception:
        pass

    # Get driver version from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            versions = result.stdout.strip().splitlines()
            if versions:
                gpu_info["driver_version"] = versions[0].strip()
    except Exception:
        pass

    return gpu_info
