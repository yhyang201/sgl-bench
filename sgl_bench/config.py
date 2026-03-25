"""TOML configuration loading and validation."""

import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python < 3.11 requires 'tomli' package: pip install tomli")


DEFAULTS = {
    "server": {
        "backend": "sglang",
        "extra_args": "",
        "startup_timeout": 600,
    },
    "benchmark": {
        "extra_args": "",
    },
    "warmup": {
        "num_prompts": 3,
        "seed": 8413927,
    },
    "accuracy": {
        "api_key": "empty",
    },
    "output": {
        "dir": "./records",
        "auto_commit": True,
        "auto_push": True,
    },
    "run": {
        "runs": 1,
    },
}


def _apply_defaults(config: dict) -> dict:
    """Apply default values to missing fields."""
    for section, defaults in DEFAULTS.items():
        if section not in config:
            config[section] = {}
        for key, value in defaults.items():
            config[section].setdefault(key, value)

    # warmup.enabled: auto-detect from whether benchmark has extra_args
    if "enabled" not in config["warmup"]:
        has_bench = bool(config.get("benchmark", {}).get("extra_args", "").strip())
        config["warmup"]["enabled"] = has_bench

    return config


def _validate(config: dict) -> None:
    """Validate required fields exist."""
    if not config.get("server", {}).get("model_path"):
        raise ValueError("Missing required config: server.model_path")


def extract_port(extra_args: str, default: int = 30000) -> int:
    """Extract --port value from an extra_args string."""
    match = re.search(r"--port\s+(\d+)", extra_args)
    if match:
        return int(match.group(1))
    return default


def extract_tp_size(extra_args: str, default: int = 1) -> int:
    """Extract --tp-size value from an extra_args string."""
    match = re.search(r"--tp-size\s+(\d+)", extra_args)
    if match:
        return int(match.group(1))
    return default


def load_toml(path: str) -> dict:
    """Load a raw TOML file without defaults or validation."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def merge_configs(server_cfg: dict, bench_cfg: dict) -> dict:
    """Merge a server config and a bench/accuracy config into a full config.

    server_cfg provides [server] (and optionally [warmup], [output], [run]).
    bench_cfg provides [benchmark] or [accuracy] (and optionally [warmup], [output], [run]).
    bench_cfg values take precedence for shared sections.
    """
    merged = {}

    # Server section comes from server_cfg only
    merged["server"] = dict(server_cfg.get("server", {}))

    # Benchmark / accuracy / stress come from bench_cfg only
    for section in ["benchmark", "accuracy", "stress"]:
        if section in bench_cfg:
            merged[section] = dict(bench_cfg[section])

    # For warmup, output, run: bench_cfg overrides server_cfg
    for section in ["warmup", "output", "run"]:
        base = dict(server_cfg.get(section, {}))
        override = bench_cfg.get(section, {})
        base.update(override)
        if base:
            merged[section] = base

    merged = _apply_defaults(merged)
    _validate(merged)
    return merged


def load_config(path: str) -> dict:
    """Load a TOML config file, apply defaults, and validate.

    Each TOML file is fully self-contained — no inheritance.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config = _apply_defaults(config)
    _validate(config)
    return config
