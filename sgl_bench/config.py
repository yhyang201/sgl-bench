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
        "extra_args": "",
        "startup_timeout": 600,
    },
    "benchmark": {
        "extra_args": "",
    },
    "warmup": {
        "enabled": True,
        "num_prompts": 3,
        "seed": 8413927,
    },
    "output": {
        "dir": "./records",
        "auto_commit": True,
        "auto_push": False,
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


def load_config(path: str) -> dict:
    """Load a TOML config file, apply defaults, and validate."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config = _apply_defaults(config)
    _validate(config)
    return config
