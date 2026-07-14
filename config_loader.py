"""
config_loader.py
-----------------
Single source of truth for reading config/config.yaml.
Every other module calls get_config() instead of parsing YAML itself,
so there's one place to change if the config format ever changes.
"""

from pathlib import Path
import yaml

# Project root = two levels up from this file (src/utils/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_cached_config = None


def get_config() -> dict:
    """Load and cache config.yaml. Returns a plain dict."""
    global _cached_config
    if _cached_config is None:
        with open(CONFIG_PATH, "r") as f:
            _cached_config = yaml.safe_load(f)
    return _cached_config


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to project root) into an absolute Path,
    creating parent directories if they don't exist."""
    full_path = PROJECT_ROOT / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


if __name__ == "__main__":
    cfg = get_config()
    print("Loaded config for project:", cfg["project"]["name"])
    print("Random seed:", cfg["project"]["random_seed"])
