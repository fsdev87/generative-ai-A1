"""Directory configuration shared by all scripts.

Every directory comes from an environment variable with a default inside the
repository, so the same code runs locally and on Colab (where the notebooks
point the variables at Google Drive). See docs/CONVENTIONS.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "manifests"

_DEFAULTS = {
    "DATA_DIR": ROOT / "data",
    "CACHE_DIR": ROOT / "data" / "cache",
    "CKPT_DIR": ROOT / "outputs" / "checkpoints",
    "OPTUNA_DIR": ROOT / "outputs" / "optuna",
    "ONNX_DIR": ROOT / "outputs" / "onnx",
    "OUTPUT_DIR": ROOT / "outputs",
}


def get_dir(env, *parts):
    """Return (and create) the directory for `env`, optionally with subfolders."""
    if env not in _DEFAULTS:
        raise KeyError(f"unknown directory variable {env!r}; expected one of {sorted(_DEFAULTS)}")
    path = Path(os.environ.get(env) or _DEFAULTS[env]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
