"""Task 4 configuration: defaults, YAML/CLI merging, model configs and output folders.

A training run is described by one flat dict (DEFAULTS lists every key). Precedence:
DEFAULTS < YAML file (--config) < smoke overrides (--smoke) < explicit CLI flags.
The generator and the discriminator share `base_channels` and `style_dim` (one
"base channel count" and one "style-embedding dimension" hyperparameter, as in the
brief), which keeps their capacities balanced and the search space small.
Smoke runs live under separate "smoke" subfolders so they never touch real results.
"""
import os
from pathlib import Path

import yaml

from src.common.paths import get_dir
from src.data.fs2k import MAX_ZOOM, NUM_STYLES, SKETCH_CHANNELS

TASK = "task4"
MODEL_NAME = "cgan"
STUDY_NAME = "task4_cgan"

DEFAULTS = {
    # Models (src.task4.models); G and D share base_channels and style_dim
    "base_channels": 64,
    "dropout": 0.5,
    "style_dim": 16,
    "d_layers": 3,
    # Generator loss: adversarial + lambda_l1 * L1
    "lambda_l1": 100.0,
    # Optimisation: Adam(beta1, 0.999) for G and D; constant LR, then linear decay to 0
    # over the last `decay_fraction` of the epochs (pix2pix schedule)
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "batch_size": 8,
    "epochs": 120,
    "decay_fraction": 0.5,
    # Paired augmentation (src.data.fs2k.paired_augment)
    "hflip": True,
    "max_zoom": MAX_ZOOM,
    # Run control
    "seed": 42,
    "log_every": 10,
    "sample_every": 5,
    "num_workers": 2,
    "amp": True,
}

SMOKE_OVERRIDES = {
    "base_channels": 8,
    "style_dim": 8,
    "batch_size": 4,
    "epochs": 2,
    "num_workers": 0,
    "log_every": 1,
    "sample_every": 1,
}

# Changing these on --resume does not change what is being trained
RUN_CONTROL_KEYS = ("num_workers", "log_every", "sample_every")


def load_yaml_config(path):
    """Read a YAML config; unknown keys are an error so typos do not pass silently."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}; valid keys: {sorted(DEFAULTS)}")
    return config


def resolve_config(yaml_path=None, cli=None, smoke=False):
    config = dict(DEFAULTS)
    if yaml_path:
        config.update(load_yaml_config(yaml_path))
    if smoke:
        config.update(SMOKE_OVERRIDES)
    config.update(cli or {})
    return config


def save_yaml_config(config, path, header=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    path.write_text(comment + yaml.safe_dump(config, sort_keys=False))
    return path


def generator_config(config):
    """UNetGenerator constructor arguments from a training config."""
    return {"in_channels": 3, "out_channels": SKETCH_CHANNELS, "base_channels": config["base_channels"],
            "dropout": config["dropout"], "style_dim": config["style_dim"], "num_styles": NUM_STYLES}


def discriminator_config(config):
    """PatchDiscriminator constructor arguments (same base channels and style_dim as G)."""
    return {"photo_channels": 3, "sketch_channels": SKETCH_CHANNELS, "base_channels": config["base_channels"],
            "n_layers": config["d_layers"], "style_dim": config["style_dim"], "num_styles": NUM_STYLES}


def task_dir(env, smoke=False, *parts):
    """get_dir(env, 'task4', ...); smoke runs use a separate 'smoke' subfolder."""
    return get_dir(env, TASK, *(["smoke"] if smoke else []), *parts)


def checkpoint_dir(smoke=False):
    """CKPT_DIR/task4/cgan (fixed location, docs/CONVENTIONS.md)."""
    return task_dir("CKPT_DIR", smoke, MODEL_NAME)


def output_dir(smoke=False, *parts):
    return task_dir("OUTPUT_DIR", smoke, *parts)


def best_config_path(smoke=False):
    return output_dir(smoke) / "best_config.yaml"


def study_name(smoke=False):
    return STUDY_NAME + ("_smoke" if smoke else "")


def resolve_num_workers(requested, device):
    """DataLoader workers: as requested on CUDA machines; 0 on Windows/CPU (slow process spawning)."""
    return 0 if os.name == "nt" or device.type != "cuda" else int(requested)
