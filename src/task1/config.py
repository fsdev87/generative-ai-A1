"""Task 1 configuration: defaults, YAML/CLI merging, variant names and output folders.

A training run is described by one flat dict (DEFAULTS lists every key). Precedence:
DEFAULTS < YAML file (--config) < smoke overrides (--smoke) < explicit CLI flags.

Variants: the main model has no skip connections and is called "udae"; ablation
variants with limited skips get their own name (e.g. "udae_skip16"), checkpoint
folder and output folder, so they never overwrite the main model. Smoke runs live
under a separate "smoke" subfolder for the same reason.
"""
import os
from pathlib import Path

import yaml

from src.common.paths import get_dir

TASK = "task1"
MAIN_VARIANT = "udae"
STUDY_NAME = "task1_udae"
IMAGE_SIZE = 128

DEFAULTS = {
    # Model (src.models.autoencoder.ConvAutoencoder); the latent is latent_channels x 8 x 8 at depth 4
    "base_channels": 32,
    "latent_channels": 32,
    "depth": 4,
    "dropout": 0.0,
    "skip_resolutions": [],
    # Loss: alpha * L1 + (1 - alpha) * (1 - SSIM)
    "alpha": 0.8,
    # Optimisation: AdamW, linear warm-up then cosine decay to min_lr_ratio * lr (per step)
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 60,
    "warmup_epochs": 1,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "patience": 12,
    # Run control
    "seed": 42,
    "sample_every": 5,
    "num_workers": 2,
    "amp": True,
}

SMOKE_OVERRIDES = {
    "base_channels": 4,
    "latent_channels": 8,
    "batch_size": 8,
    "epochs": 2,
    "num_workers": 0,
    "sample_every": 1,
}

MODEL_KEYS = ("base_channels", "depth", "latent_channels", "dropout", "skip_resolutions")


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
    config["skip_resolutions"] = sorted(int(r) for r in config["skip_resolutions"])
    return config


def save_yaml_config(config, path, header=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    path.write_text(comment + yaml.safe_dump(config, sort_keys=False))
    return path


def model_config(config):
    """The ConvAutoencoder constructor arguments from a training config."""
    return {k: (tuple(config[k]) if k == "skip_resolutions" else config[k]) for k in MODEL_KEYS} | {
        "image_size": IMAGE_SIZE}


def variant_name(skip_resolutions):
    """'udae' for the main model, 'udae_skip16' (etc.) for skip-connection ablation variants."""
    skips = sorted(int(r) for r in skip_resolutions)
    return MAIN_VARIANT if not skips else f"{MAIN_VARIANT}_skip" + "_".join(map(str, skips))


def task_dir(env, smoke=False, *parts):
    """get_dir(env, 'task1', ...); smoke runs use a separate 'smoke' subfolder."""
    return get_dir(env, TASK, *(["smoke"] if smoke else []), *parts)


def checkpoint_dir(variant=MAIN_VARIANT, smoke=False):
    return task_dir("CKPT_DIR", smoke, variant)


def output_dir(variant=MAIN_VARIANT, smoke=False):
    """OUTPUT_DIR/task1 for the main model (fixed locations used by other tasks), else a variant subfolder."""
    if variant == MAIN_VARIANT:
        return task_dir("OUTPUT_DIR", smoke)
    return task_dir("OUTPUT_DIR", smoke, "variants", variant)


def best_config_path(smoke=False):
    return task_dir("OUTPUT_DIR", smoke) / "best_config.yaml"


def study_name(smoke=False):
    return STUDY_NAME + ("_smoke" if smoke else "")


def resolve_num_workers(requested, device):
    """DataLoader workers: as requested on CUDA machines; 0 on Windows (slow process spawning)."""
    return 0 if os.name == "nt" or device.type != "cuda" else int(requested)


def channel_widths(base_channels, depth):
    """Encoder channels per resolution level, as built by ConvAutoencoder."""
    return [base_channels * min(2**i, 8) for i in range(depth + 1)]


def bottleneck_info(mcfg):
    """Latent size, compression ratio and how many values the skip connections carry per image."""
    depth, size = mcfg["depth"], mcfg.get("image_size", IMAGE_SIZE)
    latent_side = size // 2**depth
    latent_dim = mcfg["latent_channels"] * latent_side**2
    input_dim = 3 * size * size
    chans = channel_widths(mcfg["base_channels"], depth)
    skip_values = sum(chans[i] * (size // 2**i) ** 2 for i in range(depth) if size // 2**i in mcfg["skip_resolutions"])
    return {
        "latent_shape": [mcfg["latent_channels"], latent_side, latent_side],
        "latent_dim": latent_dim,
        "compression_ratio": input_dim / latent_dim,
        "skip_values": skip_values,
        "skip_values_vs_input": skip_values / input_dim,
    }
