"""Task 3 configuration: defaults, YAML/CLI merging and output folders.

A run is described by one flat dict (DEFAULTS lists every key). Precedence:
DEFAULTS < YAML file (--config) < smoke overrides (--smoke) < explicit CLI flags.
Smoke runs write under a separate "smoke" subfolder and Optuna study, so they can never
overwrite (or be resumed into) the real model, even when run on Colab against Drive.
"""
import os
from pathlib import Path

import numpy as np
import yaml

from src.common.paths import get_dir

TASK = "task3"
MODEL_NAME = "moe"
STUDY_NAME = "task3_moe"
FINAL_GROUP = "task3-final"
OPTUNA_GROUP = "task3-optuna"

DEFAULTS = {
    # Routing: w = softmax(G(x) / tau); tau is fixed per run (a buffer baked into the ONNX graph)
    "tau": 1.0,
    # Loss (brief's starting values): l1 * L1 + ssim * (1 - SSIM) + ce * CE + balance * L_balance
    "lambda_l1": 0.8,
    "lambda_ssim": 0.2,
    "lambda_ce": 0.1,
    "lambda_balance": 0.01,
    "ce_input": "logits",        # CE on the raw gate logits G(x) ("routing": on G(x) / tau); see notes
    # Stage 1, warm-up: experts frozen (no gradients, eval-mode BatchNorm), only the gate trains
    "warmup_epochs": 2,
    "lr_warmup": 3e-4,           # gate learning rate in the warm-up (constant)
    # Stage 2, joint fine-tuning: gate at lr, experts at lr * expert_lr_scale, cosine decay per step
    "joint_epochs": 20,
    "lr": 1e-4,                  # searched by Optuna; the search range never exceeds lr_warmup
    "expert_lr_scale": 0.1,
    "min_lr_ratio": 0.01,
    "freeze_expert_bn": True,    # experts keep their Task 2 BatchNorm statistics in the joint stage
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "batch_size": 32,            # multiple of 4: BalancedBatchSampler gives 8 images per condition
    "patience": 6,               # early stopping (joint epochs without a better val score)
    # Routing-collapse criterion on the validation set (see routing.detect_collapse)
    "collapse_min_usage": 0.05,
    "collapse_max_off_class": 0.5,
    # Run control
    "seed": 42,
    "amp": True,
    "channels_last": True,
    "num_workers": None,         # None: 2 on CUDA (Colab has 2 vCPUs), 0 on CPU / Windows
    "sample_every": 2,           # epochs between W&B validation image grids
}

SMOKE_OVERRIDES = {
    "warmup_epochs": 1,
    "joint_epochs": 1,
    "batch_size": 8,
    "num_workers": 0,
    "sample_every": 1,
}

# Changing these on --resume does not change the trained model, so no warning is printed
RUN_CONTROL_KEYS = ("num_workers", "sample_every")


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
    config.update({k: v for k, v in (cli or {}).items() if v is not None})
    validate_config(config)
    return config


def validate_config(config):
    if config["batch_size"] % 4:
        raise ValueError(f"batch_size must be a multiple of 4 (balanced batches), got {config['batch_size']}")
    if config["ce_input"] not in ("logits", "routing"):
        raise ValueError(f"ce_input must be 'logits' or 'routing', got {config['ce_input']!r}")
    if config["tau"] <= 0:
        raise ValueError("tau must be positive")
    if config["warmup_epochs"] < 0 or config["joint_epochs"] < 0 or config["warmup_epochs"] + config["joint_epochs"] < 1:
        raise ValueError("need at least one epoch")


def plain(obj):
    """Tuples -> lists, numpy scalars -> Python, so configs round-trip through YAML/JSON."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_yaml_config(config, path, header=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    path.write_text(comment + yaml.safe_dump(plain(config), sort_keys=False))
    return path


def task_dir(env, smoke=False, *parts):
    """get_dir(env, 'task3', ...); smoke runs use a separate 'smoke' subfolder."""
    return get_dir(env, TASK, *(["smoke"] if smoke else []), *parts)


def checkpoint_dir(smoke=False):
    return task_dir("CKPT_DIR", smoke, MODEL_NAME)


def output_dir(smoke=False, *parts):
    return task_dir("OUTPUT_DIR", smoke, *parts)


def onnx_path(smoke=False):
    return (get_dir("ONNX_DIR", "smoke") if smoke else get_dir("ONNX_DIR")) / f"{MODEL_NAME}.onnx"


def best_config_path(smoke=False):
    return output_dir(smoke) / "best_config.yaml"


def study_name(smoke=False):
    return STUDY_NAME + ("_smoke" if smoke else "")


def resolve_num_workers(requested, device):
    """DataLoader workers: 2 on CUDA machines (Colab), 0 on CPU or Windows (slow process spawning)."""
    if os.name == "nt" or device.type != "cuda":
        return 0
    return 2 if requested is None else int(requested)
