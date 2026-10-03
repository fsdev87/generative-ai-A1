"""Helpers shared by the Task 2 scripts (classifier, specialists, routing).

Configuration files, the learning-rate schedule, AMP, data loaders, a cached copy of
the deterministic validation set, W&B runs that resume after a Colab disconnect,
classification metrics and the confusion-matrix figure.
"""
import math
import os
import secrets
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.common.checkpoint import load_checkpoint
from src.common.paths import get_dir
from src.data.corruptions import CLASSES

TASK = "task2"
SPECIALIST_TYPES = CLASSES[1:]  # ("salt", "blur", "occlusion")
CLASSIFIER_STUDY = "task2_classifier"
SPECIALIST_STUDY = "task2_specialists"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def study_name(base, smoke):
    """Smoke runs use their own study: create_study(fresh=True) must never delete the real one."""
    return f"{base}_smoke" if smoke else base


def best_config_path(study):
    return get_dir("OUTPUT_DIR", TASK) / f"{study}_best_config.yaml"


def build_config(defaults, path=None, overrides=None):
    """defaults <- YAML file <- overrides (None values ignored). Unknown keys are an error."""
    cfg = dict(defaults)
    layers = []
    if path:
        with open(path) as f:
            layers.append(yaml.safe_load(f) or {})
    if overrides:
        layers.append({k: v for k, v in overrides.items() if v is not None})
    for layer in layers:
        unknown = set(layer) - set(defaults)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}; expected a subset of {sorted(defaults)}")
        cfg.update(layer)
    return cfg


def plain(obj):
    """Tuples -> lists, numpy scalars -> Python, so configs round-trip through YAML/JSON."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    return path


def resolve_workers(requested, device):
    """Two workers on Colab (2 vCPUs); 0 on CPU smoke runs (no process start-up cost)."""
    if requested is not None:
        return int(requested)
    return min(2, os.cpu_count() or 1) if device.type == "cuda" else 0


def config_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if plain(a.get(k)) != plain(b.get(k))}


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def param_groups(model, weight_decay):
    """Decay conv/linear weights only; BatchNorm parameters and biases are not decayed."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    """Per-iteration schedule: linear warm-up, then cosine decay to min_lr_ratio x the peak LR."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Amp:
    """float16 autocast + GradScaler on CUDA; plain float32 elsewhere."""

    def __init__(self, device):
        self.device = device
        self.enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.enabled)

    def step(self, loss, optimizer):
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


def memory_format(device):
    # NHWC is the native layout of the T4 tensor-core convolutions used under AMP
    return torch.channels_last if device.type == "cuda" else torch.contiguous_format


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128x128 inputs: pick the fastest conv kernels once
    return device


def model_state(model):
    """state_dict on CPU in the default (NCHW-contiguous) layout, for portable checkpoints."""
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_loader(dataset, device, num_workers, batch_size=None, batch_sampler=None, shuffle=False, drop_last=False):
    kwargs = dict(num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


@torch.no_grad()
def materialize(dataset, batch_size=64, num_workers=0):
    """Apply a ManifestDataset's deterministic corruptions once and keep the tensors.

    The validation manifest never changes, so re-applying its corruptions every epoch
    (and every Optuna trial) would only cost CPU time.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    parts = {"input": [], "target": [], "label": [], "level": [], "image_idx": []}
    for batch in loader:
        parts["level"].extend(batch["level"])  # strings: collated into a list, not a tensor
        for key in ("input", "target", "label", "image_idx"):
            parts[key].append(batch[key])
    return {k: (v if k == "level" else torch.cat(v)) for k, v in parts.items()}


def batches(tensors, batch_size):
    n = len(tensors["label"])
    for start in range(0, n, batch_size):
        yield {k: v[start:start + batch_size] for k, v in tensors.items() if isinstance(v, torch.Tensor)}


# --------------------------------------------------------------------------- #
# Checkpoints and W&B
# --------------------------------------------------------------------------- #
def load_resume_state(path, smoke, tag):
    """last.pt to resume from, or None. A real run never resumes from a --smoke checkpoint.

    Smoke and real runs share the checkpoint folders, so a quick check run before the real
    training leaves a finished tiny model there; resuming from it would skip the real training.
    """
    state = load_checkpoint(path)
    if state.get("smoke", False) and not smoke:
        print(f"[{tag}] ignoring {path}: it is from a --smoke run; training the real model from scratch")
        return None
    return state


def require_real_checkpoint(path, smoke):
    """Refuse to evaluate or export a --smoke checkpoint as if it were the trained model."""
    if not smoke and load_checkpoint(path).get("smoke", False):
        raise RuntimeError(f"{path} is a --smoke checkpoint (tiny model on synthetic data), not a trained "
                           "model: run the real training first.")


def guard_smoke_overwrite(ckpt_dir, smoke):
    """Refuse to let a smoke run overwrite checkpoints of a real training run."""
    if not smoke:
        return
    for name in ("last.pt", "best.pt"):
        path = Path(ckpt_dir) / name
        if path.exists() and not load_checkpoint(path).get("smoke", False):
            raise RuntimeError(f"{path} belongs to a real training run; a --smoke run will not overwrite it. "
                               "Point CKPT_DIR at a temporary directory for smoke tests.")


@contextmanager
def tracked_run(name, group, config, run_id=None, job_type="train", tags=None):
    """W&B run with a fixed id, so a training resumed after a disconnect continues the same run.

    Same project/name/group/config as src.common.tracking.init_run, which has no id/resume
    arguments. (The WANDB_RUN_ID environment variable is not an option: wandb reads it
    once per process, so every later Optuna trial would resume the first trial's run.)
    The run is always finished, also when Optuna prunes the trial.
    """
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "genai-a1"), name=name, group=group,
                     job_type=job_type, config=plain(config), tags=tags, id=run_id or new_run_id(),
                     resume="allow")
    try:
        run.define_metric("*", step_metric="epoch")
        yield run
    finally:
        run.finish()


def new_run_id():
    return secrets.token_hex(4)


def disable_wandb_for_smoke(smoke):
    if smoke:
        os.environ["WANDB_MODE"] = "disabled"


def figure_to_array(fig):
    """Render a matplotlib figure to an RGB uint8 array (for wandb.Image) and close it."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


def log_figure(run, key, fig, caption=None):
    import wandb

    run.log({key: wandb.Image(figure_to_array(fig), caption=caption)})


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred):
    """Accuracy, macro and per-class precision/recall/F1/support, raw and row-normalised confusion."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
        "confusion_normalized": cm_norm.tolist(),
    }


def flat_metrics(metrics, prefix):
    """Scalar metrics for W&B: prefix/accuracy, prefix/f1_blur, ..."""
    out = {f"{prefix}/{k}": metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    for c, m in metrics["per_class"].items():
        out.update({f"{prefix}/{k}_{c}": m[k] for k in ("precision", "recall", "f1")})
    return out


def confusion_figure(metrics, title=""):
    """Row-normalised 4x4 confusion matrix (rows = true class) annotated with rates and counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm, norm = np.array(metrics["confusion"]), np.array(metrics["confusion_normalized"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save_figure(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
