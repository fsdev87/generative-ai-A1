"""Checkpoints with resume support (Colab sessions can disconnect at any time).

Training saves `last.pt` every epoch (model, optimizer, scheduler, scaler, epoch,
early-stopping state) and `best.pt` (model and metrics of the best validation epoch).
Every checkpoint stores `model_config`, so models are rebuilt as `Model(**config)`.
"""
import os
from pathlib import Path

import torch


def save_checkpoint(path, **state):
    """Write `state` atomically: a crash mid-write never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    # Our own trusted files; they contain optimizer state and plain Python objects
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model(model_cls, checkpoint, map_location="cpu"):
    """Rebuild a model from a checkpoint path (or loaded dict) with model_config and model_state."""
    if not isinstance(checkpoint, dict):
        checkpoint = load_checkpoint(checkpoint, map_location)
    model = model_cls(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model
