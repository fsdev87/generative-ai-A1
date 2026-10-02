"""Weights & Biases helpers. Set WANDB_MODE=disabled for offline smoke tests."""
import os
from contextlib import contextmanager

import numpy as np


def init_run(name, group, config=None, job_type="train", tags=None):
    import wandb

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "genai-a1"),
        name=name,
        group=group,
        job_type=job_type,
        config=config or {},
        tags=tags,
    )


@contextmanager
def wandb_run(name, group, config=None, job_type="train", tags=None):
    """A W&B run that is always finished, also when Optuna prunes the trial."""
    run = init_run(name, group, config, job_type, tags)
    try:
        yield run
    finally:
        run.finish()


def image_grid(rows, pad=2):
    """Tile rows of equally sized HWC float images in [0, 1] into one uint8 image."""
    rows = [[np.clip(np.asarray(img, dtype=np.float32), 0, 1) for img in row] for row in rows]
    h, w = rows[0][0].shape[:2]
    n_cols = max(len(row) for row in rows)
    grid = np.ones((len(rows) * (h + pad) + pad, n_cols * (w + pad) + pad, 3), dtype=np.float32)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            y, x = pad + r * (h + pad), pad + c * (w + pad)
            grid[y:y + h, x:x + w] = img[..., :3]
    return (grid * 255).round().astype(np.uint8)


def log_image_grid(run, key, rows, caption=None, step=None):
    import wandb

    run.log({key: wandb.Image(image_grid(rows), caption=caption)}, step=step)


def log_artifact(run, path, name, artifact_type="model", aliases=None, metadata=None):
    """Upload a file (e.g. best.pt) as a versioned W&B artifact."""
    import wandb

    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=aliases)
