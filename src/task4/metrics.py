"""Task 4 sketch metrics and the validation objective shared by training, Optuna and evaluation.

Generated and ground-truth sketches live in [-1, 1] (the generator ends in tanh); every
metric here converts to [0, 1] first, so L1 and PSNR use data range 1 like Tasks 1-3.
"""
import torch

from src.common.losses import ssim
from src.common.metrics import psnr


def to_unit(x):
    """[-1, 1] -> [0, 1], clamped (a tanh output can leave the range by a rounding error)."""
    return ((x.float() + 1.0) / 2.0).clamp(0, 1)


def edge_energy(x):
    """Mean absolute finite-difference gradient per image, shape (N,).

    Sketches are line drawings, so this is essentially how much ink edge the image has.
    The ratio generated/ground-truth detects the blur a large L1 weight produces: a
    ratio well below 1 means the model draws soft grey strokes instead of sharp lines.
    """
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().flatten(1).mean(1)
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().flatten(1).mean(1)
    return 0.5 * (dx + dy)


@torch.no_grad()
def sketch_metrics(output, target):
    """Per-image metrics for sketches in [-1, 1]; returns CPU tensors of shape (N,).

    l1 and psnr are pixel fidelity, ssim is structure, edge_ratio is the sharpness
    diagnostic (1 = as much edge energy as the ground truth).
    """
    out, tgt = to_unit(output), to_unit(target)
    tgt_edges = edge_energy(tgt)
    return {
        "l1": (out - tgt).abs().flatten(1).mean(1).cpu(),
        "ssim": ssim(out, tgt, reduction="none").cpu(),
        "psnr": psnr(out, tgt).cpu(),
        "edge_ratio": (edge_energy(out) / tgt_edges.clamp_min(1e-6)).cpu(),
    }


def sketch_score(ssim_value, l1_value):
    """Validation objective for model selection and Optuna (higher is better).

    Equal weight on structure (SSIM) and pixel fidelity (1 - L1), both already in [0, 1].
    Like `restoration_score` for Tasks 1-3 it does not involve the adversarial loss or
    lambda_l1, so trials with different loss weights and different discriminators are
    compared on the same scale. Its limitation (reconstruction metrics mildly prefer the
    blurry L1-only optimum) is why edge_ratio is logged next to it and why lambda_l1 is
    searched in a bounded range; see docs/fs2k_notes.md.
    """
    return 0.5 * ssim_value + 0.5 * (1.0 - l1_value)
