"""Image quality metrics and the shared validation objective for restoration models."""
import torch

from .losses import ssim

PSNR_CAP_DB = 100.0


@torch.no_grad()
def psnr(x, y, data_range=1.0):
    """Per-image PSNR in dB, shape (N,). Capped at 100 dB so identical images stay finite."""
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1)
    return (10 * torch.log10(data_range**2 / mse.clamp_min(1e-30))).clamp_max(PSNR_CAP_DB)


@torch.no_grad()
def image_metrics(output, target):
    """Per-image PSNR, SSIM and MSE as CPU tensors of shape (N,)."""
    return {
        "psnr": psnr(output, target).cpu(),
        "ssim": ssim(output, target, reduction="none").cpu(),
        "mse": (output.float() - target.float()).pow(2).flatten(1).mean(1).cpu(),
    }


def restoration_score(psnr_value, ssim_value):
    """Shared validation objective for restoration models (higher is better).

    Equal weight on structure (SSIM) and pixel fidelity (PSNR divided by 40 dB to bring
    it to a comparable 0-1 range). It does not depend on the loss weights being tuned,
    so Optuna trials with different alpha are compared on the same scale.
    """
    return 0.5 * ssim_value + 0.5 * psnr_value / 40.0
