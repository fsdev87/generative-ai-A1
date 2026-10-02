"""Losses for the restoration models (Tasks 1-3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size, sigma, channels, device):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, size, size).contiguous()


def ssim(x, y, data_range=1.0, window_size=11, sigma=1.5, k1=0.01, k2=0.03, reduction="mean"):
    """Structural similarity (Wang et al., 2004) between image batches x and y (N, C, H, W).

    Gaussian 11x11 window (sigma 1.5), averaged over channels and over the valid region
    (windows fully inside the image), as in Wang et al.'s reference code and scikit-image.
    torchmetrics instead reflect-pads and includes the border windows, which gives
    slightly different values (about 3e-5 on 64x64 test images). Runs in float32 with autocast disabled:
    the variance terms E[x^2] - E[x]^2 lose all precision in float16.
    Returns the batch mean, or per-image values (N,) with reduction="none".
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        c = x.shape[1]
        w = _gaussian_window(window_size, sigma, c, x.device)
        mu_x = F.conv2d(x, w, groups=c)
        mu_y = F.conv2d(y, w, groups=c)
        var_x = F.conv2d(x * x, w, groups=c) - mu_x**2
        var_y = F.conv2d(y * y, w, groups=c) - mu_y**2
        cov = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
        c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
        per_image = ssim_map.flatten(1).mean(1)
    return per_image if reduction == "none" else per_image.mean()


class RestorationLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM): the reconstruction loss of Tasks 1 and 2.

    Returns (loss, parts) where parts holds the detached L1 and SSIM values for logging.
    """

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l1 = F.l1_loss(output.float(), target.float())
        s = ssim(output, target)
        loss = self.alpha * l1 + (1 - self.alpha) * (1 - s)
        return loss, {"l1": l1.detach(), "ssim": s.detach()}
