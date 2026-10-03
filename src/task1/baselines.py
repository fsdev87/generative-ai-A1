"""Reference baselines for Task 1, evaluated on the same test manifest as the model.

identity: the corrupted input itself (no restoration), i.e. how bad each corruption is.

oracle classical: classical filters that are TOLD the corruption, information the
autoencoder never receives (it sees only the corrupted image):
  - salt-and-pepper: 3x3 median filter (the textbook impulse-noise filter);
  - Gaussian blur: unsharp masking with the *known* blur kernel (k, sigma) and amount 1,
    x_hat = y + (y - h*y). This is one step of Van Cittert's iterative deconvolution: the
    error spectrum (1 - H) X of the blurred image becomes (1 - H)^2 X, smaller wherever
    0 < H < 1. Clipping to [0, 1] can only move pixels closer to the (in-range) target;
  - occlusion: OpenCV Telea inpainting (fast-marching) of the *known* rectangle mask;
  - clean: returned unchanged (the oracle knows nothing needs fixing).
"""
import cv2
import numpy as np
import torch

from src.data.corruptions import gaussian_blur
from src.data.pets import ManifestDataset

INPAINT_RADIUS = 3
cv2.setNumThreads(0)  # filters run inside DataLoader workers; avoid nested thread pools


def median_filter(img):
    return cv2.medianBlur(np.ascontiguousarray(img, dtype=np.float32), 3)


def unsharp_mask(img, k, sigma, amount=1.0):
    blurred = gaussian_blur(img, k, sigma)  # exactly the kernel and padding used to corrupt the image
    return np.clip(img + amount * (img - blurred), 0.0, 1.0).astype(np.float32)


def occlusion_mask(rects, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    for y, x, h, w in rects:
        mask[y:y + h, x:x + w] = 255
    return mask


def telea_inpaint(img, rects, radius=INPAINT_RADIUS):
    mask = occlusion_mask(rects, img.shape[0])
    # cv2.inpaint needs 8-bit colour images; the visible pixels come from the uint8 cache, so this is lossless
    u8 = np.round(img * 255).astype(np.uint8)
    filled = cv2.inpaint(u8, mask, radius, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    return np.where(mask[..., None] > 0, filled, img).astype(np.float32)


def oracle_restore(img, spec):
    """Restore a corrupted float32 (H, W, 3) image using the known corruption spec."""
    t = spec["type"]
    if t == "clean":
        return img.copy()
    if t == "salt":
        return median_filter(img)
    if t == "blur":
        return unsharp_mask(img, spec["k"], spec["sigma"])
    if t == "occlusion":
        return telea_inpaint(img, spec["rects"])
    raise ValueError(f"unknown corruption type: {t}")


class OracleClassicalDataset(ManifestDataset):
    """A ManifestDataset whose "input" is already restored by the oracle classical baseline.

    Evaluating it with an identity predict_fn reuses evaluate_restoration unchanged, and the
    filtering runs in the DataLoader workers.
    """

    def __getitem__(self, i):
        item = super().__getitem__(i)
        corrupted = item["input"].permute(1, 2, 0).numpy()
        restored = oracle_restore(np.ascontiguousarray(corrupted), self.entries[i])
        item["input"] = torch.from_numpy(np.ascontiguousarray(restored.transpose(2, 0, 1)))
        return item
