"""The backend's NumPy PSNR/SSIM must match the training metrics in src.common (PyTorch)."""
import numpy as np
import pytest

from app.services.images import psnr, ssim

torch = pytest.importorskip("torch")
from src.common.losses import ssim as torch_ssim  # noqa: E402

RNG = np.random.default_rng(42)


def reference_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """src.common.losses.ssim on (H, W, C) arrays."""
    to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float()  # noqa: E731
    return float(torch_ssim(to_tensor(x), to_tensor(y)))


def smooth_image(shape):
    """A random image with structure (blurred noise plus a gradient), values in [0, 1]."""
    noise = RNG.random(shape)
    for axis in (0, 1):
        noise = (noise + np.roll(noise, 1, axis) + np.roll(noise, -1, axis)) / 3
    ramp = np.linspace(0, 0.3, shape[1])[None, :, None]
    return np.clip(noise * 0.7 + ramp, 0, 1)


def pairs():
    clean = smooth_image((128, 128, 3))
    salt = clean.copy()
    mask = RNG.random(clean.shape[:2]) < 0.08
    salt[mask] = RNG.integers(0, 2, (mask.sum(), 1))
    occluded = clean.copy()
    occluded[30:70, 20:90] = 0
    noisy = np.clip(clean + RNG.normal(0, 0.1, clean.shape), 0, 1)
    other = smooth_image((64, 96, 3))
    return [
        ("salt", salt, clean),
        ("occlusion", occluded, clean),
        ("gaussian noise", noisy, clean),
        ("unrelated", RNG.random((128, 128, 3)), clean),
        ("non-square", np.clip(other + 0.05, 0, 1), other),
        ("identical", clean, clean),
    ]


@pytest.mark.parametrize("name,x,y", pairs(), ids=[p[0] for p in pairs()])
def test_ssim_matches_training_implementation(name, x, y):
    ours, theirs = ssim(x, y), reference_ssim(x, y)
    assert abs(ours - theirs) < 1e-5, (name, ours, theirs)


def test_ssim_of_identical_images_is_one():
    image = smooth_image((128, 128, 3))
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-12)
    assert ssim(image[..., 0], image[..., 0]) == pytest.approx(1.0, abs=1e-12)  # greyscale


def test_ssim_rejects_mismatched_or_tiny_images():
    with pytest.raises(ValueError):
        ssim(np.zeros((128, 128, 3)), np.zeros((64, 64, 3)))
    with pytest.raises(ValueError):
        ssim(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))


def test_psnr_cap_and_value():
    image = smooth_image((32, 32, 3))
    assert psnr(image, image) == 100.0
    assert psnr(np.full((4, 4, 3), 0.1), np.zeros((4, 4, 3))) == pytest.approx(20.0)
