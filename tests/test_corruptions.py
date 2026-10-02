"""Sanity checks for the corruption pipeline. Run: python -m pytest tests -q"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

from src.data.corruptions import (
    CLASSES, TEST_LEVELS, apply_spec, gaussian_blur, level_spec, make_occlusion_rects, sample_spec,
)
from src.data.pets import BalancedBatchSampler, RuntimeCorruptionDataset

RNG = np.random.default_rng(0)
IMG = RNG.random((128, 128, 3)).astype(np.float32)


def test_salt_fraction_and_values():
    spec = {"type": "salt", "p": 0.08, "seed": 1}
    out = apply_spec(IMG, spec)
    changed = np.any(out != IMG, axis=2)
    assert abs(changed.mean() - 0.08) < 0.01
    vals = out[changed]
    assert np.all((vals == 0.0) | (vals == 1.0))
    white = np.all(vals == 1.0, axis=1).mean()
    assert abs(white - 0.5) < 0.05


def test_salt_deterministic():
    spec = {"type": "salt", "p": 0.1, "seed": 7}
    assert np.array_equal(apply_spec(IMG, spec), apply_spec(IMG, spec))


def test_blur_matches_torchvision():
    for k, s in [(3, 0.7), (5, 1.5), (7, 2.5)]:
        ours = gaussian_blur(IMG, k, s)
        ref = tv_gaussian_blur(torch.from_numpy(IMG.transpose(2, 0, 1)), [k, k], [s, s]).numpy().transpose(1, 2, 0)
        assert np.abs(ours - ref).max() < 1e-5


def test_occlusion_test_levels_cover():
    for level, params in TEST_LEVELS["occlusion"].items():
        for seed in range(50):
            spec = level_spec("occlusion", level, np.random.default_rng(seed))
            assert len(spec["rects"]) == params["n"]
            assert abs(spec["cover"] - params["cover"]) <= 0.01
            out = apply_spec(IMG, spec)
            assert abs(np.all(out == 0, axis=2).mean() - spec["cover"]) < 0.005


def test_training_specs_in_range():
    rng = np.random.default_rng(1)
    for _ in range(300):
        s = sample_spec("salt", rng)
        assert 0.02 <= s["p"] <= 0.15
        b = sample_spec("blur", rng)
        assert b["k"] in (3, 5, 7) and 0.5 <= b["sigma"] <= 2.5
        o = sample_spec("occlusion", rng)
        assert 1 <= len(o["rects"]) <= 3 and 0.10 <= o["cover"] <= 0.35


def test_occlusion_rects_inside_image():
    rng = np.random.default_rng(2)
    for _ in range(200):
        rects, _ = make_occlusion_rects(3, 0.35, rng)
        for y, x, h, w in rects:
            assert 0 <= y and y + h <= 128 and 0 <= x and x + w <= 128


def test_balanced_batches():
    images = (RNG.random((40, 128, 128, 3)) * 255).astype(np.uint8)
    ds = RuntimeCorruptionDataset(images)
    loader = DataLoader(ds, batch_sampler=BalancedBatchSampler(len(ds), 8, seed=0))
    for batch in loader:
        counts = torch.bincount(batch["label"], minlength=len(CLASSES))
        assert torch.all(counts == 2)
        assert batch["input"].shape == (8, 3, 128, 128)
