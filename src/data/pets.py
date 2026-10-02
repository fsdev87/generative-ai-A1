"""Oxford-IIIT Pet data: splits, 128x128 image cache, corruption manifests and datasets.

The same split and manifests are used by Tasks 1, 2 and 3.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.common.paths import MANIFEST_DIR, get_dir

from .corruptions import CLASSES, CLASS_TO_IDX, LEVELS, apply_spec, level_spec, sample_spec

IMAGE_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.2
VAL_MANIFEST_SEED = 1001
TEST_MANIFEST_SEED = 2002


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def read_official_list(pets_dir, name):
    """Image ids from annotations/{trainval,test}.txt (first token of every line)."""
    path = Path(pets_dir) / "annotations" / f"{name}.txt"
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip() and not line.startswith("#")]


def make_splits(pets_dir, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    """80/20 train/val split of the official trainval list; official test untouched."""
    trainval = sorted(read_official_list(pets_dir, "trainval"))
    test = sorted(read_official_list(pets_dir, "test"))
    perm = np.random.default_rng(seed).permutation(len(trainval))
    n_val = int(round(val_fraction * len(trainval)))
    val = sorted(trainval[i] for i in perm[:n_val])
    train = sorted(trainval[i] for i in perm[n_val:])
    return {"seed": seed, "val_fraction": val_fraction, "train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Image cache (clean images only, resized once)
# --------------------------------------------------------------------------- #
def load_image(path, size=IMAGE_SIZE):
    """RGB, resized to size x size (no crop), float32 in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def build_cache(pets_dir, ids, out_path, size=IMAGE_SIZE):
    """Store the clean resized images of one split as a uint8 (N, H, W, 3) .npy file."""
    images = np.empty((len(ids), size, size, 3), dtype=np.uint8)
    for i, image_id in enumerate(ids):
        with Image.open(Path(pets_dir) / "images" / f"{image_id}.jpg") as im:
            images[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC))
    np.save(out_path, images)
    return images


def load_cache(cache_dir, split):
    return np.load(Path(cache_dir) / f"pets_{split}_{IMAGE_SIZE}.npy")


def synthetic_images(n, size=IMAGE_SIZE, seed=0):
    """Smooth random colour images (uint8) standing in for pets in smoke tests."""
    rng = np.random.default_rng(seed)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        small = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        images[i] = np.asarray(Image.fromarray(small).resize((size, size), Image.BICUBIC))
    return images


def load_pets(smoke=False, cache_dir=None):
    """Clean images per split (uint8, N x 128 x 128 x 3) and the val/test manifests.

    smoke=True returns a tiny synthetic dataset with matching manifests, so every
    script can be smoke-tested on CPU without the real data.
    """
    if smoke:
        sizes = {"train": 32, "val": 16, "test": 4}
        images = {split: synthetic_images(n, seed=i) for i, (split, n) in enumerate(sizes.items())}
        manifests = {
            "val": make_val_manifest([f"val_{i}" for i in range(sizes["val"])]),
            "test": make_test_manifest([f"test_{i}" for i in range(sizes["test"])]),
        }
        return images, manifests

    cache_dir = Path(cache_dir) if cache_dir else get_dir("CACHE_DIR")
    images = {split: load_cache(cache_dir, split) for split in ("train", "val", "test")}
    manifests = {split: load_json(MANIFEST_DIR / f"pets_{split}_manifest.json") for split in ("val", "test")}
    for split, manifest in manifests.items():
        expected = max(e["image_idx"] for e in manifest["entries"]) + 1
        if expected != len(images[split]):
            raise ValueError(f"{split}: manifest indexes {expected} images but the cache holds {len(images[split])}")
    return images, manifests


# --------------------------------------------------------------------------- #
# Deterministic manifests
# --------------------------------------------------------------------------- #
def make_val_manifest(val_ids, seed=VAL_MANIFEST_SEED, size=IMAGE_SIZE):
    """One condition per validation image, balanced across the four classes,
    with severities sampled from the training ranges."""
    rng = np.random.default_rng(seed)
    types = np.resize(np.array(CLASSES), len(val_ids))
    rng.shuffle(types)
    entries = []
    for idx, (image_id, ctype) in enumerate(zip(val_ids, types)):
        entry_seed = int(rng.integers(2**31))
        spec = sample_spec(str(ctype), np.random.default_rng(entry_seed), size)
        entries.append({"image_idx": idx, "image_id": image_id, "entry_seed": entry_seed, **spec})
    return {"split": "val", "seed": seed, "image_size": size, "entries": entries}


def make_test_manifest(test_ids, seed=TEST_MANIFEST_SEED, size=IMAGE_SIZE):
    """Every test image: clean + 3 corruptions x 3 fixed severity levels (10 variants)."""
    rng = np.random.default_rng(seed)
    entries = []
    for idx, image_id in enumerate(test_ids):
        variants = [("clean", None)] + [(t, lv) for t in CLASSES[1:] for lv in LEVELS]
        for ctype, level in variants:
            entry_seed = int(rng.integers(2**31))
            spec = level_spec(ctype, level, np.random.default_rng(entry_seed), size)
            entries.append({"image_idx": idx, "image_id": image_id, "entry_seed": entry_seed, **spec})
    return {"split": "test", "seed": seed, "image_size": size, "entries": entries}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class RuntimeCorruptionDataset(Dataset):
    """Training dataset: a fresh corruption type and severity on every load.

    `index` may be an int (condition sampled uniformly from `conditions`) or an
    (int, condition) tuple supplied by BalancedBatchSampler.
    Returns dict(input, target, label, index).
    """

    def __init__(self, images, conditions=CLASSES, hflip=True):
        self.images = images  # uint8 (N, H, W, 3)
        self.conditions = tuple(conditions)
        self.hflip = hflip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # torch's RNG is seeded differently in every DataLoader worker and epoch
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        if isinstance(index, (tuple, list)):
            index, ctype = index
        else:
            ctype = self.conditions[rng.integers(len(self.conditions))]
        clean = self.images[index].astype(np.float32) / 255.0
        if self.hflip and rng.random() < 0.5:
            clean = clean[:, ::-1]
        spec = sample_spec(ctype, rng, clean.shape[0])
        return {
            "input": _to_tensor(apply_spec(clean, spec)),
            "target": _to_tensor(clean),
            "label": CLASS_TO_IDX[ctype],
            "index": index,
        }


class ManifestDataset(Dataset):
    """Validation/test dataset: corruptions read from a deterministic manifest.

    `types` optionally restricts the entries (e.g. ["blur"] for a specialist).
    Returns dict(input, target, label, level, image_idx).
    """

    def __init__(self, images, manifest, types=None):
        self.images = images
        entries = manifest["entries"]
        self.entries = [e for e in entries if types is None or e["type"] in types]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["image_idx"]].astype(np.float32) / 255.0
        return {
            "input": _to_tensor(apply_spec(clean, e)),
            "target": _to_tensor(clean),
            "label": CLASS_TO_IDX[e["type"]],
            "level": e["level"] or "none",
            "image_idx": e["image_idx"],
        }


class BalancedBatchSampler(Sampler):
    """Yields batches of (index, condition) pairs with equal counts of every condition.

    Images are reshuffled every epoch; batch_size must be divisible by the number
    of conditions. The last incomplete batch is dropped to keep every batch balanced.
    """

    def __init__(self, n_items, batch_size, conditions=CLASSES, seed=None):
        if batch_size % len(conditions):
            raise ValueError(f"batch_size {batch_size} not divisible by {len(conditions)}")
        self.n_items = n_items
        self.batch_size = batch_size
        self.conditions = tuple(conditions)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_items // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(self.n_items)
        per_class = self.batch_size // len(self.conditions)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            conds = np.repeat(self.conditions, per_class)
            self.rng.shuffle(conds)
            yield [(int(i), str(c)) for i, c in zip(idx, conds)]
