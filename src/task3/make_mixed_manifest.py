"""Extra experiment (beyond the brief's corruption definitions): a deterministic mixed-corruption test set.

The brief motivates soft routing with images that contain more than one corruption, but its
test manifest has exactly one corruption per entry. This script builds a small fixed set of
two-corruption combinations from the existing corruption functions and fixed seeds:

    salt+blur        blur (medium) then salt-and-pepper (medium)
    blur+occlusion   blur (medium) then occlusion (medium)
    salt+occlusion   occlusion (medium) then salt-and-pepper (medium)

Order of application: blur first (optics), then occlusion (an object in front), then
salt-and-pepper last (sensor/transmission noise), so salt pixels are never smeared by the blur.
Each component uses the fixed medium *test* severity (TEST_LEVELS). Images: a fixed random
subset (seed 3003) of the official test images; corruptions are re-created from the stored specs.

    python -m src.task3.make_mixed_manifest            # writes manifests/pets_mixed_manifest.json
"""
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.paths import MANIFEST_DIR
from src.data.corruptions import CLASS_TO_IDX, apply_spec, level_spec
from src.data.pets import load_json, load_pets, save_json

MIXED_SEED = 3003
N_IMAGES = 300
LEVEL = "medium"
COMBOS = {  # name -> components in application order
    "salt+blur": ("blur", "salt"),
    "blur+occlusion": ("blur", "occlusion"),
    "salt+occlusion": ("occlusion", "salt"),
}
MANIFEST_PATH = MANIFEST_DIR / "pets_mixed_manifest.json"


def test_images(test_manifest):
    """(image_idx, image_id) of every test image, in cache order."""
    return sorted({(e["image_idx"], e["image_id"]) for e in test_manifest["entries"]})


def make_mixed_manifest(test_manifest, n_images=N_IMAGES, seed=MIXED_SEED, level=LEVEL, size=128):
    rng = np.random.default_rng(seed)
    images = test_images(test_manifest)
    chosen = sorted(rng.choice(len(images), size=min(n_images, len(images)), replace=False).tolist())
    entries = []
    for i in chosen:
        image_idx, image_id = images[i]
        for name, components in COMBOS.items():
            entry_seed = int(rng.integers(2**31))
            comp_rng = np.random.default_rng(entry_seed)
            specs = [level_spec(c, level, comp_rng, size) for c in components]
            entries.append({"image_idx": image_idx, "image_id": image_id, "entry_seed": entry_seed, "type": name,
                            "level": level, "components": specs})
    return {"split": "test_mixed", "seed": seed, "image_size": size, "level": level,
            "combos": {k: list(v) for k, v in COMBOS.items()}, "entries": entries,
            "note": "Extra experiment beyond the brief's corruption definitions (two corruptions per image)."}


def load_mixed(smoke=False):
    """(clean test images, mixed manifest). Smoke builds the manifest in memory from the synthetic test set."""
    images, manifests = load_pets(smoke=smoke)
    if smoke:
        return images["test"], make_mixed_manifest(manifests["test"], n_images=4)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} missing: run python -m src.task3.make_mixed_manifest")
    return images["test"], load_json(MANIFEST_PATH)


class MixedManifestDataset(Dataset):
    """Inputs with two corruptions applied in order. Returns input, target, combo index, components
    (multi-hot over CLASSES), image_idx."""

    def __init__(self, images, manifest):
        self.images = images
        self.entries = manifest["entries"]
        self.combos = list(manifest["combos"])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["image_idx"]].astype(np.float32) / 255.0
        x = clean
        for spec in e["components"]:
            x = apply_spec(x, spec)
        multi_hot = torch.zeros(len(CLASS_TO_IDX))
        multi_hot[[CLASS_TO_IDX[s["type"]] for s in e["components"]]] = 1
        return {"input": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                "target": torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1))),
                "combo": self.combos.index(e["type"]), "components": multi_hot, "image_idx": e["image_idx"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args(argv)
    if MANIFEST_PATH.exists() and not args.force:
        print(f"[skip] {MANIFEST_PATH} exists (use --force to rebuild; the result is identical for the same seed)")
        return load_json(MANIFEST_PATH)
    manifest = make_mixed_manifest(load_json(MANIFEST_DIR / "pets_test_manifest.json"), n_images=args.n_images)
    save_json(manifest, MANIFEST_PATH)
    print(f"wrote {MANIFEST_PATH}: {len(manifest['entries'])} entries "
          f"({args.n_images} test images x {len(COMBOS)} combinations, {LEVEL} severity)")
    return manifest


if __name__ == "__main__":
    main()
