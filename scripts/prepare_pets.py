"""Create the Oxford-IIIT Pet split, the deterministic val/test manifests and the 128x128 image cache.

Usage:
    python scripts/prepare_pets.py --pets-dir data/oxford_pets [--cache-dir data/cache] [--skip-cache]

Split and manifests go to manifests/ (small, committed to Git so every run and
every task uses identical data). The image cache contains clean images only.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pets import (  # noqa: E402
    build_cache, load_json, make_splits, make_test_manifest, make_val_manifest, save_json,
)

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pets-dir", type=Path, default=Path("data/oxford_pets"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--skip-cache", action="store_true", help="only write split and manifests")
    parser.add_argument("--force", action="store_true", help="regenerate split/manifests even if present")
    args = parser.parse_args()

    split_path = MANIFEST_DIR / "pets_split.json"
    if split_path.exists() and not args.force:
        splits = load_json(split_path)
        print(f"[skip] using existing {split_path.name}")
    else:
        splits = make_splits(args.pets_dir)
        save_json(splits, split_path)
    print(f"split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")

    for name, builder, ids in [
        ("pets_val_manifest.json", make_val_manifest, splits["val"]),
        ("pets_test_manifest.json", make_test_manifest, splits["test"]),
    ]:
        path = MANIFEST_DIR / name
        if path.exists() and not args.force:
            print(f"[skip] using existing {name}")
            continue
        manifest = builder(ids)
        save_json(manifest, path)
        counts = Counter((e["type"], e["level"]) for e in manifest["entries"])
        print(f"{name}: {len(manifest['entries'])} entries")
        for key in sorted(counts, key=str):
            print(f"  {key}: {counts[key]}")

    if not args.skip_cache:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            out = args.cache_dir / f"pets_{split}_128.npy"
            if out.exists() and not args.force:
                print(f"[skip] {out.name} exists")
                continue
            images = build_cache(args.pets_dir, splits[split], out)
            print(f"cached {split}: {images.shape} -> {out}")


if __name__ == "__main__":
    main()
