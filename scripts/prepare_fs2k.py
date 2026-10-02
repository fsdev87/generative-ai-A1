"""Verify the FS2K pairing, write the train/val/test split manifest and build the 128x128 cache (Task 4).

Usage:
    python scripts/prepare_fs2k.py [--fs2k-dir data/FS2K] [--skip-cache] [--force]

The split goes to manifests/fs2k_split.json (small, committed, so every run and every
machine uses the same validation set; an existing file is checked, never silently
replaced). The cache (fs2k_{train,val,test}_128.npz in CACHE_DIR) holds uint8 photos,
one-channel uint8 sketches, style labels and pair ids.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import get_dir  # noqa: E402
from src.data.fs2k import (  # noqa: E402
    SPLIT_PATH, SPLITS, build_cache, cache_path, check_split, load_split, make_split, official_pairs,
    save_split, split_counts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs2k-dir", type=Path, default=None, help="default: DATA_DIR/FS2K")
    parser.add_argument("--skip-cache", action="store_true", help="only verify the data and write the split")
    parser.add_argument("--force", action="store_true", help="regenerate the split and cache even if present")
    args = parser.parse_args()
    fs2k_dir = args.fs2k_dir or get_dir("DATA_DIR") / "FS2K"

    pairs = official_pairs(fs2k_dir)  # raises on any missing, unannotated or size-mismatched pair
    print(f"official pairs: {len(pairs['train'])} train / {len(pairs['test'])} test "
          "(one-to-one, photo and sketch sizes equal)")

    if SPLIT_PATH.exists() and not args.force:
        split = load_split()
        check_split(split, pairs)
        print(f"[skip] using existing {SPLIT_PATH.name} (consistent with the official lists)")
    else:
        split = make_split(pairs)
        save_split(split)
        print(f"wrote {SPLIT_PATH}")
    for name, counts in split_counts(split).items():
        sources = Counter(i.split("/")[0] for i in split[name])
        print(f"  {name:5s}: {len(split[name]):4d} pairs, per style {counts}, per source {dict(sorted(sources.items()))}")

    if args.skip_cache:
        return
    by_id = {p["id"]: p for p in pairs["train"] + pairs["test"]}
    for name in SPLITS:
        out = cache_path(name)
        if out.exists() and not args.force:
            print(f"[skip] {out.name} exists")
            continue
        data = build_cache([by_id[i] for i in split[name]], out)
        print(f"cached {name}: photos {data['photos'].shape}, sketches {data['sketches'].shape} -> {out}")


if __name__ == "__main__":
    main()
