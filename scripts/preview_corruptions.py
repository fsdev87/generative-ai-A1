"""Save a grid showing every corruption at every fixed test severity (report figure).

Usage:
    python scripts/preview_corruptions.py --cache-dir data/cache --out outputs/corruption_grid.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.corruptions import CLASSES, LEVELS, apply_spec, level_spec  # noqa: E402
from src.data.pets import load_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/corruption_grid.png"))
    args = parser.parse_args()

    images = load_cache(args.cache_dir, args.split)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(images), args.n_images, replace=False)
    columns = [("clean", None)] + [(t, lv) for t in CLASSES[1:] for lv in LEVELS]

    fig, axes = plt.subplots(len(picks), len(columns), figsize=(1.6 * len(columns), 1.7 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(picks):
        clean = images[idx].astype(np.float32) / 255.0
        for c, (ctype, level) in enumerate(columns):
            spec = level_spec(ctype, level, rng)
            ax = axes[r, c]
            ax.imshow(np.clip(apply_spec(clean, spec), 0, 1))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(ctype if level is None else f"{ctype}\n{level}", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
