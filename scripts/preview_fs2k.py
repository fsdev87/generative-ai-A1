"""Save a grid of preprocessed FS2K photo/sketch pairs, one row per style (report figure).

Usage:
    python scripts/preview_fs2k.py [--split train] [--per-style 4] [--seed 0] [--out PATH] [--smoke]

Images come from the 128x128 cache, i.e. exactly what the models see. Each pair is
labelled with its pair id, which shows the photo source (photo1/2/3).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common.paths import get_dir  # noqa: E402
from src.data.fs2k import NUM_STYLES, STYLE_NAMES, load_fs2k  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--per-style", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: OUTPUT_DIR/task4/figures/fs2k_pairs.png")
    parser.add_argument("--smoke", action="store_true", help="use the synthetic smoke data")
    args = parser.parse_args()
    out = args.out or get_dir("OUTPUT_DIR", "task4", "figures") / "fs2k_pairs.png"

    data = load_fs2k(smoke=args.smoke)[args.split]
    rng = np.random.default_rng(args.seed)
    k = args.per_style
    fig, axes = plt.subplots(NUM_STYLES, 2 * k, figsize=(1.45 * 2 * k, 1.75 * NUM_STYLES), squeeze=False)
    for s in range(NUM_STYLES):
        members = np.flatnonzero(data["styles"] == s)
        picks = rng.choice(members, size=min(k, len(members)), replace=False)
        for j in range(k):
            for t in range(2):
                ax = axes[s, 2 * j + t]
                ax.set_xticks([]), ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if j >= len(picks):
                    continue
                i = picks[j]
                if t == 0:
                    ax.imshow(data["photos"][i])
                    ax.set_xlabel(str(data["ids"][i]), fontsize=6, color="#555555", loc="left")
                else:
                    ax.imshow(data["sketches"][i], cmap="gray", vmin=0, vmax=255)
        axes[s, 0].set_ylabel(f"{STYLE_NAMES[s]}\n(style {s})", fontsize=9)
    name = "Synthetic smoke data" if args.smoke else "FS2K"
    fig.suptitle(f"{name} {args.split}: photo / sketch pairs after preprocessing (128x128)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
