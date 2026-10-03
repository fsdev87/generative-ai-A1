"""Task 4: evaluate the trained generator on the official FS2K test set.

Metrics against the ground-truth sketch: L1/MAE and PSNR (pixel fidelity), SSIM
(structure) and edge_ratio (sharpness diagnostic: generated edge energy divided by the
ground truth's, 1 = as sharp as the artist's drawing). Everything is reported per style
and per photo source, because in the official training portion style and source are
confounded (styles 0/1 only from photo1, style 2 only from photo2/photo3), so the test
set contains (source, style) combinations never seen in training. Sketches are compared
against the original ground truth without background normalisation, so the grey paper of
the photo3 sketches is part of the error (see docs/fs2k_notes.md).

Writes per-entry records, summary tables (CSV + LaTeX) and the figures (examples,
style swap, failure cases) to OUTPUT_DIR/task4/{eval,tables,figures}/.

    python -m src.task4.evaluate
    python -m src.task4.evaluate --smoke
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.evaluation import save_csv, save_latex_table
from src.common.utils import get_device
from src.data.fs2k import NUM_STYLES, STYLE_NAMES, load_fs2k, load_split
from src.task4.models import UNetGenerator

from .config import checkpoint_dir, output_dir
from .metrics import sketch_metrics, sketch_score, to_unit
from .train import generate, tensorize

METRICS = ("l1", "ssim", "psnr", "edge_ratio", "score")
FMT = {"l1": "{:.4f}", "ssim": "{:.4f}", "psnr": "{:.2f}", "edge_ratio": "{:.3f}", "score": "{:.4f}"}


def seen_combinations(smoke=False):
    """(source, style) pairs that occur in our training split; the rest are unseen at test time."""
    if smoke:
        return None
    split = load_split()
    return {(i.split("/")[0], split["style"][i]) for i in split["train"]}


@torch.no_grad()
def evaluate_generator(generator, data, device, seen=None):
    """One record per test pair: metrics, style, photo source and whether the combination was seen."""
    output = generate(generator, data["photo"], data["style"], device)
    target = data["sketch"].to(output.device)
    per_image = sketch_metrics(output, target)
    records = []
    for i, pair_id in enumerate(data["id"]):
        source = pair_id.split("/")[0]
        style = int(data["style"][i])
        values = {k: float(v[i]) for k, v in per_image.items()}
        records.append({
            "id": pair_id, "style": style, "style_name": STYLE_NAMES[style], "source": source,
            "seen_combination": None if seen is None else ((source, style) in seen),
            **values, "score": sketch_score(values["ssim"], values["l1"]),
        })
    return records, output.cpu()


def summarize(records, keys):
    """Mean metrics and count per group of `keys` (plus an "all" row when keys is empty)."""
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    rows = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        rs = groups[key]
        rows.append({**dict(zip(keys, key)), "n": len(rs),
                     **{m: float(np.mean([r[m] for r in rs])) for m in METRICS}})
    return rows


def macro_average(rows):
    """Unweighted mean over groups: the test set has 619/381/46 pairs per style, so the
    pooled mean is dominated by style 0 and hides how style 2 does."""
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def style_sensitivity(generator, data, device):
    """Mean L1 with the true style vs with each wrong style (does the condition matter?)."""
    target = to_unit(data["sketch"])
    true_l1 = float((to_unit(generate(generator, data["photo"], data["style"], device).cpu())
                     - target).abs().flatten(1).mean(1).mean())
    wrong = []
    for shift in range(1, NUM_STYLES):
        out = generate(generator, data["photo"], (data["style"] + shift) % NUM_STYLES, device).cpu()
        wrong.append(float((to_unit(out) - target).abs().flatten(1).mean(1).mean()))
    return {"l1_true_style": true_l1, "l1_wrong_style": float(np.mean(wrong)),
            "style_gap": float(np.mean(wrong)) - true_l1}


def failure_notes(record, seen_known):
    """Short, data-driven pointers for the report's failure discussion."""
    notes = []
    if seen_known and record["seen_combination"] is False:
        notes.append(f"(source, style) combination unseen in training ({record['source']}, "
                     f"style {record['style']})")
    if record["source"] == "photo3":
        notes.append("photo3 sketches are drawn on grey paper (background offset, see notes)")
    if record["edge_ratio"] < 0.7:
        notes.append(f"strokes too soft (edge ratio {record['edge_ratio']:.2f})")
    elif record["edge_ratio"] > 1.4:
        notes.append(f"too much ink (edge ratio {record['edge_ratio']:.2f})")
    return "; ".join(notes) or "high reconstruction error without an obvious data cause"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _axes(n_rows, n_cols, scale=1.6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(scale * n_cols, scale * n_rows + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    return fig, axes


def _show(ax, image, gray=False):
    if gray:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(image, 0, 1))


def example_figure(data, outputs, entries, path, title, labels=None):
    """Rows of photo | ground truth | generated for the selected test entries."""
    fig, axes = _axes(len(entries), 3)
    for r, i in enumerate(entries):
        _show(axes[r, 0], to_unit(data["photo"][i]).permute(1, 2, 0).numpy())
        _show(axes[r, 1], to_unit(data["sketch"][i])[0].numpy(), gray=True)
        _show(axes[r, 2], to_unit(outputs[i])[0].numpy(), gray=True)
        axes[r, 0].set_ylabel(labels[r] if labels else data["id"][i], fontsize=6)
    for c, name in enumerate(["Photo", "Ground truth", "Generated"]):
        axes[0, c].set_title(name, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    _close(fig)


def style_swap_figure(generator, data, entries, device, path):
    """The same photo generated in all three styles, next to its own ground truth."""
    photos = data["photo"][entries]
    per_style = [generate(generator, photos, torch.full((len(entries),), s, dtype=torch.long), device).cpu()
                 for s in range(NUM_STYLES)]
    fig, axes = _axes(len(entries), 2 + NUM_STYLES)
    for r, i in enumerate(entries):
        _show(axes[r, 0], to_unit(data["photo"][i]).permute(1, 2, 0).numpy())
        _show(axes[r, 1], to_unit(data["sketch"][i])[0].numpy(), gray=True)
        for s in range(NUM_STYLES):
            _show(axes[r, 2 + s], to_unit(per_style[s][r])[0].numpy(), gray=True)
        axes[r, 0].set_ylabel(f"{data['id'][i]}\ntrue: {STYLE_NAMES[int(data['style'][i])]}", fontsize=6)
    for c, name in enumerate(["Photo", "Ground truth", *[f"Generated\n{n}" for n in STYLE_NAMES]]):
        axes[0, c].set_title(name, fontsize=8)
    fig.suptitle("Style swap: one photo, every style condition", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    _close(fig)


def _close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def representative_entries(records, per_style=2):
    """Entries closest to the median score of each style: typical results, not cherry-picked."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["style"]].append((i, r))
    chosen = []
    for style in sorted(groups):
        items = groups[style]
        median = np.median([r["score"] for _, r in items])
        chosen += [i for i, _ in sorted(items, key=lambda it: abs(it[1]["score"] - median))[:per_style]]
    return chosen


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(checkpoint=None, smoke=False, n_failures=6, device=None):
    device = device or get_device()
    ckpt_path = checkpoint or checkpoint_dir(smoke=smoke) / "best.pt"
    ckpt = load_checkpoint(ckpt_path)
    generator = build_model(UNetGenerator, ckpt).to(device).eval()
    data = tensorize(load_fs2k(smoke=smoke)["test"])
    seen = seen_combinations(smoke)

    records, outputs = evaluate_generator(generator, data, device, seen)
    by_style = summarize(records, ("style_name",))
    by_source = summarize(records, ("source",))
    by_source_style = summarize(records, ("source", "style_name"))
    overall = summarize(records, ())[0]
    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt["epoch"],
        "n_test_pairs": len(records),
        "overall": overall,
        "macro_over_styles": macro_average(by_style),
        "by_style": by_style,
        "by_source": by_source,
        "by_source_style": by_source_style,
        "style_sensitivity": style_sensitivity(generator, data, device),
        "validation_at_best_epoch": ckpt["metrics"],
    }
    if seen is not None:
        summary["by_seen_combination"] = summarize(records, ("seen_combination",))

    out = output_dir(smoke)
    save_csv(records, out / "eval" / "test_records.csv")
    (out / "eval" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, rows, keys in [("by_style", by_style, ("style_name",)), ("by_source", by_source, ("source",)),
                             ("by_source_style", by_source_style, ("source", "style_name"))]:
        save_csv(rows, out / "tables" / f"{name}.csv")
        columns = [(k, k.replace("_", " ")) for k in keys] + [("n", "n")] + [(m, m.upper()) for m in METRICS]
        save_latex_table(rows, columns, out / "tables" / f"{name}.tex",
                         caption=f"Task 4 test metrics {name.replace('_', ' ')}", label=f"tab:task4-{name}",
                         fmt=FMT)

    figures = out / "figures"
    def labels_for(entries):
        return [f"{records[i]['id']}\n{records[i]['style_name']}, score {records[i]['score']:.3f}"
                for i in entries]

    typical = representative_entries(records)
    example_figure(data, outputs, typical, figures / "test_examples.png",
                   "Typical results (median score per style)", labels=labels_for(typical))
    worst = [i for i, _ in sorted(enumerate(records), key=lambda it: it[1]["score"])[:n_failures]]
    example_figure(data, outputs, worst, figures / "failure_cases.png", "Failure cases (lowest score)",
                   labels=labels_for(worst))
    style_swap_figure(generator, data, representative_entries(records, per_style=1), device,
                      figures / "style_swap.png")
    failures = [{"rank": k + 1, **{f: records[i][f] for f in ("id", "style_name", "source", *METRICS)},
                 "note": failure_notes(records[i], seen is not None)} for k, i in enumerate(worst)]
    save_csv(failures, out / "eval" / "failure_cases.csv")

    print(f"[task4] test pairs {len(records)} | L1 {overall['l1']:.4f} SSIM {overall['ssim']:.4f} "
          f"PSNR {overall['psnr']:.2f} edge {overall['edge_ratio']:.3f} score {overall['score']:.4f}")
    for row in by_style:
        print(f"  {row['style_name']:8s} n={row['n']:4d} L1 {row['l1']:.4f} SSIM {row['ssim']:.4f} "
              f"PSNR {row['psnr']:.2f} edge {row['edge_ratio']:.3f}")
    for row in by_source:
        print(f"  {row['source']:8s} n={row['n']:4d} L1 {row['l1']:.4f} SSIM {row['ssim']:.4f}")
    print(f"  style sensitivity: {summary['style_sensitivity']}")
    print(f"[task4] results -> {out}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="explicit checkpoint path (default: CKPT_DIR/task4/cgan/best.pt)")
    parser.add_argument("--failures", type=int, default=6, help="failure cases in the figure and CSV")
    parser.add_argument("--smoke", action="store_true", help="synthetic data and the smoke checkpoint")
    args = parser.parse_args(argv)
    return run(args.checkpoint, smoke=args.smoke, n_failures=args.failures)


if __name__ == "__main__":
    main()
