"""Task 3: test-set evaluation and gating analysis of the soft MoE (official test manifest).

Runs CKPT_DIR/task3/moe/best.pt in float32 (the precision of the exported ONNX model) on all
36,690 test entries and writes, under OUTPUT_DIR/task3/:

  eval/test_records.csv      one row per entry: evaluate_restoration columns + w_clean, w_salt,
                             w_blur, w_occlusion, dominant branch, normalised routing entropy
  eval/summary.json          headline metrics, routing summary, expert-activity check, collapse verdict
  tables/test_by_*.{csv,tex} standard restoration tables (+ n_exact / psnr_finite, see below)
  tables/routing_by_type_level.{csv,tex}, routing_by_type.{csv,tex}
                             average branch weights per true corruption type and severity
  tables/expert_activity.{csv,tex}
                             per branch: usage, weight on own class / other classes, how often dominant
  tables/routing_entropy.{csv,tex}
  figures/examples_a.png, examples_b.png, failures.png, failures_misrouted.png
                             Target | Input | Output | |Error| (same style as Tasks 1-2)
  figures/routing_heatmap_type_level.png, routing_matrix.png, weight_distributions.png,
  figures/routing_entropy.png, gating_dominant.png, gating_spread.png

PSNR is capped at 100 dB for outputs identical to the target (src.common.metrics). The soft
MoE never bypasses exactly, but a clean input with w_clean rounding to 1.0 in float32 can
reach the cap; tables therefore report n_exact (entries at the cap) and psnr_finite (mean
over the others) next to the plain mean, as Task 2 does.

    python -m src.task3.evaluate [--checkpoint PATH] [--batch-size 64] [--smoke]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.evaluation import (
    evaluate_restoration, predict_entries, representative_entries, save_csv, save_latex_table,
    save_restoration_figure, standard_tables, worst_entries,
)
from src.common.metrics import PSNR_CAP_DB, restoration_score
from src.common.utils import get_device
from src.data.corruptions import CLASSES, CLASS_TO_IDX, LEVELS
from src.data.pets import ManifestDataset, load_pets

from .config import checkpoint_dir, output_dir
from .model import BRANCHES, SoftMoERestorer
from .plots import (
    save_branch_figure, save_entropy_histogram, save_heatmap, save_weight_distributions,
)
from .routing import detect_collapse, routing_entropy, routing_stats

WEIGHT_COLS = [f"w_{c}" for c in CLASSES]  # w_clean is the identity branch


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_moe(path, device):
    model = build_model(SoftMoERestorer, path).to(device).eval()
    return model, load_checkpoint(path)


def make_predict_fn(model):
    """predict_fn for evaluate_restoration: output plus routing extras (moved to CPU once per batch)."""

    def predict(x):
        output, weights, _ = model(x)
        w = weights.float().cpu()
        extras = {col: w[:, k] for k, col in enumerate(WEIGHT_COLS)}
        extras["dominant"] = [CLASSES[i] for i in w.argmax(dim=1).tolist()]
        extras["entropy"] = routing_entropy(w).float()
        return output, extras

    return predict


@torch.no_grad()
def branch_samples(model, dataset, entries, records, device):
    """Figure-ready samples with every branch output and its weight."""
    items = [dataset[i] for i in entries]
    x = torch.stack([it["input"] for it in items]).to(device)
    output, weights, branches = model(x)

    def hwc(t):
        return t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    samples = []
    for k, (entry, it) in enumerate(zip(entries, items)):
        r = records[entry]
        level = "" if r["level"] == "none" else f" {r['level']}"
        samples.append({"input": hwc(it["input"]), "target": hwc(it["target"]), "output": hwc(output[k]),
                        "branches": [hwc(branches[k, b]) for b in range(len(BRANCHES))],
                        "weights": weights[k].float().cpu().tolist(),
                        "label": f"{r['type']}{level}\nPSNR {r['psnr']:.1f}"})
    return samples


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _exact_columns(df, rows, keys):
    """n_exact (PSNR at the 100 dB cap) and psnr_finite (mean PSNR of the other entries) per row."""
    out = []
    for row in rows:
        mask = np.ones(len(df), dtype=bool)
        for k in keys:
            if row[k] != "all":
                mask &= (df[k] == row[k]).to_numpy()
        sub = df[mask]
        exact = sub["psnr"] >= PSNR_CAP_DB
        out.append({**row, "n_exact": int(exact.sum()),
                    "psnr_finite": float(sub.loc[~exact, "psnr"].mean()) if (~exact).any() else float("nan")})
    return out


def _latex_ready(rows):
    return [{k: ("--" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows]


def restoration_tables(records, tables_dir):
    df = pd.DataFrame(records)
    corrupted = df[df["type"] != "clean"]
    keys = {"by_type": ("type",), "by_type_level": ("type", "level"), "by_level": ("level",)}
    out = {}
    for name, rows in standard_tables(records).items():
        rows = _exact_columns(corrupted if name == "by_level" else df, rows, keys[name])
        if name == "by_type":  # mean over the three corruptions, without the clean entries
            rows.append({"type": "corrupted", "n": len(corrupted), "psnr": float(corrupted["psnr"].mean()),
                         "ssim": float(corrupted["ssim"].mean()), "mse": float(corrupted["mse"].mean()),
                         "n_exact": int((corrupted["psnr"] >= PSNR_CAP_DB).sum()),
                         "psnr_finite": float(corrupted.loc[corrupted["psnr"] < PSNR_CAP_DB, "psnr"].mean())})
        save_csv(rows, tables_dir / f"test_{name}.csv")
        first = [(k, k.capitalize()) for k in keys[name]]
        save_latex_table(_latex_ready(rows), first + [("n", "n"), ("psnr", "PSNR"), ("psnr_finite", "PSNR*"),
                                                      ("ssim", "SSIM"), ("n_exact", "Exact")],
                         tables_dir / f"test_{name}.tex",
                         caption="Soft MoE on the test set. PSNR* excludes outputs identical to the target "
                                 "(PSNR capped at 100 dB, counted in Exact).",
                         label=f"tab:task3_{name}", fmt={"psnr_finite": "{:.2f}"})
        out[name] = rows
    return out


def routing_tables(df, tables_dir):
    """Average branch weights (and entropy, routing accuracy) per true type and severity level."""
    df = df.assign(correct=(df["dominant"] == df["type"]).astype(float))
    agg = {**{c: "mean" for c in WEIGHT_COLS}, "entropy": "mean", "correct": "mean"}
    level_order = {lv: i for i, lv in enumerate(("none", *LEVELS))}
    by_tl = (df.groupby(["type", "level"]).agg(n=("entry", "size"), **{k: (k, v) for k, v in agg.items()})
             .reset_index())
    by_tl = by_tl.sort_values(by=["type", "level"], key=lambda s: s.map(CLASS_TO_IDX) if s.name == "type"
                              else s.map(level_order)).reset_index(drop=True)
    by_t = df.groupby("type").agg(n=("entry", "size"), **{k: (k, v) for k, v in agg.items()}).reset_index()
    by_t = by_t.sort_values("type", key=lambda s: s.map(CLASS_TO_IDX)).reset_index(drop=True)
    cols = [(f"w_{c}", b) for c, b in zip(CLASSES, BRANCHES)] + [("entropy", "Entropy"), ("correct", "Top-1")]
    fmt = {k: "{:.3f}" for k, _ in cols}
    for name, table, first in (("routing_by_type_level", by_tl, [("type", "Input"), ("level", "Level")]),
                               ("routing_by_type", by_t, [("type", "Input")])):
        rows = table.to_dict("records")
        save_csv(rows, tables_dir / f"{name}.csv")
        save_latex_table(rows, first + [("n", "n")] + cols, tables_dir / f"{name}.tex",
                         caption="Mean routing weight of each branch per true input type"
                                 + (" and severity" if "level" in table else "")
                                 + "; entropy normalised by log 4; Top-1: largest weight on the matching branch.",
                         label=f"tab:task3_{name}", fmt=fmt)
    return by_tl, by_t


def expert_activity(df, stats, tables_dir):
    """Is any expert inactive, or does one dominate unrelated inputs? One row per branch."""
    labels = df["type"].map(CLASS_TO_IDX).to_numpy()
    w = df[WEIGHT_COLS].to_numpy()
    dominant = w.argmax(axis=1)
    rows = []
    for k, (cls, branch) in enumerate(zip(CLASSES, BRANCHES)):
        others = [c for c in range(len(CLASSES)) if c != k]
        worst = max(others, key=lambda c: stats["matrix"][c, k])
        on_others = labels != k
        rows.append({
            "branch": branch, "usage": float(stats["usage"][k]), "own_class": float(stats["own"][k]),
            "off_class": float(stats["off_class"][k]), "max_off_class": float(stats["matrix"][worst, k]),
            "max_off_class_type": CLASSES[worst],
            "dominant_share": float((dominant == k).mean()),
            "dominant_on_other_classes": float((dominant[on_others] == k).mean()) if on_others.any() else float("nan"),
            "w_above_0.5_on_other_classes": float((w[on_others, k] > 0.5).mean()) if on_others.any() else float("nan"),
        })
    save_csv(rows, tables_dir / "expert_activity.csv")
    save_latex_table(rows, [("branch", "Branch"), ("usage", "Usage"), ("own_class", "Own"), ("off_class", "Others"),
                            ("max_off_class", "Max other"), ("max_off_class_type", "on"),
                            ("dominant_on_other_classes", "Dom. others")],
                     tables_dir / "expert_activity.tex",
                     caption="Expert activity on the test set. Usage: class-balanced mean weight (1/4 when balanced); "
                             "Own / Others: mean weight on inputs of the branch's own class / the other three classes; "
                             "Max other: the largest mean weight on one other class; Dom. others: share of other-class "
                             "inputs where this branch has the largest weight.",
                     label="tab:task3_expert_activity",
                     fmt={k: "{:.3f}" for k in ("usage", "own_class", "off_class", "max_off_class",
                                                 "dominant_on_other_classes")})
    return rows


def entropy_table(df, tables_dir):
    rows = []
    for t in (*CLASSES, "all"):
        e = df["entropy"] if t == "all" else df.loc[df["type"] == t, "entropy"]
        rows.append({"type": t, "n": len(e), "mean": float(e.mean()), "median": float(e.median()),
                     "p90": float(e.quantile(0.9)), "share_above_0.5": float((e > 0.5).mean())})
    save_csv(rows, tables_dir / "routing_entropy.csv")
    save_latex_table(rows, [("type", "Input"), ("n", "n"), ("mean", "Mean"), ("median", "Median"), ("p90", "P90"),
                            ("share_above_0.5", "$>0.5$")], tables_dir / "routing_entropy.tex",
                     caption="Routing entropy normalised by log 4 (0: one branch, 1: uniform weights).",
                     label="tab:task3_routing_entropy",
                     fmt={k: "{:.3f}" for k in ("mean", "median", "p90", "share_above_0.5")})
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _with_weights(samples, entries, records):
    for s, e in zip(samples, entries):
        w = "/".join(f"{records[e][c]:.2f}" for c in WEIGHT_COLS)
        s["label"] = f"{s['label']}\nw={w}"
    return samples


def restoration_figures(model, dataset, records, fig_dir, device):
    predict = make_predict_fn(model)
    chosen = representative_entries(records, per_group=2)
    corrupted = [t for t in CLASSES if t != "clean"]
    worst = worst_entries(records, n=4, types=corrupted)
    misrouted = [r for r in records if r["dominant"] != r["type"]]
    figures = {"examples_a": chosen[0::2], "examples_b": chosen[1::2], "failures": worst,
               "failures_misrouted": [r["entry"] for r in sorted(misrouted, key=lambda r: r["ssim"])[:6]]}
    for name, entries in figures.items():
        if entries:
            samples = _with_weights(predict_entries(predict, dataset, entries, device), entries, records)
            save_restoration_figure(samples, fig_dir / f"{name}.png")
    return figures


def gating_figures(model, dataset, records, df, by_tl, stats, fig_dir, device):
    labels = df["type"].map(CLASS_TO_IDX).to_numpy()
    w = df[WEIGHT_COLS].to_numpy()
    rows = [f"{t} {lv}" if lv != "none" else t for t, lv in zip(by_tl["type"], by_tl["level"])]
    save_heatmap(by_tl[WEIGHT_COLS].to_numpy(), rows, fig_dir / "routing_heatmap_type_level.png",
                 title="Mean routing weight per true corruption type and severity")
    save_heatmap(stats["matrix"], CLASSES, fig_dir / "routing_matrix.png", title="Mean routing weight per true class")
    save_weight_distributions(w, labels, fig_dir / "weight_distributions.png")
    save_entropy_histogram(df["entropy"].to_numpy(), labels, fig_dir / "routing_entropy.png")

    # One expert dominates: per true class, the entry with the lowest routing entropy
    dominant = [int(df.loc[df["type"] == t, "entropy"].idxmin()) for t in CLASSES if (df["type"] == t).any()]
    # Weights spread across experts: the highest-entropy entries, at most one per test image
    spread = []
    for i in df.sort_values("entropy", ascending=False).index:
        if df.at[i, "image_idx"] not in {df.at[j, "image_idx"] for j in spread}:
            spread.append(int(i))
        if len(spread) == 6:
            break
    entries = {"gating_dominant": dominant, "gating_spread": spread}
    titles = {"gating_dominant": "One branch dominates (lowest routing entropy per true class)",
              "gating_spread": "Weights spread across branches (highest routing entropy)"}
    for name, chosen in entries.items():
        save_branch_figure(branch_samples(model, dataset, chosen, records, device), fig_dir / f"{name}.png",
                           title=titles[name])
    return entries


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate(checkpoint=None, smoke=False, batch_size=64, device=None):
    device = device or get_device()
    checkpoint = checkpoint or checkpoint_dir(smoke) / "best.pt"
    model, ckpt = load_moe(checkpoint, device)
    images, manifests = load_pets(smoke=smoke)
    dataset = ManifestDataset(images["test"], manifests["test"])
    out = output_dir(smoke)
    eval_dir, tables_dir, fig_dir = out / "eval", out / "tables", out / "figures"
    print(f"[task3] evaluating {checkpoint} (epoch {ckpt.get('epoch')}, tau {model.config['tau']}) on "
          f"{len(dataset)} test entries, float32 on {device}", flush=True)

    with torch.no_grad():
        records = evaluate_restoration(make_predict_fn(model), dataset, device, batch_size=batch_size,
                                       num_workers=2 if device.type == "cuda" else 0)
    save_csv(records, eval_dir / "test_records.csv")
    df = pd.DataFrame(records)
    tables = restoration_tables(records, tables_dir)
    labels = df["type"].map(CLASS_TO_IDX).to_numpy()
    stats = routing_stats(df[WEIGHT_COLS].to_numpy(), labels)
    by_tl, by_t = routing_tables(df, tables_dir)
    activity = expert_activity(df, stats, tables_dir)
    entropy = entropy_table(df, tables_dir)
    cfg = ckpt.get("config") or {}
    collapsed, reason = detect_collapse(stats, cfg.get("collapse_min_usage", 0.05),
                                        cfg.get("collapse_max_off_class", 0.5))

    with torch.no_grad():
        figures = restoration_figures(model, dataset, records, fig_dir, device)
        gating = gating_figures(model, dataset, records, df, by_tl, stats, fig_dir, device)

    corrupted = df[df["type"] != "clean"]
    summary = {
        "checkpoint": str(checkpoint), "epoch": ckpt.get("epoch"), "tau": model.config["tau"],
        "n_entries": len(df),
        "metrics": {
            "all": {"psnr": float(df["psnr"].mean()), "ssim": float(df["ssim"].mean())},
            "corrupted": {"psnr": float(corrupted["psnr"].mean()), "ssim": float(corrupted["ssim"].mean()),
                          "restoration_score": restoration_score(float(corrupted["psnr"].mean()),
                                                                 float(corrupted["ssim"].mean()))},
            "by_type": {r["type"]: {k: r[k] for k in ("psnr", "psnr_finite", "ssim", "n_exact")}
                        for r in tables["by_type"]},
        },
        "routing": {
            "matrix_true_x_branch": stats["matrix"].round(4).tolist(),
            "usage": dict(zip(BRANCHES, np.round(stats["usage"], 4).tolist())),
            "off_class": dict(zip(BRANCHES, np.round(stats["off_class"], 4).tolist())),
            "top1_accuracy": stats["accuracy"], "mean_entropy": stats["entropy"],
            "collapsed": collapsed, "collapse_reason": reason,
            "expert_activity": activity, "entropy": entropy,
        },
        "figure_entries": {**figures, **gating},
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[task3] corrupted inputs: PSNR {summary['metrics']['corrupted']['psnr']:.2f} dB, SSIM "
          f"{summary['metrics']['corrupted']['ssim']:.4f} | routing top-1 {stats['accuracy']:.3f}, usage "
          + "/".join(f"{u:.3f}" for u in stats["usage"]) + f" | collapse: {reason or 'none'}")
    print(f"[task3] wrote {eval_dir}, {tables_dir}, {fig_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="default: CKPT_DIR/task3/moe/best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="synthetic test set and the smoke checkpoint")
    args = parser.parse_args(argv)
    return evaluate(args.checkpoint, smoke=args.smoke, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
