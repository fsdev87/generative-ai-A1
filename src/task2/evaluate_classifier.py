"""Evaluate the Task 2 corruption classifier on the test manifest (and on validation for reference).

Writes to OUTPUT_DIR/task2/:
  tables/classifier_metrics.json            accuracy, macro P/R/F1, per-class metrics, confusion (test and val)
  tables/classifier_{test,val}_per_class.csv/.tex
  tables/classifier_{test,val}_confusion.csv  row-normalised (rows = true class) and counts
  tables/classifier_test_by_type_level.csv/.tex  accuracy and prediction shares per type x severity
  tables/classifier_blur_strength.csv       blur detection rate vs actual blur strength (test levels, val bins)
  tables/classifier_clean_sharpness.csv     clean<->blur: false "blur" rate of clean photos by natural sharpness
  tables/classifier_dark_regions.csv        clean<->occlusion vs dark image content
  tables/classifier_test_predictions.csv    one row per test entry (probabilities and analysis covariates)
  figures/classifier_confusion_{test,val}.png, classifier_blur_detection.png,
  figures/classifier_sharpness.png, classifier_misclassified.png

Analysis covariates:
  blur_strength  std (px) of the applied truncated Gaussian kernel (src.data.corruptions.blur_strength)
  sharpness      variance of the Laplacian of the grey-scale input (a standard focus measure)
  dark_fraction  share of near-black pixels (all channels <= 0.1) in the clean image
  hidden_fraction  for occlusion: share of the occluded area that was already near-black,
                   i.e. where a black rectangle is almost invisible

    python -m src.task2.evaluate_classifier --smoke
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.checkpoint import build_model
from src.common.evaluation import save_latex_table
from src.common.paths import get_dir
from src.data.corruptions import CLASSES, LEVELS, blur_strength
from src.data.pets import ManifestDataset, load_pets
from src.models.classifier import CorruptionClassifier

from .common import (TASK, classification_metrics, confusion_figure, disable_wandb_for_smoke,
                     require_real_checkpoint, save_figure, setup_device)

DARK_LEVEL = 0.1                      # near-black: every channel <= 0.1 (about 25/255)
DARK_BINS = [0, 0.05, 0.15, 0.30, 1.0001]
HIDDEN_BINS = [0, 0.1, 0.3, 0.6, 1.0001]
CLASS_COLORS = {"clean": "#2a78d6", "salt": "#eb6834", "blur": "#1baf7a", "occlusion": "#eda100"}
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def sharpness(x):
    """Variance of the 3x3 Laplacian of the grey-scale image (valid region), per image."""
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return F.conv2d(gray, LAPLACIAN.to(x.device)).flatten(1).var(dim=1)


def wilson(k, n, z=1.96):
    """95% Wilson score interval of a binomial proportion (sensible also for small n or p near 0/1)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


@torch.no_grad()
def predict_split(model, images, manifest, device, batch_size=128, num_workers=0):
    """One row per manifest entry: true/predicted class, probabilities and the analysis covariates."""
    dataset = ManifestDataset(images, manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, sharp = [], []
    model.eval()
    for batch in loader:
        x = batch["input"].to(device)
        probs.append(torch.softmax(model(x).float(), dim=1).cpu())
        sharp.append(sharpness(x).cpu())
    probs, sharp = torch.cat(probs).numpy(), torch.cat(sharp).numpy()

    dark = images.max(axis=3) <= round(DARK_LEVEL * 255)  # (N, H, W) near-black pixels of the clean images
    rows = []
    for i, e in enumerate(dataset.entries):
        row = {"entry": i, "image_idx": e["image_idx"], "image_id": e.get("image_id"), "type": e["type"],
               "level": e["level"] or "none", "true_label": CLASSES.index(e["type"]),
               "pred_label": int(probs[i].argmax()), "pred_class": CLASSES[int(probs[i].argmax())],
               **{f"prob_{c}": float(probs[i, k]) for k, c in enumerate(CLASSES)},
               "sharpness": float(sharp[i]), "dark_fraction": float(dark[e["image_idx"]].mean()),
               "blur_strength": blur_strength(e["k"], e["sigma"]) if e["type"] == "blur" else np.nan,
               "hidden_fraction": np.nan}
        if e["type"] == "occlusion":
            mask = np.zeros(dark.shape[1:], dtype=bool)
            for y, x0, h, w in e["rects"]:
                mask[y:y + h, x0:x0 + w] = True
            row["hidden_fraction"] = float(dark[e["image_idx"]][mask].mean())
        row["correct"] = int(row["pred_label"] == row["true_label"])
        rows.append(row)
    return pd.DataFrame(rows), dataset


def per_class_table(metrics):
    rows = [{"class": c, **m} for c, m in metrics["per_class"].items()]
    rows.append({"class": "macro avg", "precision": metrics["macro_precision"], "recall": metrics["macro_recall"],
                 "f1": metrics["macro_f1"], "support": metrics["n"]})
    rows.append({"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": metrics["accuracy"],
                 "support": metrics["n"]})
    return pd.DataFrame(rows)


def confusion_table(metrics):
    norm, counts = np.array(metrics["confusion_normalized"]), np.array(metrics["confusion"])
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({"true": c, **{f"pred_{p}": norm[i, j] for j, p in enumerate(CLASSES)},
                     **{f"count_{p}": int(counts[i, j]) for j, p in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def by_type_level(df):
    """Accuracy and the share of each predicted class per (type, severity)."""
    rows = []
    for (ctype, level), g in df.groupby(["type", "level"], sort=False):
        rows.append({"type": ctype, "level": level, "n": len(g), "accuracy": g["correct"].mean(),
                     **{f"pred_{c}": (g["pred_class"] == c).mean() for c in CLASSES}})
    order = [("clean", "none")] + [(t, lv) for t in CLASSES[1:] for lv in LEVELS]
    out = pd.DataFrame(rows)
    out["_order"] = [order.index(k) if k in order else len(order) for k in zip(out["type"], out["level"])]
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def rate_row(g, source, label, strength):
    n, k_blur, k_clean = len(g), int((g["pred_class"] == "blur").sum()), int((g["pred_class"] == "clean").sum())
    lo, hi = wilson(k_blur, n)
    return {"source": source, "group": label, "n": n, "blur_strength": strength,
            "p_pred_blur": k_blur / n if n else np.nan, "ci_low": lo, "ci_high": hi,
            "p_pred_clean": k_clean / n if n else np.nan}


def blur_strength_table(test_df, val_df, n_bins=6):
    """Detection rate P(pred = blur) as a function of the actual blur strength.

    Test: the three fixed levels. Validation: blur strengths sampled from the training
    ranges, in quantile bins. Strength 0 = clean images (the false-alarm rate).
    """
    rows = []
    for source, df in (("test", test_df), ("val", val_df)):
        rows.append(rate_row(df[df["type"] == "clean"], source, "clean", 0.0))
        blur = df[df["type"] == "blur"]
        if source == "test":
            for lv in LEVELS:
                g = blur[blur["level"] == lv]
                if len(g):
                    rows.append(rate_row(g, source, lv, float(g["blur_strength"].mean())))
        elif len(blur):
            bins = pd.qcut(blur["blur_strength"], q=min(n_bins, blur["blur_strength"].nunique()), duplicates="drop")
            for interval, g in blur.groupby(bins, observed=True):
                rows.append(rate_row(g, source, f"{interval.left:.2f}-{interval.right:.2f}",
                                     float(g["blur_strength"].mean())))
    return pd.DataFrame(rows)


def clean_sharpness_table(df, n_bins=4):
    """Clean photos by natural sharpness quartile: how often each is mistaken for blur."""
    clean = df[df["type"] == "clean"]
    if clean.empty:
        return pd.DataFrame()
    bins = pd.qcut(clean["sharpness"], q=min(n_bins, clean["sharpness"].nunique()), duplicates="drop")
    rows = []
    for k, (interval, g) in enumerate(clean.groupby(bins, observed=True)):
        lo, hi = wilson(int((g["pred_class"] == "blur").sum()), len(g))
        rows.append({"sharpness_quantile_bin": k + 1, "sharpness_low": interval.left, "sharpness_high": interval.right,
                     "n": len(g), "p_pred_blur": (g["pred_class"] == "blur").mean(), "ci_low": lo, "ci_high": hi,
                     "p_pred_clean": (g["pred_class"] == "clean").mean()})
    return pd.DataFrame(rows)


def dark_region_table(df):
    """clean -> occlusion false alarms vs dark_fraction; missed occlusions vs hidden_fraction."""
    rows = []
    clean = df[df["type"] == "clean"]
    for interval, g in clean.groupby(pd.cut(clean["dark_fraction"], DARK_BINS, right=False), observed=True):
        k = int((g["pred_class"] == "occlusion").sum())
        lo, hi = wilson(k, len(g))
        rows.append({"population": "clean", "covariate": "dark_fraction", "bin": f"[{interval.left:.2f}, {min(interval.right, 1):.2f})",
                     "n": len(g), "rate": k / len(g), "rate_meaning": "P(pred=occlusion)", "ci_low": lo, "ci_high": hi,
                     "p_pred_clean": (g["pred_class"] == "clean").mean()})
    occ = df[df["type"] == "occlusion"]
    for interval, g in occ.groupby(pd.cut(occ["hidden_fraction"], HIDDEN_BINS, right=False), observed=True):
        k = int((g["pred_class"] == "occlusion").sum())
        lo, hi = wilson(k, len(g))
        rows.append({"population": "occlusion", "covariate": "hidden_fraction",
                     "bin": f"[{interval.left:.2f}, {min(interval.right, 1):.2f})", "n": len(g), "rate": k / len(g),
                     "rate_meaning": "P(pred=occlusion)", "ci_low": lo, "ci_high": hi,
                     "p_pred_clean": (g["pred_class"] == "clean").mean()})
    return pd.DataFrame(rows)


def blur_figure(table):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    styles = {"val": dict(color="#2a78d6", marker="o", label="validation\n(training-range blur,\nquantile bins)"),
              "test": dict(color="#eb6834", marker="s", label="test (fixed levels)")}
    for source, style in styles.items():
        t = table[table["source"] == source].sort_values("blur_strength")
        if t.empty:
            continue
        # clip: at p = 0 or 1 the interval bound can sit a rounding error past p, and errorbar rejects < 0
        yerr = np.clip(np.vstack([t["p_pred_blur"] - t["ci_low"], t["ci_high"] - t["p_pred_blur"]]), 0, None)
        ax.errorbar(t["blur_strength"], t["p_pred_blur"], yerr=yerr, lw=2, ms=8, capsize=3, **style)
        ax.plot(t["blur_strength"], t["p_pred_clean"], lw=1.5, ls="--", color=style["color"], alpha=0.8)
    ax.text(1.03, 0.35, "solid: P(pred = blur)\nwith 95% Wilson CI\ndashed: P(pred = clean)\nstrength 0 = clean images",
            transform=ax.transAxes, fontsize=8, color="#52514e", va="top")
    ax.set_xlabel("Blur strength: std of the applied kernel (px)")
    ax.set_ylabel("Share of inputs")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Blur detection vs actual blur strength", fontsize=10)
    fig.tight_layout()
    return fig


def sharpness_figure(df):
    """Distribution of input sharpness: clean photos vs the three blur test levels."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    values = np.log10(df["sharpness"].clip(lower=1e-8))
    bins = np.linspace(values.min(), values.max() + 1e-9, 40)
    groups = [("clean", df["type"] == "clean", "#2a78d6")]
    groups += [(f"blur {lv}", (df["type"] == "blur") & (df["level"] == lv), c)
               for lv, c in zip(LEVELS, ("#1baf7a", "#eda100", "#eb6834"))]
    for label, mask, color in groups:
        if mask.any():
            ax.hist(values[mask], bins=bins, histtype="step", lw=2, color=color, label=label, density=True)
    clean_blur = (df["type"] == "clean") & (df["pred_class"] == "blur")
    if clean_blur.sum() > 1:
        ax.hist(values[clean_blur], bins=bins, color="#2a78d6", alpha=0.25, density=True,
                label="clean, predicted blur")
    ax.set_xlabel("log10 sharpness (variance of the Laplacian of the input)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Natural sharpness of clean photos overlaps mild blur", fontsize=10)
    fig.tight_layout()
    return fig


def pick_errors(df, n=16):
    """Most confident errors, taken round-robin over (true, predicted) pairs so every confusion is shown."""
    errors = df[df["correct"] == 0].copy()
    if errors.empty:
        return []
    errors["confidence"] = errors[[f"prob_{c}" for c in CLASSES]].max(axis=1)
    queues = [g.sort_values("confidence", ascending=False)["entry"].tolist()
              for _, g in errors.groupby(["type", "pred_class"])]
    queues.sort(key=len, reverse=True)
    chosen = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop(0))
    return chosen


def misclassified_figure(df, dataset, entries, ncols=4):
    plt = _mpl()
    nrows = (len(entries) + ncols - 1) // ncols
    fig = plt.figure(figsize=(2.3 * ncols, 3.3 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.35)
    rows = df.set_index("entry")
    for k, entry in enumerate(entries):
        r, c = divmod(k, ncols)
        row = rows.loc[entry]
        cell = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1.1], hspace=0.08)
        ax = fig.add_subplot(cell[0])
        ax.imshow(dataset[entry]["input"].permute(1, 2, 0).numpy().clip(0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        level = "" if row["level"] == "none" else f" ({row['level']})"
        ax.set_title(f"true {row['type']}{level}\npred {row['pred_class']}", fontsize=8)
        bar_ax = fig.add_subplot(cell[1])
        probs = [row[f"prob_{cl}"] for cl in CLASSES]
        bar_ax.barh(range(len(CLASSES)), probs, color=[CLASS_COLORS[cl] for cl in CLASSES], height=0.7)
        bar_ax.set_yticks(range(len(CLASSES)), [cl[:5] for cl in CLASSES], fontsize=6)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(0, 1)
        bar_ax.tick_params(axis="x", labelsize=6)
        bar_ax.spines[["top", "right"]].set_visible(False)
        for i, p in enumerate(probs):
            bar_ax.text(min(p + 0.02, 0.8), i, f"{p:.2f}", va="center", fontsize=6, color="#0b0b0b")
    fig.suptitle("Misclassified test inputs (most confident errors per confusion pair) with class probabilities",
                 fontsize=9)
    return fig


def save_table(df, path, latex=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if latex and not df.empty:
        columns, caption, label = latex
        rows = [{k: ("--" if pd.isna(v) else float(v)) if isinstance(v, (float, np.floating)) else v
                 for k, v in r.items()} for r in df.to_dict("records")]
        save_latex_table(rows, columns, path.with_suffix(".tex"), caption=caption, label=label,
                         fmt={k: "{:.4f}" for k, _ in columns if k not in ("n", "support")})


def evaluate(smoke=False, num_workers=None, n_examples=16):
    disable_wandb_for_smoke(smoke)
    device = setup_device()
    workers = num_workers if num_workers is not None else (2 if device.type == "cuda" else 0)
    ckpt = get_dir("CKPT_DIR", TASK, "classifier") / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found: train the classifier first (src.task2.train_classifier)")
    require_real_checkpoint(ckpt, smoke)
    model = build_model(CorruptionClassifier, ckpt).to(device).eval()
    images, manifests = load_pets(smoke=smoke)
    tables, figures = get_dir("OUTPUT_DIR", TASK, "tables"), get_dir("OUTPUT_DIR", TASK, "figures")

    frames, summary = {}, {"checkpoint": str(ckpt)}
    for split in ("test", "val"):
        df, dataset = predict_split(model, images[split], manifests[split], device, num_workers=workers)
        metrics = classification_metrics(df["true_label"], df["pred_label"])
        frames[split], summary[split] = (df, dataset), metrics
        cols = [("class", "Class"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("support", "n")]
        save_table(per_class_table(metrics), tables / f"classifier_{split}_per_class.csv",
                   (cols, f"Corruption classifier, {split} set: per-class metrics", f"tab:t2_clf_{split}"))
        save_table(confusion_table(metrics), tables / f"classifier_{split}_confusion.csv")
        save_figure(confusion_figure(metrics, f"Classifier, {split} set (row-normalised, counts in brackets)"),
                    figures / f"classifier_confusion_{split}.png")
        print(f"[classifier] {split}: n={metrics['n']} accuracy {metrics['accuracy']:.4f} "
              f"macro-F1 {metrics['macro_f1']:.4f} macro-P {metrics['macro_precision']:.4f} "
              f"macro-R {metrics['macro_recall']:.4f}")

    test_df, test_ds = frames["test"]
    val_df = frames["val"][0]
    test_df.to_csv(tables / "classifier_test_predictions.csv", index=False)
    tl = by_type_level(test_df)
    save_table(tl, tables / "classifier_test_by_type_level.csv",
               ([("type", "Type"), ("level", "Level"), ("n", "n"), ("accuracy", "Acc.")]
                + [(f"pred_{c}", f"P({c})") for c in CLASSES],
                "Classifier accuracy and prediction shares per corruption type and severity (test)",
                "tab:t2_clf_type_level"))
    blur = blur_strength_table(test_df, val_df)
    save_table(blur, tables / "classifier_blur_strength.csv")
    save_figure(blur_figure(blur), figures / "classifier_blur_detection.png")
    clean_sharp = clean_sharpness_table(test_df)
    save_table(clean_sharp, tables / "classifier_clean_sharpness.csv")
    save_figure(sharpness_figure(test_df), figures / "classifier_sharpness.png")
    dark = dark_region_table(test_df)
    save_table(dark, tables / "classifier_dark_regions.csv")
    errors = pick_errors(test_df, n_examples)
    if errors:
        save_figure(misclassified_figure(test_df, test_ds, errors), figures / "classifier_misclassified.png")

    clean, blur_low = test_df[test_df["type"] == "clean"], test_df[(test_df["type"] == "blur") & (test_df["level"] == "low")]
    summary["analysis"] = {
        "clean_pred_blur": float((clean["pred_class"] == "blur").mean()) if len(clean) else None,
        "clean_pred_occlusion": float((clean["pred_class"] == "occlusion").mean()) if len(clean) else None,
        "blur_low_pred_clean": float((blur_low["pred_class"] == "clean").mean()) if len(blur_low) else None,
        "median_sharpness_clean_pred_clean": float(clean[clean["pred_class"] == "clean"]["sharpness"].median())
        if (clean["pred_class"] == "clean").any() else None,
        "median_sharpness_clean_pred_blur": float(clean[clean["pred_class"] == "blur"]["sharpness"].median())
        if (clean["pred_class"] == "blur").any() else None,
        "n_test_errors": int((test_df["correct"] == 0).sum()),
    }
    (tables / "classifier_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[classifier] analysis: {summary['analysis']}")
    print(f"[classifier] tables in {tables}, figures in {figures}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--n-examples", type=int, default=16, help="misclassified examples in the figure")
    args = parser.parse_args(argv)
    return evaluate(smoke=args.smoke, num_workers=args.num_workers, n_examples=args.n_examples)


if __name__ == "__main__":
    main()
