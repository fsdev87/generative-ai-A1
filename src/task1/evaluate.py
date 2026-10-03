"""Task 1: evaluate a UDAE checkpoint on the official test manifest (36,690 entries).

Outputs (main model "udae" -> OUTPUT_DIR/task1/; ablation variants -> OUTPUT_DIR/task1/variants/<name>/):
  eval/test_records.csv        one row per test entry (evaluate_restoration columns)
  eval/summary.json            headline numbers, model/bottleneck info, diagnostics
  eval/diagnostics.json        salt-and-pepper impulse survival and clean-image detail ratio
  tables/by_type|by_type_level|by_level.{csv,tex}   standard tables (shared with Tasks 2-3)
  tables/comparison_by_type|comparison_by_type_level.{csv,tex}   model vs identity vs oracle classical
  figures/representative_examples_{1,2}.png   12 typical examples (3 clean + 1 per corruption x severity)
  figures/failure_cases.png             worst example per input condition + largest losses vs the input
  figures/severity_curves.png           PSNR and SSIM vs severity per corruption type
  figures/training_curves.png           loss and validation metrics per epoch (from last.pt)
Baseline records are computed once and cached in OUTPUT_DIR/task1/eval/baselines/.

    python -m src.task1.evaluate                          # main model
    python -m src.task1.evaluate --variant udae_skip16    # an ablation variant
    python -m src.task1.evaluate --compare udae udae_skip16 udae_skip32 udae_skip128
    python -m src.task1.evaluate --smoke
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.checkpoint import build_model, load_checkpoint
from src.common.evaluation import (
    evaluate_restoration, predict_entries, representative_entries, save_csv, save_latex_table,
    save_restoration_figure, standard_tables, summarize,
)
from src.common.metrics import PSNR_CAP_DB, restoration_score
from src.common.utils import count_parameters, get_device
from src.data.corruptions import CLASSES, LEVELS
from src.data.pets import ManifestDataset, load_pets
from src.models.autoencoder import ConvAutoencoder

from .baselines import OracleClassicalDataset
from .config import MAIN_VARIANT, bottleneck_info, checkpoint_dir, output_dir, resolve_num_workers, task_dir

METHODS = {"identity": "Corrupted input", "oracle_classical": "Oracle classical", "model": "UDAE"}
CORRUPTIONS = CLASSES[1:]
# Categorical slots 1-4 of the reference palette (fixed order, one hue per entity)
COLORS = {"model": "#2a78d6", "oracle_classical": "#eb6834", "identity": "#1baf7a"}
CONDITION_COLORS = dict(zip(CLASSES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# Records and tables
# --------------------------------------------------------------------------- #
def read_records(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(entry=int(r["entry"]), image_idx=int(r["image_idx"]),
                 **{m: float(r[m]) for m in ("psnr", "ssim", "mse")})
    return rows


def with_corrupted_row(records, keys=("type",)):
    """summarize() by type plus one 'corrupted' row pooling salt, blur and occlusion."""
    pooled = [{**r, "type": "corrupted"} for r in records if r["type"] != "clean"]
    return summarize(records, keys) + summarize(pooled, keys)


def write_table(rows, out_dir, name, columns, caption, fmt=None):
    save_csv(rows, out_dir / f"{name}.csv")
    save_latex_table(rows, columns, out_dir / f"{name}.tex", caption=caption, label=f"tab:task1_{name}", fmt=fmt)


def write_standard_tables(records, tables_dir, title):
    tables = standard_tables(records)
    cols = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("n", "$n$")]
    write_table(tables["by_type"], tables_dir, "by_type", [("type", "Input")] + cols,
                f"{title}: test results per input condition.")
    write_table(tables["by_type_level"], tables_dir, "by_type_level",
                [("type", "Input"), ("level", "Severity")] + cols,
                f"{title}: test results per input condition and severity.")
    write_table(tables["by_level"], tables_dir, "by_level", [("level", "Severity")] + cols,
                f"{title}: test results per severity (salt, blur and occlusion pooled).")
    return tables


def comparison_rows(method_records, keys):
    """Wide rows: one per group, PSNR and SSIM of every method side by side."""
    per_method = {m: {tuple(r[k] for k in keys): r for r in with_corrupted_row(recs, keys)}
                  for m, recs in method_records.items()}
    rows = []
    for key in per_method["model"]:
        row = dict(zip(keys, key))
        for m in method_records:
            row[f"{m}_psnr"] = per_method[m][key]["psnr"]
            row[f"{m}_ssim"] = per_method[m][key]["ssim"]
        row["n"] = per_method["model"][key]["n"]
        rows.append(row)
    return rows


def write_comparison_tables(method_records, tables_dir):
    caption = ("Test PSNR (dB) / SSIM of the UDAE against the corrupted input (no restoration) and classical "
               "oracle baselines that are told the corruption (3x3 median filter for salt-and-pepper, unsharp "
               "masking with the known blur kernel, Telea inpainting of the known occlusion mask); the UDAE "
               "receives no such information. Clean inputs are returned unchanged by both baselines "
               r"(exact, PSNR $=\infty$, stored as the 100 dB cap in the CSV), so the pooled row "
               "covers the three corruptions only.")
    fmt = {f"{m}_{q}": ("{:.2f}" if q == "psnr" else "{:.4f}") for m in method_records for q in ("psnr", "ssim")}
    for keys, name in ((("type",), "comparison_by_type"), (("type", "level"), "comparison_by_type_level")):
        rows = comparison_rows(method_records, keys)
        save_csv(rows, tables_dir / f"{name}.csv")
        latex_rows = [{k: (r"$\infty$" if k.endswith("_psnr") and v >= PSNR_CAP_DB - 1e-6 else v)
                       for k, v in r.items()} for r in rows]
        columns = [(k, k.capitalize() if k == "type" else "Severity") for k in keys]
        columns += [(f"{m}_{q}", f"{METHODS[m]} {q.upper()}") for m in method_records for q in ("psnr", "ssim")]
        save_latex_table(latex_rows, columns, tables_dir / f"{name}.tex", caption=caption,
                         label=f"tab:task1_{name}", fmt=fmt)


# --------------------------------------------------------------------------- #
# Baselines (model independent, cached)
# --------------------------------------------------------------------------- #
def baseline_records(images, manifest, device, batch_size, workers, smoke, recompute=False):
    cache = task_dir("OUTPUT_DIR", smoke, "eval", "baselines")
    datasets = {"identity": ManifestDataset(images, manifest),
                "oracle_classical": OracleClassicalDataset(images, manifest)}
    records = {}
    for name, dataset in datasets.items():
        path = cache / f"{name}_records.csv"
        if path.exists() and not recompute:
            records[name] = read_records(path)
            if len(records[name]) == len(dataset):
                continue
        print(f"[eval] baseline {name} on {len(dataset)} test entries", flush=True)
        records[name] = evaluate_restoration(lambda x: x, dataset, device, batch_size, workers)
        save_csv(records[name], path)
    return records


# --------------------------------------------------------------------------- #
# Diagnostics for the skip-connection ablation
# --------------------------------------------------------------------------- #
_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def hf_energy(x):
    """Mean absolute Laplacian of the luminance per image: the amount of fine detail."""
    luma = (x * _LUMA.to(x.device)).sum(1, keepdim=True)
    return F.conv2d(luma, _LAPLACE.to(x.device)).abs().flatten(1).mean(1)


@torch.no_grad()
def leakage_diagnostics(predict_fn, images, manifest, device, batch_size, workers):
    """Two numbers that show what skip connections do (see docs/task1_notes.md).

    impulse survival (salt-and-pepper entries): among pixels hit by an impulse (input pure
    black/white in all channels and far from the clean value), the fraction where the output
    is still closer to the impulse than to the clean value. Identity = 1, perfect = 0.
    detail ratio (clean entries): hf_energy(output) / hf_energy(target); below 1 means the
    output is smoother than the clean image (detail lost in the bottleneck).
    """
    loader = DataLoader(ManifestDataset(images, manifest, types=["clean", "salt"]), batch_size=batch_size,
                        num_workers=workers)
    hits, survived, ratios, labels, levels = [], [], [], [], []
    for batch in loader:
        x, y = batch["input"].to(device), batch["target"].to(device)
        out = predict_fn(x).float().clamp(0, 1)
        impulse = ((x == 0).all(1) | (x == 1).all(1)) & ((x - y).abs().mean(1) > 0.25)
        closer = (out - x).abs().mean(1) < (out - y).abs().mean(1)
        hits.append(impulse.flatten(1).sum(1).cpu())
        survived.append((impulse & closer).flatten(1).sum(1).cpu())
        ratios.append((hf_energy(out) / hf_energy(y).clamp_min(1e-8)).cpu())
        labels += batch["label"].tolist()
        levels += list(batch["level"])
    hits, survived, ratios = torch.cat(hits), torch.cat(survived), torch.cat(ratios)
    labels, levels = np.array(labels), np.array(levels)
    salt, clean = labels == CLASSES.index("salt"), labels == CLASSES.index("clean")

    def survival(mask):
        n = int(hits[torch.from_numpy(mask)].sum())
        return float(survived[torch.from_numpy(mask)].sum()) / n if n else None

    return {
        "salt_impulse_survival": {**{lv: survival(salt & (levels == lv)) for lv in LEVELS}, "all": survival(salt)},
        "clean_detail_ratio": float(ratios[torch.from_numpy(clean)].mean()) if clean.any() else None,
        "n_salt_entries": int(salt.sum()), "n_clean_entries": int(clean.sum()),
    }


# --------------------------------------------------------------------------- #
# Example selection
# --------------------------------------------------------------------------- #
def pick_representative(records, n_clean=3):
    """Entries closest to the median SSIM of their (type, level) group: 3 clean + 1 per corruption x severity."""
    chosen = representative_entries(records, per_group=n_clean)
    types = {r["entry"]: (r["type"], r["level"]) for r in records}
    picked, seen = [], {}
    for e in chosen:
        group = types[e]
        limit = n_clean if group[0] == "clean" else 1
        if seen.get(group, 0) < limit:
            picked.append(e)
            seen[group] = seen.get(group, 0) + 1
    return picked


def pick_failures(records, identity_records, n_vs_input=2):
    """Different failure modes: the lowest-SSIM output for every input condition, plus the
    corrupted entries where the model gains least PSNR over (or even loses to) its input."""
    picks = []
    for t in CLASSES:
        pool = sorted((r for r in records if r["type"] == t), key=lambda r: r["ssim"])
        if pool:
            picks.append((pool[0]["entry"], "worst SSIM"))
    used = {e for e, _ in picks}
    gains = sorted((r["psnr"] - identity_records[r["entry"]]["psnr"], r["entry"])
                   for r in records if r["type"] != "clean" and r["entry"] not in used)
    for gain, entry in gains[:n_vs_input]:
        picks.append((entry, "below input" if gain < 0 else "least gain"))
    return picks


def _label(record, tag=None, input_psnr=None):
    """Two short lines (the row label is rotated): '[mode: ]type level' and 'PSNR dB [(input)] / SSIM'."""
    name = record["type"] + ("" if record["level"] == "none" else f" {record['level']}")
    first = f"{tag}: {name}" if tag else name
    shown_input = input_psnr is not None and input_psnr < PSNR_CAP_DB
    psnr_text = f"{record['psnr']:.1f} dB" + (f" (in {input_psnr:.1f})" if shown_input else "")
    return f"{first}\n{psnr_text} / {record['ssim']:.3f}"


def save_examples(predict_fn, dataset, records, identity, entries_with_tags, path, device):
    entries = [e for e, _ in entries_with_tags]
    samples = predict_entries(predict_fn, dataset, entries, device)
    for s, (e, tag) in zip(samples, entries_with_tags):
        s["label"] = _label(records[e], tag, identity[e]["psnr"])
    save_restoration_figure(samples, path)
    return samples


# --------------------------------------------------------------------------- #
# Charts (static PNGs for the report)
# --------------------------------------------------------------------------- #
def _style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_severity_curves(method_records, path):
    """PSNR and SSIM vs severity (small multiples: one panel per corruption type and metric)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = {m: {(r["type"], r["level"]): r for r in summarize(recs, ("type", "level"))}
              for m, recs in method_records.items()}
    clean = tables["model"].get(("clean", "none"))
    styles = {"model": ("-", "o"), "oracle_classical": ("--", "s"), "identity": (":", "^")}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    x = np.arange(len(LEVELS))
    for row, (metric, unit) in enumerate((("psnr", "PSNR (dB)"), ("ssim", "SSIM"))):
        for col, t in enumerate(CORRUPTIONS):
            ax = axes[row, col]
            _style_axis(ax)
            for m in ("identity", "oracle_classical", "model"):
                ys = [tables[m][(t, lv)][metric] for lv in LEVELS]
                ls, marker = styles[m]
                ax.plot(x, ys, ls, marker=marker, color=COLORS[m], linewidth=2, markersize=6.5,
                        markeredgecolor="white", markeredgewidth=1.2, label=METHODS[m])
            if clean is not None:
                ax.axhline(clean[metric], color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
                ax.annotate("UDAE on clean input", (x[-1], clean[metric]), xytext=(0, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=7, color=MUTED)
            if row == 0:
                ax.set_title({"salt": "Salt-and-pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}[t],
                             fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(unit, fontsize=9, color=INK)
            ax.set_xticks(x, [lv.capitalize() for lv in LEVELS])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("Test severity level", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves(history, best_epoch, path):
    """Train/val loss and per-condition validation PSNR/SSIM per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax in axes:
        _style_axis(ax)
        if best_epoch:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
        ax.set_xlabel("Epoch", fontsize=9, color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].plot(epochs, [h["train_loss"] for h in history], color=COLORS["model"], linewidth=2, label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "--", color=COLORS["oracle_classical"], linewidth=2,
                 label="Validation")
    axes[0].set_title("Loss  α·L1 + (1−α)(1−SSIM)", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    for ax, metric, title in ((axes[1], "psnr", "Validation PSNR (dB)"), (axes[2], "ssim", "Validation SSIM")):
        for c in CLASSES:
            key = f"val_{c}/{metric}"
            if key in history[0]:
                ax.plot(epochs, [h[key] for h in history], color=CONDITION_COLORS[c], linewidth=2, label=c)
        ax.set_title(title, fontsize=10, color=INK)
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Evaluation of one checkpoint
# --------------------------------------------------------------------------- #
def load_model(path, device):
    ckpt = load_checkpoint(path)
    return build_model(ConvAutoencoder, ckpt).to(device).eval(), ckpt


def evaluate_variant(variant=MAIN_VARIANT, checkpoint=None, smoke=False, batch_size=128, num_workers=2,
                     recompute_baselines=False, device=None, data=None, figures=True):
    device = device or get_device()
    workers = resolve_num_workers(num_workers, device)
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_dir(variant, smoke) / "best.pt"
    out = output_dir(variant, smoke)
    model, ckpt = load_model(ckpt_path, device)
    images, manifests = data or load_pets(smoke=smoke)
    test_ds = ManifestDataset(images["test"], manifests["test"])
    print(f"[eval] {variant}: {ckpt_path} (epoch {ckpt['epoch']}) on {len(test_ds)} test entries -> {out}",
          flush=True)

    def predict_fn(x):
        return model(x)  # float32: the same numbers the ONNX export reproduces

    records = evaluate_restoration(predict_fn, test_ds, device, batch_size, workers)
    save_csv(records, out / "eval" / "test_records.csv")
    title = "UDAE" if variant == MAIN_VARIANT else f"UDAE variant {variant}"
    tables = write_standard_tables(records, out / "tables", title)

    baselines = baseline_records(images["test"], manifests["test"], device, batch_size, workers, smoke,
                                 recompute_baselines)
    method_records = {**baselines, "model": records}
    write_comparison_tables(method_records, out / "tables")

    diagnostics = leakage_diagnostics(predict_fn, images["test"], manifests["test"], device, batch_size, workers)
    (out / "eval" / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    identity = baselines["identity"]
    rep = pick_representative(records)
    failures = pick_failures(records, identity)
    if figures:
        fig_dir = out / "figures"
        half = (len(rep) + 1) // 2  # two 6-row figures fit a page better than one 12-row figure
        for i, part in enumerate((rep[:half], rep[half:]), start=1):
            save_examples(predict_fn, test_ds, records, identity, [(e, None) for e in part],
                          fig_dir / f"representative_examples_{i}.png", device)
        save_examples(predict_fn, test_ds, records, identity, failures, fig_dir / "failure_cases.png", device)
        plot_severity_curves(method_records, fig_dir / "severity_curves.png")
        last = ckpt_path.parent / "last.pt"
        if last.exists():
            state = load_checkpoint(last)
            plot_training_curves(state["history"], state.get("best_epoch"), fig_dir / "training_curves.png")

    by_type = with_corrupted_row(records)
    for row in by_type:
        row["score"] = restoration_score(row["psnr"], row["ssim"])
    summary = {
        "variant": variant, "checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "val_metrics": ckpt["metrics"],
        "model_config": ckpt["model_config"], "training_config": ckpt.get("config"),
        "n_params": count_parameters(model), **bottleneck_info(ckpt["model_config"]),
        "n_test_entries": len(records), "by_type": by_type, "by_level": tables["by_level"],
        "by_type_level": tables["by_type_level"], "diagnostics": diagnostics,
        "representative_entries": rep, "failure_entries": [{"entry": e, "mode": tag} for e, tag in failures],
        "wandb_url": ckpt.get("wandb_url"),
    }
    (out / "eval" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    corrupted = next(r for r in by_type if r["type"] == "corrupted")
    print(f"[eval] {variant}: corrupted inputs PSNR {corrupted['psnr']:.2f} dB, SSIM {corrupted['ssim']:.4f}; "
          f"impulse survival {diagnostics['salt_impulse_survival']['all']}, clean detail ratio "
          f"{diagnostics['clean_detail_ratio']}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Skip-connection ablation: variants side by side
# --------------------------------------------------------------------------- #
def compare_variants(variants, smoke=False, device=None, data=None):
    device = device or get_device()
    summaries = {}
    for v in variants:
        path = output_dir(v, smoke) / "eval" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m src.task1.evaluate --variant {v}` first")
        summaries[v] = json.loads(path.read_text())

    rows = []
    for v, s in summaries.items():
        by_type = {r["type"]: r for r in s["by_type"]}
        survival = s["diagnostics"]["salt_impulse_survival"]
        row = {"variant": v, "skips": "+".join(map(str, s["model_config"]["skip_resolutions"])) or "none",
               "params_m": s["n_params"] / 1e6, "latent_dim": s["latent_dim"],
               "skip_values_vs_input": s["skip_values_vs_input"]}
        for t in (*CLASSES, "corrupted"):
            row[f"{t}_psnr"], row[f"{t}_ssim"] = by_type[t]["psnr"], by_type[t]["ssim"]
        row["impulse_survival_high"] = survival["high"]
        row["impulse_survival_all"] = survival["all"]
        row["clean_detail_ratio"] = s["diagnostics"]["clean_detail_ratio"]
        rows.append(row)
    tables = task_dir("OUTPUT_DIR", smoke, "tables")
    fmt = {k: "{:.2f}" for k in rows[0] if k.endswith("_psnr") or k in ("params_m", "skip_values_vs_input")}
    fmt.update({k: "{:.4f}" for k in rows[0] if k.endswith("_ssim")})
    fmt.update({k: "{:.3f}" for k in ("impulse_survival_high", "impulse_survival_all", "clean_detail_ratio")})
    columns = [("variant", "Model"), ("skips", "Skips"), ("skip_values_vs_input", "Skip values / input"),
               ("clean_psnr", "Clean"), ("salt_psnr", "Salt"), ("blur_psnr", "Blur"),
               ("occlusion_psnr", "Occl."), ("corrupted_ssim", "SSIM (corr.)"),
               ("impulse_survival_high", "Impulse surv. (high)"), ("clean_detail_ratio", "Detail ratio")]
    write_table(rows, tables, "skip_ablation", columns, fmt=fmt, caption=(
        "Skip-connection ablation on the test set. PSNR (dB) per input condition; SSIM pooled over the three "
        "corruptions; skip values / input: numbers carried by the skip paths per image relative to the "
        "3x128x128 input; impulse survival: fraction of salt-and-pepper pixels still closer to the impulse "
        "than to the clean value (high severity, identity = 1); detail ratio: Laplacian energy of the output "
        "relative to the clean target on clean inputs (1 = as detailed as the target)."))

    # Same entries for every variant: typical (median-SSIM) examples of the reference model
    reference = read_records(output_dir(variants[0], smoke) / "eval" / "test_records.csv")
    groups = [("clean", "none"), ("salt", "high"), ("blur", "medium"), ("occlusion", "high")]
    rep = representative_entries(reference, per_group=1)
    by_entry = {r["entry"]: r for r in reference}
    entries = [e for g in groups for e in rep if (by_entry[e]["type"], by_entry[e]["level"]) == g]
    images, manifests = data or load_pets(smoke=smoke)
    test_ds = ManifestDataset(images["test"], manifests["test"])
    outputs = {}
    for v in variants:
        model, _ = load_model(Path(summaries[v]["checkpoint"]), device)
        outputs[v] = [s["output"] for s in predict_entries(lambda x: model(x), test_ds, entries, device)]
    base = predict_entries(lambda x: x, test_ds, entries, device)
    path = task_dir("OUTPUT_DIR", smoke, "figures") / "skip_ablation_examples.png"
    _save_ablation_figure(base, outputs, path)
    print(f"[eval] skip ablation table -> {tables / 'skip_ablation.csv'}; figure -> {path}")
    return rows


def _save_ablation_figure(base, outputs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["Target", "Input"] + list(outputs)
    fig, axes = plt.subplots(len(base), len(titles), figsize=(1.9 * len(titles), 1.95 * len(base)), squeeze=False)
    for r, sample in enumerate(base):
        panels = [sample["target"], sample["input"]] + [outputs[v][r] for v in outputs]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(sample["label"], fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=MAIN_VARIANT,
                        help="checkpoint folder CKPT_DIR/task1/<variant>/best.pt (default udae)")
    parser.add_argument("--checkpoint", help="explicit checkpoint path (outputs still go to the variant folder)")
    parser.add_argument("--compare", nargs="+", metavar="VARIANT",
                        help="only build the side-by-side ablation table/figure from evaluated variants")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers on CUDA (0 on Windows/CPU)")
    parser.add_argument("--recompute-baselines", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="evaluate the smoke checkpoint on synthetic data")
    args = parser.parse_args(argv)
    if args.compare:
        return compare_variants(args.compare, smoke=args.smoke)
    return evaluate_variant(args.variant, args.checkpoint, smoke=args.smoke, batch_size=args.batch_size,
                            num_workers=args.num_workers, recompute_baselines=args.recompute_baselines or args.smoke)


if __name__ == "__main__":
    main()
