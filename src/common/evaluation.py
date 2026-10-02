"""Shared evaluation for the restoration tasks (1-3).

Per-entry metrics over a deterministic manifest, summary tables by corruption type
and severity, example selection and the Target | Input | Output | |Error| figure, so
the three restoration systems are evaluated and presented identically.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES, LEVELS

from .metrics import image_metrics

_LEVEL_ORDER = {"none": -1, **{lv: i for i, lv in enumerate(LEVELS)}, "all": len(LEVELS)}
_TYPE_ORDER = {**{t: i for i, t in enumerate(CLASSES)}, "all": len(CLASSES)}


def _to_python(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def _unpack(result):
    return result if isinstance(result, tuple) else (result, {})


@torch.no_grad()
def evaluate_restoration(predict_fn, dataset, device, batch_size=64, num_workers=0):
    """Run predict_fn over a ManifestDataset and return one record per entry.

    predict_fn(x) returns the restored batch, or (restored, extras) where extras maps
    names to per-sample values (e.g. routing weights), copied into the records.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    records = []
    for batch in loader:
        x, y = batch["input"].to(device), batch["target"].to(device)
        output, extras = _unpack(predict_fn(x))
        metrics = image_metrics(output.clamp(0, 1), y)
        for i in range(x.shape[0]):
            record = {
                "entry": len(records),
                "image_idx": int(batch["image_idx"][i]),
                "type": CLASSES[int(batch["label"][i])],
                "level": batch["level"][i],
                **{k: float(v[i]) for k, v in metrics.items()},
            }
            record.update({k: _to_python(v[i]) for k, v in extras.items()})
            records.append(record)
    return records


def summarize(records, keys=("type", "level")):
    """Mean PSNR/SSIM/MSE and count per group of `keys` (use "all" rows via standard_tables)."""
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)

    def order(key):
        return tuple(_TYPE_ORDER.get(v, _LEVEL_ORDER.get(v, 99)) for v in key)

    rows = []
    for key in sorted(groups, key=order):
        rs = groups[key]
        rows.append({**dict(zip(keys, key)), "n": len(rs),
                     **{m: float(np.mean([r[m] for r in rs])) for m in ("psnr", "ssim", "mse")}})
    return rows


def standard_tables(records):
    """The three tables every restoration task reports.

    by_type: one row per corruption type plus "all";
    by_type_level: type x severity (clean has level "none");
    by_level: severity across the three corruptions (clean excluded).
    """
    everything = [{**r, "type": "all"} for r in records]
    corrupted = [r for r in records if r["type"] != "clean"]
    return {
        "by_type": summarize(records, ("type",)) + summarize(everything, ("type",)),
        "by_type_level": summarize(records, ("type", "level")),
        "by_level": summarize(corrupted, ("level",)),
    }


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_latex_table(rows, columns, path, caption="", label="", fmt=None):
    """Write a booktabs table. columns: list of (key, header); fmt: key -> format string."""
    fmt = {"psnr": "{:.2f}", "ssim": "{:.4f}", "mse": "{:.5f}", **(fmt or {})}
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * 1 + "r" * (len(columns) - 1) + "}", r"\toprule",
        " & ".join(h for _, h in columns) + r" \\", r"\midrule",
    ]
    for row in rows:
        cells = [fmt[k].format(row[k]) if k in fmt and isinstance(row[k], float) else str(row[k]) for k, _ in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def representative_entries(records, per_group=1):
    """Entries closest to the median SSIM of each (type, level) group: typical, not cherry-picked."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["type"], r["level"])].append(r)
    chosen = []
    for key in sorted(groups, key=lambda k: (_TYPE_ORDER[k[0]], _LEVEL_ORDER[k[1]])):
        rs = groups[key]
        median = np.median([r["ssim"] for r in rs])
        chosen += [r["entry"] for r in sorted(rs, key=lambda r: abs(r["ssim"] - median))[:per_group]]
    return chosen


def worst_entries(records, n=4, metric="ssim", types=None):
    """The n lowest-scoring entries, optionally restricted to some corruption types."""
    pool = [r for r in records if types is None or r["type"] in types]
    return [r["entry"] for r in sorted(pool, key=lambda r: r[metric])[:n]]


def _hwc(t):
    return t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def predict_entries(predict_fn, dataset, entries, device):
    """Run predict_fn on selected dataset entries; returns figure-ready samples."""
    items = [dataset[i] for i in entries]
    x = torch.stack([it["input"] for it in items]).to(device)
    output, _ = _unpack(predict_fn(x))
    return [
        {"target": _hwc(it["target"]), "input": _hwc(it["input"]), "output": _hwc(output[k]),
         "label": f"{CLASSES[it['label']]} {it['level'] if it['level'] != 'none' else ''}".strip()}
        for k, it in enumerate(items)
    ]


def save_restoration_figure(samples, path, error_vmax=0.5):
    """Rows of Target | Input | Output | |Error| (mean absolute error over RGB, shared colour scale)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(samples), 4, figsize=(4 * 1.9, len(samples) * 1.95), squeeze=False)
    for r, s in enumerate(samples):
        panels = [s["target"], s["input"], s["output"], np.abs(s["output"] - s["target"]).mean(axis=2)]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            if c == 3:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=error_vmax)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["Target", "Input", "Output", "|Error|"][c], fontsize=9)
        if s.get("label"):
            axes[r, 0].set_ylabel(s["label"], fontsize=8)
    fig.colorbar(im, ax=axes[:, 3].tolist(), fraction=0.05, pad=0.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
