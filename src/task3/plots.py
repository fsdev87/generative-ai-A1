"""Figures of the Task 3 gating analysis (matplotlib, Agg backend, PNG at 200 dpi for the report).

Colours: one fixed categorical slot per branch (identity, salt, blur, occlusion) used in
every figure, a single-hue sequential ramp (Blues) for heatmaps, with every cell labelled
by its value so nothing is read from colour alone.
"""
from pathlib import Path

import numpy as np

from src.data.corruptions import CLASSES

from .model import BRANCHES

BRANCH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # identity, salt, blur, occlusion
SYSTEM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
HATCHES = ("", "//", "..", "xx", "\\\\", "--")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    plt = _plt()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_to_array(fig):
    """Render a figure to a uint8 RGB array (for W&B) and close it."""
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    _plt().close(fig)
    return array


def heatmap_figure(matrix, row_labels, col_labels=BRANCHES, title="", xlabel="branch", ylabel="true input"):
    """Annotated heatmap of mean routing weights (values in [0, 1])."""
    plt = _plt()
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 1.6, 0.42 * len(row_labels) + 1.2))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            ax.text(c, r, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean weight")
    return fig


def save_heatmap(matrix, row_labels, path, **kwargs):
    _save(heatmap_figure(matrix, row_labels, **kwargs), path)


def routing_heatmap_array(matrix, title=""):
    """The 4x4 routing matrix (true class x branch) as an image array, logged to W&B every epoch."""
    return figure_to_array(heatmap_figure(matrix, CLASSES, title=title))


def save_weight_distributions(weights, labels, path):
    """Per true class: the distribution of each branch weight over the test entries (violin + median)."""
    plt = _plt()
    weights, labels = np.asarray(weights), np.asarray(labels)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(3.0 * len(CLASSES), 2.8), sharey=True)
    for c, ax in enumerate(axes):
        w = weights[labels == c]
        if len(w):
            parts = ax.violinplot([w[:, k] for k in range(len(BRANCHES))], showmedians=True, widths=0.8)
            for k, body in enumerate(parts["bodies"]):
                body.set_facecolor(BRANCH_COLORS[k])
                body.set_edgecolor(BRANCH_COLORS[k])
                body.set_alpha(0.75)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                parts[key].set_color("#52514e")
                parts[key].set_linewidth(1)
        ax.set_xticks(range(1, len(BRANCHES) + 1), BRANCHES, rotation=30, fontsize=8)
        ax.set_title(f"true: {CLASSES[c]} (n={len(w)})", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("routing weight", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def save_entropy_histogram(entropy, labels, path, bins=20):
    """Normalised routing entropy per true class (0 = one branch, 1 = uniform), small multiples."""
    plt = _plt()
    entropy, labels = np.asarray(entropy), np.asarray(labels)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(3.0 * len(CLASSES), 2.4), sharey=True)
    edges = np.linspace(0, 1, bins + 1)
    for c, ax in enumerate(axes):
        e = entropy[labels == c]
        ax.hist(e, bins=edges, color=BRANCH_COLORS[c], edgecolor="white", linewidth=0.8)
        ax.set_title(f"true: {CLASSES[c]} (median {np.median(e) if len(e) else float('nan'):.3f})", fontsize=9)
        ax.set_xlabel("routing entropy / log 4", fontsize=8)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("entries", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def save_branch_figure(samples, path, title=""):
    """Rows: input | identity | salt | blur | occlusion branch outputs (weight in the title) | output | target.

    samples: dicts with HWC arrays input, branches (4 arrays), output, target, the weights (4,)
    and a label.
    """
    plt = _plt()
    cols = 7
    fig, axes = plt.subplots(len(samples), cols, figsize=(cols * 1.55, len(samples) * 1.75), squeeze=False)
    for r, s in enumerate(samples):
        panels = [s["input"], *s["branches"], s["output"], s["target"]]
        titles = ["input", *[f"{b}\nw={w:.2f}" for b, w in zip(BRANCHES, s["weights"])], "output", "target"]
        for c, (img, text) in enumerate(zip(panels, titles)):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(text, fontsize=7)
            if 1 <= c <= 4:  # frame thickness shows the branch weight
                k = c - 1
                for spine in ax.spines.values():
                    spine.set_edgecolor(BRANCH_COLORS[k])
                    spine.set_linewidth(0.5 + 4 * float(s["weights"][k]))
        axes[r, 0].set_ylabel(s.get("label", ""), fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def save_image_rows(rows, col_titles, row_labels, path, title=""):
    """A grid of HWC images in [0, 1]: one row per example, fixed column titles on the first row."""
    plt = _plt()
    n_cols = len(col_titles)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 1.6, len(rows) * 1.7), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=6)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def save_grouped_bars(groups, systems, values, path, ylabel, title=""):
    """Grouped bars: one group per x label, one bar per system (fixed colour + hatch per system).

    values[s][g] is the value of system s in group g (NaN = missing, bar omitted).
    """
    plt = _plt()
    n_sys = len(systems)
    width = 0.8 / n_sys
    fig, ax = plt.subplots(figsize=(max(6.0, 0.75 * len(groups) * max(1, n_sys) / 2 + 2), 3.0))
    x = np.arange(len(groups))
    for s, name in enumerate(systems):
        v = np.asarray(values[s], dtype=float)
        ax.bar(x + (s - (n_sys - 1) / 2) * width, np.where(np.isnan(v), 0, v), width * 0.92,
               color=SYSTEM_COLORS[s % len(SYSTEM_COLORS)], hatch=HATCHES[s % len(HATCHES)],
               edgecolor="white", linewidth=0.5, label=name)
    ax.set_xticks(x, groups, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    finite = [v for row in values for v in row if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        ax.set_ylim(max(0, lo - 0.1 * (hi - lo + 1e-9) - (0.02 if hi <= 1 else 1)), hi + 0.08 * (hi - lo + 1e-9))
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=min(n_sys, 4), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    if title:
        ax.set_title(title, fontsize=9, pad=24)
    fig.tight_layout()
    _save(fig, path)
