"""Evaluate hard-routed restoration on the full test manifest in oracle and predicted routing modes.

Both modes go through src.common.evaluation.evaluate_restoration with extras (true label,
classifier prediction and probabilities, selected expert), so the per-entry CSVs have the
same columns as Tasks 1 and 3:
  OUTPUT_DIR/task2/eval/test_records_{oracle,predicted}.csv

Clean inputs routed to the identity bypass are reproduced exactly (MSE 0, SSIM 1, PSNR
capped at 100 dB). Averaging those capped values with finite PSNRs would be meaningless,
so the tables add, per row, `n_exact` (outputs identical to the target) and `psnr_finite`
(mean PSNR over the other entries); the standard `psnr` column is kept for comparability.

Tables (OUTPUT_DIR/task2/tables/):
  routing_{mode}_{by_type,by_type_level,by_level}.csv/.tex   standard tables + exact-output columns
  routing_oracle_vs_predicted.csv/.tex   per type x severity: routing accuracy, PSNR/SSIM in both modes
  routing_misrouting.csv/.tex            every (true, predicted) confusion: how many entries, how much
                                         worse (or better) predicted routing restores them than oracle
Figures (OUTPUT_DIR/task2/figures/):
  routing_failures.png      worst restoration failures caused by classifier errors:
                            target | input | oracle output | predicted output | probabilities
  routing_examples_{mode}.png  representative entries (median SSIM per type x severity)
Summary: OUTPUT_DIR/task2/eval/summary.json (read by export_onnx for the model cards)

A misrouted entry is "harmful" when predicted routing is at least --harm-db dB worse in
PSNR than oracle routing, "beneficial" when at least --harm-db dB better, else "neutral".

    python -m src.task2.evaluate_routing --smoke
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.common.evaluation import (
    evaluate_restoration, predict_entries, representative_entries, save_csv, save_latex_table,
    save_restoration_figure, standard_tables,
)
from src.common.metrics import PSNR_CAP_DB
from src.common.paths import get_dir
from src.data.corruptions import CLASS_TO_IDX, CLASSES, LEVELS
from src.data.pets import ManifestDataset, load_pets

from .common import SPECIALIST_TYPES, TASK, disable_wandb_for_smoke, save_figure, setup_device
from .evaluate_classifier import CLASS_COLORS
from .routing import EXPERT_NAMES, ROUTING_MODES, HardRoutedRestorer, load_models

TYPE_LEVELS = [("clean", "none")] + [(t, lv) for t in SPECIALIST_TYPES for lv in LEVELS]


def make_predict_fn(restorer, dataset):
    """predict_fn for evaluate_restoration, which passes only the input batch.

    evaluate_restoration iterates the dataset in order (shuffle=False), so the true labels
    of each batch are the next len(x) manifest labels. The caller checks the alignment
    afterwards against the labels evaluate_restoration reads from the dataset itself.
    """
    labels = torch.tensor([CLASS_TO_IDX[e["type"]] for e in dataset.entries])
    state = {"offset": 0}

    def predict(x):
        y = labels[state["offset"]:state["offset"] + len(x)]
        state["offset"] += len(x)
        output, info = restorer(x, labels=y.to(x.device))
        predicted, route, probs = info["predicted"].cpu(), info["route"].cpu(), info["probs"].cpu()
        extras = {
            "true_label": y,
            "pred_label": predicted,
            "pred_class": [CLASSES[i] for i in predicted.tolist()],
            "classifier_correct": (predicted == y).int(),
            "route": route,
            "expert": [EXPERT_NAMES[i] for i in route.tolist()],
            **{f"prob_{c}": probs[:, k] for k, c in enumerate(CLASSES)},
        }
        return output, extras

    return predict, state


def run_mode(classifier, specialists, dataset, mode, device, batch_size, num_workers):
    restorer = HardRoutedRestorer(classifier, specialists, mode).to(device).eval()
    predict, state = make_predict_fn(restorer, dataset)
    records = evaluate_restoration(predict, dataset, device, batch_size=batch_size, num_workers=num_workers)
    misaligned = [r["entry"] for r in records if r["true_label"] != CLASS_TO_IDX[r["type"]]]
    if state["offset"] != len(dataset) or misaligned:
        raise RuntimeError(f"oracle labels out of step with the dataset ({len(misaligned)} entries)")
    if mode == "oracle" and any(r["route"] != r["true_label"] for r in records):
        raise RuntimeError("oracle routing did not follow the true labels")
    return records


def exact_columns(df, rows, keys):
    """Add n_exact and psnr_finite (mean over non-exact outputs) to standard_tables rows."""
    out = []
    for row in rows:
        mask = np.ones(len(df), dtype=bool)
        for k in keys:
            if row[k] == "all":
                continue
            mask &= (df[k] == row[k]).to_numpy()
        sub = df[mask]
        exact = sub["psnr"] >= PSNR_CAP_DB
        out.append({**row, "n_exact": int(exact.sum()),
                    "psnr_finite": float(sub.loc[~exact, "psnr"].mean()) if (~exact).any() else float("nan")})
    return out


def latex_rows(rows):
    return [{k: ("--" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows]


def save_standard_tables(records, mode, tables_dir):
    df = pd.DataFrame(records)
    tables = standard_tables(records)
    keys = {"by_type": ("type",), "by_type_level": ("type", "level"), "by_level": ("level",)}
    corrupted = df[df["type"] != "clean"]
    out = {}
    for name, rows in tables.items():
        rows = exact_columns(corrupted if name == "by_level" else df, rows, keys[name])
        if name == "by_type":  # mean over the three corruptions, free of the clean entries' capped PSNR
            c = corrupted
            rows.append({"type": "corrupted", "n": len(c), "psnr": float(c["psnr"].mean()), "ssim": float(c["ssim"].mean()),
                         "mse": float(c["mse"].mean()), "n_exact": int((c["psnr"] >= PSNR_CAP_DB).sum()),
                         "psnr_finite": float(c.loc[c["psnr"] < PSNR_CAP_DB, "psnr"].mean())})
        save_csv(rows, tables_dir / f"routing_{mode}_{name}.csv")
        first = [(k, k.capitalize()) for k in keys[name]]
        save_latex_table(latex_rows(rows), first + [("n", "n"), ("psnr_finite", "PSNR*"), ("ssim", "SSIM"),
                                                    ("n_exact", "Exact")],
                         tables_dir / f"routing_{mode}_{name}.tex",
                         caption=f"Hard routing ({mode}), test set. PSNR*: mean over outputs that are not exact copies "
                                 "of the target; Exact: identity-bypass outputs equal to the clean target "
                                 "(PSNR capped at 100 dB, SSIM 1).",
                         label=f"tab:t2_routing_{mode}_{name}",
                         fmt={"psnr_finite": "{:.2f}"})
        out[name] = rows
    return out


def merge_modes(oracle, predicted, harm_db):
    o, p = pd.DataFrame(oracle), pd.DataFrame(predicted)
    m = o.merge(p, on="entry", suffixes=("_oracle", "_pred"))
    m = m.rename(columns={"type_oracle": "type", "level_oracle": "level", "image_idx_oracle": "image_idx",
                          "true_label_oracle": "true_label", "pred_class_oracle": "pred_class"})
    m["misrouted"] = m["pred_label_pred"] != m["true_label"]
    m["delta_psnr"] = m["psnr_pred"] - m["psnr_oracle"]  # capped values: a misrouted clean input is "harmful"
    m["delta_ssim"] = m["ssim_pred"] - m["ssim_oracle"]
    m["outcome"] = np.where(~m["misrouted"], "correct",
                            np.where(m["delta_psnr"] <= -harm_db, "harmful",
                                     np.where(m["delta_psnr"] >= harm_db, "beneficial", "neutral")))
    return m


def finite_mean(series):
    finite = series[series < PSNR_CAP_DB]
    return float(finite.mean()) if len(finite) else float("nan")


def comparison_table(m):
    """Oracle vs predicted routing per type x severity, per type, and over all corrupted entries."""
    groups = [((t, lv), m[(m["type"] == t) & (m["level"] == lv)]) for t, lv in TYPE_LEVELS]
    groups += [((t, "all"), m[m["type"] == t]) for t in SPECIALIST_TYPES]
    groups += [(("corrupted", "all"), m[m["type"] != "clean"])]
    rows = []
    for (t, lv), g in groups:
        if g.empty:
            continue
        clean = t == "clean"
        rows.append({
            "type": t, "level": lv, "n": len(g), "routing_accuracy": float((~g["misrouted"]).mean()),
            "psnr_oracle": finite_mean(g["psnr_oracle"]), "psnr_predicted": finite_mean(g["psnr_pred"]),
            "delta_psnr": float("nan") if clean else float(g["delta_psnr"].mean()),
            "ssim_oracle": float(g["ssim_oracle"].mean()), "ssim_predicted": float(g["ssim_pred"].mean()),
            "delta_ssim": float(g["delta_ssim"].mean()),
            "exact_oracle": int((g["psnr_oracle"] >= PSNR_CAP_DB).sum()),
            "exact_predicted": int((g["psnr_pred"] >= PSNR_CAP_DB).sum()),
            "n_misrouted": int(g["misrouted"].sum()), "n_harmful": int((g["outcome"] == "harmful").sum()),
            "n_beneficial": int((g["outcome"] == "beneficial").sum()),
        })
    return rows


def misrouting_table(m):
    """One row per (true, predicted) confusion: counts by severity, effect on restoration, outcome counts."""
    rows = []
    wrong = m[m["misrouted"]]
    for (t, pred), g in wrong.groupby(["type", "pred_class_pred"]):
        n_type = int((m["type"] == t).sum())
        rows.append({
            "true": t, "predicted": pred, "expert_used": EXPERT_NAMES[CLASS_TO_IDX[pred]], "n": len(g),
            "share_of_true_type": len(g) / n_type,
            **{f"n_{lv}": int((g["level"] == lv).sum()) for lv in LEVELS},
            "psnr_oracle": finite_mean(g["psnr_oracle"]), "psnr_predicted": finite_mean(g["psnr_pred"]),
            "delta_psnr": float("nan") if t == "clean" else float(g["delta_psnr"].mean()),
            "delta_ssim": float(g["delta_ssim"].mean()),
            "n_harmful": int((g["outcome"] == "harmful").sum()), "n_neutral": int((g["outcome"] == "neutral").sum()),
            "n_beneficial": int((g["outcome"] == "beneficial").sum()),
        })
    order = {c: i for i, c in enumerate(CLASSES)}
    return sorted(rows, key=lambda r: (order[r["true"]], order[r["predicted"]]))


def pick_failures(m, n=8):
    """Worst harmful entries, round-robin over confusion pairs so each kind of failure appears."""
    harmful = m[m["outcome"] == "harmful"]
    queues = [g.sort_values(["delta_ssim", "delta_psnr"])["entry"].tolist()
              for _, g in harmful.groupby(["type", "pred_class_pred"])]
    queues.sort(key=len, reverse=True)
    chosen = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop(0))
    return chosen


@torch.no_grad()
def failure_figure(m, entries, dataset, classifier, specialists, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [dataset[e] for e in entries]
    x = torch.stack([it["input"] for it in items]).to(device)
    labels = torch.tensor([it["label"] for it in items], device=device)
    outputs = {}
    for mode in ROUTING_MODES:
        restorer = HardRoutedRestorer(classifier, specialists, mode).to(device).eval()
        outputs[mode] = restorer(x, labels=labels)[0].clamp(0, 1).cpu()
    rows = m.set_index("entry")
    fig, axes = plt.subplots(len(entries), 5, figsize=(11.0, 2.25 * len(entries)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.1]})
    hwc = lambda t: t.permute(1, 2, 0).numpy().clip(0, 1)  # noqa: E731
    for r, (entry, it) in enumerate(zip(entries, items)):
        row = rows.loc[entry]
        level = "" if row["level"] == "none" else f" {row['level']}"
        panels = [("Target", hwc(it["target"])), (f"Input: {row['type']}{level}", hwc(it["input"])),
                  (f"Oracle ({row['expert_oracle']})\nPSNR {row['psnr_oracle']:.1f} SSIM {row['ssim_oracle']:.3f}",
                   hwc(outputs["oracle"][r])),
                  (f"Predicted ({row['expert_pred']})\nPSNR {row['psnr_pred']:.1f} SSIM {row['ssim_pred']:.3f}",
                   hwc(outputs["predicted"][r]))]
        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title.replace(f"PSNR {PSNR_CAP_DB:.1f}", "exact"), fontsize=7)
        ax = axes[r, 4]
        probs = [row[f"prob_{c}_pred"] for c in CLASSES]
        ax.barh(range(len(CLASSES)), probs, color=[CLASS_COLORS[c] for c in CLASSES], height=0.7)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        for i, p in enumerate(probs):
            ax.text(min(p + 0.02, 0.78), i, f"{p:.2f}", va="center", fontsize=6)
        ax.set_title("Classifier probabilities", fontsize=7)
    fig.suptitle("Restoration failures caused by classifier errors (predicted vs oracle routing)", fontsize=9)
    fig.tight_layout()
    return fig


def mode_summary(records):
    df = pd.DataFrame(records)
    out = {}
    for t in (*CLASSES, "corrupted"):
        g = df[df["type"] != "clean"] if t == "corrupted" else df[df["type"] == t]
        if g.empty:
            continue
        out[t] = {"n": len(g), "psnr_finite": finite_mean(g["psnr"]), "ssim": float(g["ssim"].mean()),
                  "mse": float(g["mse"].mean()), "n_exact": int((g["psnr"] >= PSNR_CAP_DB).sum()),
                  "share_identity": float((g["route"] == 0).mean()),
                  "by_level": {lv: {"psnr": float(s["psnr"].mean()), "ssim": float(s["ssim"].mean()), "n": len(s)}
                               for lv, s in g.groupby("level") if lv != "none"}}
    return out


def evaluate(smoke=False, batch_size=128, num_workers=None, harm_db=1.0, n_failures=8):
    disable_wandb_for_smoke(smoke)
    device = setup_device()
    workers = num_workers if num_workers is not None else (2 if device.type == "cuda" else 0)
    classifier, specialists = load_models(device, smoke=smoke)
    images, manifests = load_pets(smoke=smoke)
    dataset = ManifestDataset(images["test"], manifests["test"])
    eval_dir, tables_dir, fig_dir = (get_dir("OUTPUT_DIR", TASK, sub) for sub in ("eval", "tables", "figures"))

    records = {}
    for mode in ROUTING_MODES:
        records[mode] = run_mode(classifier, specialists, dataset, mode, device, batch_size, workers)
        save_csv(records[mode], eval_dir / f"test_records_{mode}.csv")
        save_standard_tables(records[mode], mode, tables_dir)
        restorer = HardRoutedRestorer(classifier, specialists, mode).to(device).eval()
        chosen = representative_entries(records[mode], per_group=1)
        chosen_labels = torch.tensor([dataset[i]["label"] for i in chosen], device=device)
        samples = predict_entries(lambda x: restorer(x, labels=chosen_labels)[0], dataset, chosen, device)
        save_restoration_figure(samples, fig_dir / f"routing_examples_{mode}.png")
        print(f"[routing] {mode}: {len(records[mode])} entries evaluated")

    m = merge_modes(records["oracle"], records["predicted"], harm_db)
    comparison = comparison_table(m)
    save_csv(comparison, tables_dir / "routing_oracle_vs_predicted.csv")
    save_latex_table(latex_rows(comparison),
                     [("type", "Type"), ("level", "Level"), ("n", "n"), ("routing_accuracy", "Route acc."),
                      ("psnr_oracle", "PSNR or."), ("psnr_predicted", "PSNR pr."), ("ssim_oracle", "SSIM or."),
                      ("ssim_predicted", "SSIM pr."), ("n_misrouted", "Misr."), ("n_harmful", "Harmful")],
                     tables_dir / "routing_oracle_vs_predicted.tex",
                     caption="Oracle vs predicted hard routing on the test set. PSNR excludes outputs identical to "
                             "the target (clean inputs passed through the identity bypass).",
                     label="tab:t2_oracle_vs_predicted",
                     fmt={"routing_accuracy": "{:.4f}", "psnr_oracle": "{:.2f}", "psnr_predicted": "{:.2f}"})
    misrouting = misrouting_table(m)
    columns = ["true", "predicted", "expert_used", "n", "share_of_true_type", *[f"n_{lv}" for lv in LEVELS],
               "psnr_oracle", "psnr_predicted", "delta_psnr", "delta_ssim", "n_harmful", "n_neutral", "n_beneficial"]
    pd.DataFrame(misrouting, columns=columns).to_csv(tables_dir / "routing_misrouting.csv", index=False)
    if misrouting:
        save_latex_table(latex_rows(misrouting),
                         [("true", "True"), ("predicted", "Routed to"), ("n", "n"), ("share_of_true_type", "Share"),
                          ("delta_psnr", "$\\Delta$PSNR"), ("delta_ssim", "$\\Delta$SSIM"), ("n_harmful", "Harmful"),
                          ("n_neutral", "Neutral"), ("n_beneficial", "Benef.")],
                         tables_dir / "routing_misrouting.tex",
                         caption=f"Test entries misrouted by the classifier. $\\Delta$ = predicted minus oracle "
                                 f"routing; harmful/beneficial: at least {harm_db:g} dB worse/better.",
                         label="tab:t2_misrouting",
                         fmt={"share_of_true_type": "{:.4f}", "delta_psnr": "{:.2f}", "delta_ssim": "{:.4f}"})
    failures = pick_failures(m, n_failures)
    if failures:
        save_figure(failure_figure(m, failures, dataset, classifier, specialists, device),
                    fig_dir / "routing_failures.png")

    outcomes = m["outcome"].value_counts().to_dict()
    summary = {
        "n_entries": len(m),
        "routing_accuracy": float((~m["misrouted"]).mean()),
        "misrouted": int(m["misrouted"].sum()),
        "outcomes": {k: int(outcomes.get(k, 0)) for k in ("correct", "harmful", "neutral", "beneficial")},
        "harm_threshold_db": harm_db,
        "oracle": mode_summary(records["oracle"]),
        "predicted": mode_summary(records["predicted"]),
        "note": "psnr_finite excludes outputs identical to the target (PSNR capped at 100 dB); "
                "n_exact counts them. Oracle per-type metrics are each specialist's results on its own corruption.",
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[routing] routing accuracy {summary['routing_accuracy']:.4f}; misrouted {summary['misrouted']} "
          f"({summary['outcomes']})")
    for mode in ROUTING_MODES:
        c = summary[mode].get("corrupted", {})
        print(f"[routing] {mode}: corrupted inputs PSNR {c.get('psnr_finite', float('nan')):.2f} dB "
              f"SSIM {c.get('ssim', float('nan')):.4f}")
    print(f"[routing] outputs in {eval_dir}, {tables_dir}, {fig_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--harm-db", type=float, default=1.0, help="PSNR margin that makes a misrouting harmful")
    parser.add_argument("--n-failures", type=int, default=8, help="rows in the failure figure")
    args = parser.parse_args(argv)
    return evaluate(args.smoke, args.batch_size, args.num_workers, args.harm_db, args.n_failures)


if __name__ == "__main__":
    main()
