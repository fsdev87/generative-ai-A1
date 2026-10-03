"""Cross-task comparison on the test set: universal AE vs hard routing vs soft MoE.

Reads the per-entry CSVs (docs/CONVENTIONS.md), skipping any that do not exist yet:
    Task 1  OUTPUT_DIR/task1/eval/test_records.csv                 universal AE
    Task 2  OUTPUT_DIR/task2/eval/test_records_oracle.csv          hard routing, true label
            OUTPUT_DIR/task2/eval/test_records_predicted.csv       hard routing, classifier
    Task 3  OUTPUT_DIR/task3/eval/test_records.csv                 soft MoE
and writes OUTPUT_DIR/task3/tables/comparison_by_type_level.{csv,tex}, comparison_by_type.{csv,tex},
comparison_paired.csv and figures/comparison_{psnr,ssim}.png.

Hard routing sends a clean input through the identity bypass, so its output equals the
target and PSNR sits at the 100 dB cap. PSNR columns therefore exclude such exact entries
(PSNR*, as in the Task 2 tables) and n_exact columns count them; the clean row is best
compared by SSIM and by how many entries are left untouched.
All CSVs come from the same test manifest, one row per entry in manifest order, so entries
are paired: the paired table gives the mean PSNR gain of the soft MoE over each system and
the share of entries where it wins (when Task 2 records which entries the classifier got
wrong, also separately for those entries).

    python -m src.task3.compare [--smoke]
"""
import argparse

import numpy as np
import pandas as pd

from src.common.evaluation import save_csv, save_latex_table
from src.common.metrics import PSNR_CAP_DB
from src.common.paths import get_dir
from src.data.corruptions import CLASSES, LEVELS

from .config import output_dir
from .plots import save_grouped_bars

SYSTEMS = (  # key, label, task folder, file
    ("universal", "Universal AE (T1)", "task1", "test_records.csv"),
    ("oracle", "Hard routing, oracle (T2)", "task2", "test_records_oracle.csv"),
    ("predicted", "Hard routing, predicted (T2)", "task2", "test_records_predicted.csv"),
    ("moe", "Soft MoE (T3)", "task3", "test_records.csv"),
)
KEY_COLS = ["entry", "image_idx", "type", "level"]


def load_systems(smoke=False):
    """{key: DataFrame} for every per-entry CSV that exists (Task 3's own from the smoke folder in smoke mode)."""
    frames = {}
    for key, label, task, name in SYSTEMS:
        folder = output_dir(smoke, "eval") if task == "task3" else get_dir("OUTPUT_DIR", task, "eval")
        path = folder / name
        if path.exists():
            df = pd.read_csv(path)
            df["level"] = df["level"].fillna("none")
            frames[key] = df
            print(f"[compare] {label}: {path} ({len(df)} entries)")
        else:
            print(f"[compare] {label}: {path} missing, skipped")
    return frames


def _group_stats(df):
    exact = df["psnr"] >= PSNR_CAP_DB
    return {"psnr": float(df.loc[~exact, "psnr"].mean()) if (~exact).any() else float("nan"),
            "ssim": float(df["ssim"].mean()), "n_exact": int(exact.sum()), "n": len(df)}


def comparison_rows(frames, by_level=True):
    groups = [("clean", "none")] + [(t, lv) for t in CLASSES[1:] for lv in LEVELS] if by_level else \
        [(t, None) for t in CLASSES] + [("corrupted", None)]
    rows = []
    for t, lv in groups:
        row = {"type": t, **({"level": lv} if by_level else {})}
        for key, df in frames.items():
            if t == "corrupted":
                sub = df[df["type"] != "clean"]
            else:
                sub = df[(df["type"] == t) & ((df["level"] == lv) if by_level else True)]
            if len(sub):
                s = _group_stats(sub)
                row.update({f"{key}_psnr": s["psnr"], f"{key}_ssim": s["ssim"], f"{key}_n_exact": s["n_exact"]})
                row["n"] = s["n"]
        rows.append(row)
    return rows


def paired_rows(frames):
    """Soft MoE minus each other system, entry by entry (only when the CSVs list the same entries)."""
    if "moe" not in frames:
        return []
    moe = frames["moe"]
    rows = []
    for key, df in frames.items():
        if key == "moe":
            continue
        if len(df) != len(moe) or not (df[KEY_COLS].astype(str).values == moe[KEY_COLS].astype(str).values).all():
            print(f"[compare] {key}: entries do not match the soft MoE records, no paired comparison")
            continue
        both_finite = (df["psnr"] < PSNR_CAP_DB) & (moe["psnr"] < PSNR_CAP_DB)
        subsets = [("all corrupted", moe["type"] != "clean")] + [(t, moe["type"] == t) for t in CLASSES[1:]]
        if "classifier_correct" in df.columns:  # Task 2 predicted mode records classifier errors
            wrong = df["classifier_correct"] == 0
            subsets += [("classifier wrong", wrong & (moe["type"] != "clean")),
                        ("classifier right", ~wrong & (moe["type"] != "clean"))]
        for name, mask in subsets:
            m = mask & both_finite
            if not m.any():
                continue
            d_psnr = moe.loc[m, "psnr"] - df.loc[m, "psnr"]
            d_ssim = moe.loc[mask, "ssim"] - df.loc[mask, "ssim"]
            rows.append({"versus": key, "subset": name, "n": int(m.sum()),
                         "mean_psnr_gain": float(d_psnr.mean()), "mean_ssim_gain": float(d_ssim.mean()),
                         "moe_wins_psnr": float((d_psnr > 0).mean())})
    return rows


def write_outputs(frames, smoke=False):
    tables, figs = output_dir(smoke, "tables"), output_dir(smoke, "figures")
    labels = {k: label for k, label, _, _ in SYSTEMS}
    present = [k for k, *_ in SYSTEMS if k in frames]
    by_tl, by_t = comparison_rows(frames, True), comparison_rows(frames, False)
    for name, rows, first in (("comparison_by_type_level", by_tl, [("type", "Input"), ("level", "Level")]),
                              ("comparison_by_type", by_t, [("type", "Input")])):
        save_csv(rows, tables / f"{name}.csv")
        cols = first + [c for k in present for c in ((f"{k}_psnr", f"{labels[k]} PSNR*"), (f"{k}_ssim", "SSIM"))]
        latex = [{k: ("--" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows]
        latex = [{**{c: "--" for c, _ in cols}, **r} for r in latex]
        save_latex_table(latex, cols, tables / f"{name}.tex",
                         caption="Test-set comparison. PSNR* excludes outputs identical to the target (identity "
                                 "bypass, PSNR capped at 100 dB); the clean row of hard routing is then often empty.",
                         label=f"tab:{name}",
                         fmt={c: ("{:.4f}" if c.endswith("ssim") else "{:.2f}") for c, _ in cols[len(first):]})
    paired = paired_rows(frames)
    if paired:
        save_csv(paired, tables / "comparison_paired.csv")
    groups = [f"{r['type']} {r['level']}" if r["level"] != "none" else r["type"] for r in by_tl]
    for metric, ylabel in (("psnr", "PSNR* (dB)"), ("ssim", "SSIM")):
        values = [[r.get(f"{k}_{metric}", float("nan")) for r in by_tl] for k in present]
        save_grouped_bars(groups, [labels[k] for k in present], values, figs / f"comparison_{metric}.png", ylabel)
    return {"by_type_level": by_tl, "by_type": by_t, "paired": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="read Task 3's records from the smoke folder")
    args = parser.parse_args(argv)
    frames = load_systems(args.smoke)
    if not frames:
        print("[compare] no per-entry CSVs found: run the evaluations of Tasks 1-3 first")
        return None
    result = write_outputs(frames, args.smoke)
    for row in result["by_type"]:
        print("  " + row["type"].ljust(10) + "  ".join(
            f"{k}: {row.get(f'{k}_psnr', float('nan')):6.2f} dB / {row.get(f'{k}_ssim', float('nan')):.4f}"
            for k in frames))
    print(f"[compare] wrote {output_dir(args.smoke, 'tables')}")
    return result


if __name__ == "__main__":
    main()
