"""Cross-task comparison table with Task 1 included (Task 3's compare.py ran on Kaggle without the
Task 1 records). Reads the per-entry test records of all four systems, checks that they are paired
(same entries in the same order) and writes report/tables/crosstask_by_type.tex and
report/tables/crosstask_paired.tex.

    python report/crosstask_table.py      # from the repository root
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
CAP = 100.0 - 1e-6
SYSTEMS = [("t1", "Task 1 universal", OUT / "task1/variants/udae_skip32/eval/test_records.csv"),
           ("or", "Task 2 oracle", OUT / "task2/eval/test_records_oracle.csv"),
           ("pr", "Task 2 predicted", OUT / "task2/eval/test_records_predicted.csv"),
           ("t3", "Task 3 soft MoE", OUT / "task3/eval/test_records.csv")]
TYPES = ["clean", "salt", "blur", "occlusion"]


def load():
    frames = {k: pd.read_csv(p, encoding="utf-8") for k, _, p in SYSTEMS}
    ref = frames["t1"]
    for k, df in frames.items():
        assert len(df) == len(ref) and (df["entry"].values == ref["entry"].values).all(), k
        assert (df["type"].values == ref["type"].values).all() and (df["level"].values == ref["level"].values).all(), k
    return frames


def cell(df, mask):
    sub = df[mask]
    finite = sub[sub["psnr"] < CAP]
    psnr = f"{finite['psnr'].mean():.2f}" if len(finite) else "--"
    exact = int((sub["psnr"] >= CAP).sum())
    return psnr, f"{sub['ssim'].mean():.4f}", exact


def table_by_type(frames):
    t = frames["t1"]["type"]
    rows = []
    groups = [(ty, t == ty) for ty in TYPES] + [("corrupted", t != "clean")]
    for name, mask in groups:
        cells = []
        for k, _, _ in SYSTEMS:
            psnr, ssim, exact = cell(frames[k], mask)
            if name == "clean" and exact:
                psnr = f"{psnr} ({exact} exact)" if psnr != "--" else f"exact ({exact})"
            cells += [psnr, ssim]
        rows.append(" & ".join([name] + cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{Task 1 universal} & \multicolumn{2}{c}{Task 2 oracle} & "
            r"\multicolumn{2}{c}{Task 2 predicted} & \multicolumn{2}{c}{Task 3 soft MoE} \\" "\n"
            r"Input & PSNR* & SSIM & PSNR* & SSIM & PSNR* & SSIM & PSNR* & SSIM \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def table_paired(frames):
    t = frames["t1"]["type"]
    pairs = [("t3", "t1"), ("pr", "t1"), ("t3", "pr")]
    names = {k: n for k, n, _ in SYSTEMS}
    groups = [("all corrupted", t != "clean")] + [(ty, t == ty) for ty in TYPES[1:]]
    lines = []
    for a, b in pairs:
        for gname, mask in groups:
            pa, pb = frames[a]["psnr"], frames[b]["psnr"]
            m = mask & (pa < CAP) & (pb < CAP)
            gain = (pa[m] - pb[m])
            lines.append(f"{names[a]} vs {names[b]} & {gname} & {int(m.sum())} & {gain.mean():+.2f} & "
                         f"{100 * (gain > 0).mean():.1f}\\% \\\\")
    head = (r"\begin{tabular}{llrrr}" "\n" r"\toprule" "\n"
            r"Comparison & Inputs & $n$ & Mean $\Delta$PSNR & Wins \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(lines) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def main():
    frames = load()
    tables = REPO / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "crosstask_by_type.tex").write_text(table_by_type(frames), encoding="utf-8")
    (tables / "crosstask_paired.tex").write_text(table_paired(frames), encoding="utf-8")
    print((tables / "crosstask_by_type.tex").read_text(encoding="utf-8"))
    print((tables / "crosstask_paired.tex").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
