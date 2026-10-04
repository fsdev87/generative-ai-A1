"""Copy the figures and LaTeX tables written by the training/evaluation scripts into report/.

main.tex includes every result through \\resultfig{<file>} (report/figures/) and
\\resulttable{<file>} (report/tables/): a file that exists is used, a missing one is shown as a
red placeholder naming it. This script fills those folders from the outputs/ tree using one
naming rule, so no LaTeX has to be edited when new results arrive:

    outputs/corruption_grid.png                  -> figures/corruption_grid.png
    outputs/taskN/figures/X.png                  -> figures/taskN_X.png
    outputs/taskN/tables/X.tex                   -> tables/taskN_X.tex
    outputs/task3/mixed/X.png | X.tex            -> figures/task3_X.png | tables/task3_X.tex
    outputs/task1/optuna/task1_udae/X.png        -> figures/task1_optuna_X.png
    outputs/task2/optuna/task2_classifier/X.png  -> figures/task2_classifier_optuna_X.png
    outputs/task2/optuna/task2_specialists/X.png -> figures/task2_specialists_optuna_X.png
    outputs/task3/optuna/task3_moe/X.png         -> figures/task3_optuna_X.png
    outputs/task4/optuna/task4_cgan/X.png        -> figures/task4_optuna_X.png

Usage (from the repository root):

    python report/collect_outputs.py                  # every group
    python report/collect_outputs.py task1 task2-optuna
    python report/collect_outputs.py --outputs D:/kaggle/outputs task2 task3 task4
    python report/collect_outputs.py --missing        # list files main.tex still waits for

Figures that no script writes (Weights & Biases curve exports, application screenshots) are
listed by --missing and must be saved under the name given there.
"""
import argparse
import re
import shutil
from pathlib import Path

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parent

# group -> list of (source folder relative to outputs/, glob, destination folder, name prefix)
GROUPS = {
    "data": [(".", "corruption_grid.png", "figures", "")],
    "task1": [("task1/figures", "*.png", "figures", "task1_"),
              ("task1/tables", "*.tex", "tables", "task1_")],
    "task1-optuna": [("task1/optuna/task1_udae", "*.png", "figures", "task1_optuna_")],
    "task2-optuna": [("task2/optuna/task2_classifier", "*.png", "figures", "task2_classifier_optuna_"),
                     ("task2/optuna/task2_specialists", "*.png", "figures", "task2_specialists_optuna_")],
    "task2": [("task2/figures", "*.png", "figures", "task2_"),
              ("task2/tables", "*.tex", "tables", "task2_")],
    "task3": [("task3/figures", "*.png", "figures", "task3_"),
              ("task3/tables", "*.tex", "tables", "task3_"),
              ("task3/mixed", "*.png", "figures", "task3_"),
              ("task3/mixed", "*.tex", "tables", "task3_"),
              ("task3/optuna/task3_moe", "*.png", "figures", "task3_optuna_")],
    "task4": [("task4/figures", "*.png", "figures", "task4_"),
              ("task4/tables", "*.tex", "tables", "task4_"),
              ("task4/optuna/task4_cgan", "*.png", "figures", "task4_optuna_")],
}


def copy_group(outputs, group):
    copied = 0
    for folder, pattern, dest, prefix in GROUPS[group]:
        for src in sorted((outputs / folder).glob(pattern)):
            if "smoke" in src.relative_to(outputs).parts:
                continue
            target = REPORT / dest / f"{prefix}{src.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
            print(f"  {src.relative_to(outputs)} -> report/{dest}/{target.name}")
    return copied


def referenced_files():
    """Files main.tex includes through \\resultfig / \\resulttable."""
    tex = (REPORT / "main.tex").read_text(encoding="utf-8")
    figures = re.findall(r"\\resultfig(?:\[[^\]]*\])?\{([^}\s]+)\}", tex)
    tables = re.findall(r"\\resulttable(?:\[[^\]]*\])?\{([^}\s]+)\}", tex)
    return [("figures", f) for f in figures] + [("tables", t) for t in tables]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", help=f"groups to copy (default: all): {', '.join(GROUPS)}")
    parser.add_argument("--outputs", type=Path, default=REPO / "outputs", help="the outputs/ tree to copy from")
    parser.add_argument("--missing", action="store_true", help="only list referenced files that do not exist yet")
    args = parser.parse_args()

    if args.missing:
        missing = [f"{d}/{f}" for d, f in referenced_files() if not (REPORT / d / f).exists()]
        print("\n".join(missing) if missing else "Every referenced figure and table exists.")
        return
    unknown = set(args.groups) - set(GROUPS)
    if unknown:
        parser.error(f"unknown group(s): {', '.join(sorted(unknown))}")
    total = 0
    for group in args.groups or GROUPS:
        print(f"[{group}]")
        total += copy_group(args.outputs, group)
    print(f"{total} files copied. Rebuild the PDF (report/README.md).")


if __name__ == "__main__":
    main()
