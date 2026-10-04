# Report build instructions

IEEE conference paper (`\documentclass[conference]{IEEEtran}`) built with
pdflatex + bibtex.

| File | Contents |
|---|---|
| `main.tex` | The report. |
| `references.bib` | All references. Every entry has a comment above it with the URL it was verified against. |
| `IEEEtran.cls`, `IEEEtran.bst` | The IEEE class and bibliography style, copied from CTAN (<https://ctan.org/pkg/ieeetran>) so the document builds anywhere, including Overleaf, without installing anything. |
| `main.pdf` | Build output (not committed). |

## Build

From inside `report/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Four passes are needed: the first writes `main.aux`, `bibtex` turns the citations
into `main.bbl`, and the last two resolve the bibliography and then the
cross-references and page numbers.

Or, equivalently, with latexmk:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -c          # remove the auxiliary files, keep main.pdf
```

Always pass `-interaction=nonstopmode`. Without it a missing package makes
MiKTeX open a blocking installation dialog, which hangs an unattended build.

### Overleaf

Upload the whole `report/` folder (including `IEEEtran.cls` and `IEEEtran.bst`),
set `main.tex` as the main document and leave the compiler on pdfLaTeX. Overleaf
runs the pdflatex/bibtex sequence above automatically. No further packages are
needed: the preamble uses only `fontenc`, `inputenc`, `amsmath`, `amssymb`,
`graphicx`, `booktabs`, `multirow`, `xcolor`, `url`, `tikz`, `microtype` and
`hyperref`, all of which are in every standard TeX distribution. Upload the
`figures/` and `tables/` folders too.

### A missing package on MiKTeX

If a package really is missing, install it explicitly from the command line
rather than letting the dialog appear:

```bash
miktex packages install <package>
# older installations: mpm --install=<package>
```

## Current state of the document

Task 1, the Task 2 Optuna searches, the experimental setup, the application,
the limitations and the AI-use appendix are written from the real results
(`docs/experiment_log.md` and the files in `outputs/`). Everything that waits on
the Kaggle runs (Task 2 final models, Tasks 3 and 4) is marked:

- `\todo{...}` renders in red as `[TODO: ...]` and states exactly which figure,
  table or number belongs there.
- `\tbd` renders as a red `--` and marks a single empty cell in a placeholder
  table.
- Figures and generated tables are included with `\resultfig{<file>}` and
  `\resulttable{<file>}` (defined in the preamble). If `figures/<file>` or
  `tables/<file>` exists it is used; otherwise a red placeholder names the
  missing file and the script output it comes from. **Copying the file in is
  enough: no LaTeX has to be edited.** Generated tables keep only their tabular;
  the caption and label in `main.tex` apply, and a tabular wider than the
  column is scaled down to fit.

To find what is still outstanding:

```bash
python report/collect_outputs.py --missing     # figure and table files still missing
grep -n "todo{\|tbd" main.tex                  # text and numbers still to write
```

Still to add by hand: the YouTube link (introduction and Availability), the
download link of the ONNX models (Availability), the four application
screenshots and the Task 4 training curves exported from Weights & Biases
(file names below).

## Figures and tables

`collect_outputs.py` copies everything from the `outputs/` tree with one naming
rule (run it from the repository root after downloading new outputs):

```bash
python report/collect_outputs.py                    # all groups
python report/collect_outputs.py task2 task3 task4  # only the Kaggle results
```

| Source (`outputs/`) | Destination (`report/`) |
|---|---|
| `corruption_grid.png` | `figures/corruption_grid.png` |
| `taskN/figures/X.png` | `figures/taskN_X.png` |
| `taskN/tables/X.tex` | `tables/taskN_X.tex` |
| `task3/mixed/X.png`, `X.tex` | `figures/task3_X.png`, `tables/task3_X.tex` |
| `task1/optuna/task1_udae/X.png` | `figures/task1_optuna_X.png` |
| `task2/optuna/task2_classifier/X.png` | `figures/task2_classifier_optuna_X.png` |
| `task2/optuna/task2_specialists/X.png` | `figures/task2_specialists_optuna_X.png` |
| `task3/optuna/task3_moe/X.png` | `figures/task3_optuna_X.png` |
| `task4/optuna/task4_cgan/X.png` | `figures/task4_optuna_X.png` |

Do not copy the Task 2 tables and figures of the Colab run (`outputs/task2/tables`,
`figures`, `eval`): Task 2's final models are being retrained on Kaggle and only
those results belong in the report.

Figures that no script writes, saved by hand under these names:

| File | Content |
|---|---|
| `figures/app_universal.png`, `app_hard.png`, `app_moe.png`, `app_sketch.png` | screenshots of the four workspaces with the trained models |
| `figures/task4_training_curves.png` | W&B export of the final Task 4 run: `train/d_real`, `train/d_fake`, `train/g_adv`, `train/g_l1` and validation metrics |

The Google Stitch screens in `figures/stitch/` are copies of `design/stitch/*.png`.
The evaluation scripts write figures at 200 dpi, which is adequate for a
two-column IEEE page.

## Checks before submitting

```bash
grep -c "Overfull" main.log        # should stay small; > 10pt is visible
grep -i "undefined" main.log       # must find no undefined references or citations
```

A clean build of the current draft (Task 1 and the Task 2 searches filled in, the
Kaggle results still placeholders) produces a 28-page PDF with no undefined
references or citations and no overfull boxes. Placeholder boxes are sized like
the final figures, so the page count should change little as they are replaced.
