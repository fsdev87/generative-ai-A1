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
`graphicx`, `booktabs`, `multirow`, `xcolor`, `url` and `hyperref`, all of which
are in every standard TeX distribution.

### A missing package on MiKTeX

If a package really is missing, install it explicitly from the command line
rather than letting the dialog appear:

```bash
miktex packages install <package>
# older installations: mpm --install=<package>
```

## Current state of the document

The report is a working draft: the prose that does not depend on experimental
results (abstract framing, introduction, related work, dataset preparation,
methodology for all four tasks, experimental setup, application architecture,
limitations) is written, and everything that waits on a training run is marked.

- `\todo{...}` renders in red as `[TODO: ...]` and states exactly which figure,
  table or number belongs there.
- `\tbd` renders as a red `--` and marks a single empty cell in a results table.
  The note under each such table says, with a `\todo`, which output file
  supplies that table's numbers.
- `\placeholderfig{...}` / `\placeholderfigwide{...}` draw a framed red box of
  roughly the final figure's size, describing the figure and naming the file
  that will produce it. Replace the whole call with
  `\includegraphics[width=\columnwidth]{figures/<name>.png}` (or
  `width=\textwidth` inside a `figure*`) once the figure exists. Every figure
  and table already has its `\label`, so cross-references work now and keep
  working afterwards.

To find what is still outstanding:

```bash
grep -n "todo{\|tbd\|placeholderfig" main.tex
```

Two further items to fill in before submission: the author block at the top of
`main.tex`, and the YouTube link, which appears twice (in the introduction and
in the Availability section).

## Figures

Copy the PNGs named in the placeholder boxes from the `outputs/` tree into
`report/figures/` and keep their names, so each placeholder maps to one file.
The evaluation scripts write them at 200 dpi, which is adequate for a two-column
IEEE page.

## Checks before submitting

```bash
grep -c "Overfull" main.log        # should stay small; > 10pt is visible
grep -i "undefined" main.log       # must find no undefined references or citations
```

A clean build of the current draft produces an 18-page PDF with no undefined
references or citations and no LaTeX warnings. The page count will change as
placeholders are replaced by real figures, tables and numbers.
