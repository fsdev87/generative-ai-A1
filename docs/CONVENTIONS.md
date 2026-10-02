# Project conventions

Shared contract for all code in this repository. Several people/agents work in
parallel, so interfaces here are fixed; propose changes instead of diverging.
The assignment brief is `GenAI_Assignment#1.pdf` in the repo root.

## Repository layout and ownership

| Path | Contents | Owner |
|---|---|---|
| `src/data/corruptions.py`, `src/data/pets.py` | Corruptions, Oxford Pets split/manifests/datasets | foundation (done) |
| `src/common/` | Shared training/eval utilities (paths, losses, metrics, checkpoints, W&B, Optuna, ONNX, evaluation, Colab setup) | foundation |
| `src/models/` | `ConvAutoencoder`, `CorruptionClassifier` (shared by Tasks 1-3) | foundation |
| `src/task1/` ... `src/task4/` | Task-specific train / Optuna / evaluate / export scripts | task owner |
| `src/data/fs2k.py` | FS2K pairing, split, cache, paired dataset | Task 4 |
| `backend/` | FastAPI application + its Dockerfile | backend |
| `frontend/` | React + Tailwind application (after the Stitch design) | frontend |
| `report/` | IEEE LaTeX report | report |
| `docs/` | Conventions, API reference, research notes | shared |
| `notebooks/` | `colab_setup.ipynb`, `task{1..4}_colab.ipynb` runners | task owner |
| `manifests/` | Committed splits and corruption manifests. Never regenerate the Pets ones. | foundation |
| `tests/` | pytest suites, one file per area (`test_<area>.py`) | each owner |

Rules for parallel work:
- Only create/edit files in your own area. Do not edit `requirements.txt`, `README.md`,
  `.gitignore` or another area; list needed changes in your final report instead.
- Do not commit, do not `pip install` (ask for missing packages).
- Run only your own test files while others are working (`python -m pytest tests/test_<area>.py -q`).

## Paths

Scripts never hard-code directories. Use `src.common.paths.get_dir(ENV, *subdirs)`
(creates the directory). Defaults are inside the repo; Colab sets them to Google Drive.

| Env var | Local default | Colab value |
|---|---|---|
| `DATA_DIR` | `data/` | `/content/data` (raw datasets, fast local disk) |
| `CACHE_DIR` | `data/cache/` | `MyDrive/GenAI_A1/cache` (preprocessed arrays) |
| `CKPT_DIR` | `outputs/checkpoints/` | `MyDrive/GenAI_A1/checkpoints` |
| `OPTUNA_DIR` | `outputs/optuna/` | `MyDrive/GenAI_A1/optuna` |
| `ONNX_DIR` | `outputs/onnx/` | `MyDrive/GenAI_A1/onnx` |
| `OUTPUT_DIR` | `outputs/` | `MyDrive/GenAI_A1/outputs` (figures, tables, CSVs) |

Per-task subfolders: `get_dir("CKPT_DIR", "task1")`, `get_dir("OUTPUT_DIR", "task1", "figures")`, etc.

## Tensors and labels

- **Tasks 1-3**: float32 RGB in `[0, 1]`, NCHW, 128x128. Restoration models end in a sigmoid.
- **Class order** everywhere (classifier logits, MoE weights, tables):
  `CLASSES = ("clean", "salt", "blur", "occlusion")` from `src.data.corruptions`.
  Index 0 is the clean/identity branch.
- **Severity levels**: `("low", "medium", "high")`; clean entries have level `None`
  (`"none"` once collated by a DataLoader).
- **Task 4**: photos and sketches in `[-1, 1]` (generator ends in tanh); style is an
  int64 index 0/1/2 shown to users as "Style 1/2/3".

## Data access (Tasks 1-3)

```python
from src.data.pets import load_pets, RuntimeCorruptionDataset, ManifestDataset, BalancedBatchSampler
images, manifests = load_pets(smoke=args.smoke)   # images["train"|"val"|"test"]: uint8 (N,128,128,3)
train_ds = RuntimeCorruptionDataset(images["train"], conditions=CLASSES)   # fresh corruption each load
val_ds = ManifestDataset(images["val"], manifests["val"], types=None)       # deterministic
```
`smoke=True` returns a tiny synthetic dataset with matching manifests, for CPU smoke tests.
Use `BalancedBatchSampler` (as `batch_sampler=`) whenever batches must be class-balanced
(Task 2 classifier, Task 3 MoE). Batch sizes must be multiples of 4 there.

## Models and checkpoints

Models are built from a plain config dict and expose it as `model.config`.
Checkpoints are written with `src.common.checkpoint`:
`{"model_config": dict, "model_state": state_dict, "epoch": int, "metrics": dict, ...}` and
rebuilt with `build_model(ConvAutoencoder, ckpt_path)`. Training saves `last.pt` every
epoch (full state, used to resume after a Colab disconnect) and `best.pt` (best validation score).

### Fixed locations shared between tasks

| What | Path |
|---|---|
| Task 1 model | `CKPT_DIR/task1/udae/{last,best}.pt` |
| Task 2 classifier | `CKPT_DIR/task2/classifier/{last,best}.pt` |
| Task 2 specialists | `CKPT_DIR/task2/specialist_{salt,blur,occlusion}/{last,best}.pt` |
| Task 3 model | `CKPT_DIR/task3/moe/{last,best}.pt` |
| Task 4 models | `CKPT_DIR/task4/cgan/{last,best}.pt` |
| Per-entry test results | `OUTPUT_DIR/task1/eval/test_records.csv`, `OUTPUT_DIR/task2/eval/test_records_{oracle,predicted}.csv`, `OUTPUT_DIR/task3/eval/test_records.csv` (one row per test-manifest entry, columns from `evaluate_restoration`) |
| Figures / tables / Optuna reports | `OUTPUT_DIR/task{n}/figures/`, `OUTPUT_DIR/task{n}/tables/`, `OUTPUT_DIR/task{n}/optuna/<study>/` |

Task 3 loads Task 2's `best.pt` files with `build_model`; the cross-task comparison
(Task 1 vs hard routing vs soft MoE) reads the per-entry CSVs.

## Losses and metrics

- `src.common.losses.ssim` — SSIM of Wang et al. (2004), 11x11 Gaussian window, sigma 1.5,
  valid region, mean over channels; always computed in float32 (autocast disabled inside).
- `RestorationLoss(alpha)` = `alpha * L1 + (1 - alpha) * (1 - SSIM)`.
- `src.common.metrics.psnr` — per image, data range 1, **capped at 100 dB** (identical images).
  Where an identity bypass makes outputs exact (Task 2/3 clean inputs), say so in tables
  rather than averaging capped values silently.
- `restoration_score(psnr, ssim) = 0.5 * ssim + 0.5 * psnr / 40` — the shared validation
  objective for restoration models (Optuna and model selection). It is independent of the
  loss weights being tuned, so trials with different alpha are comparable.

## Experiment tracking (Weights & Biases)

- Project `genai-a1` (override with `WANDB_PROJECT`). Use `src.common.tracking.init_run`.
- Group names: `task1-optuna`, `task1-final`, `task2-classifier-optuna`, `task2-classifier-final`,
  `task2-specialists-optuna`, `task2-specialist-{salt,blur,occlusion}`, `task3-optuna`,
  `task3-final`, `task4-optuna`, `task4-final`.
- Every Optuna trial is its own run (params in `config`, per-epoch validation metrics logged).
- Final runs log: train/val losses, metrics per corruption type, sample images at fixed
  intervals using **fixed** validation indices, and the best checkpoint as a model artifact.
- Local smoke tests run with `WANDB_MODE=disabled`.

## Optuna

- `create_study(name, direction)` from `src.common.optuna_utils`: SQLite on local disk,
  copied to `OPTUNA_DIR` after every finished trial, restored from there on restart.
  `run_study(study, objective, n_trials_total, sync)` runs only the remaining trials.
- Study names: `task1_udae`, `task2_classifier`, `task2_specialists`, `task3_moe`, `task4_cgan`.
- TPE sampler with seed 42; `MedianPruner` unless a task justifies another pruner.
- After the study: `save_study_report(study, out_dir)` writes plots (history, importances,
  parallel coordinate, slice) and `trials.csv` for the report.

## ONNX contract

opset 17, float32, dynamic batch axis. Files live in `ONNX_DIR`:

| File | Inputs | Outputs |
|---|---|---|
| `udae.onnx` | `input` [N,3,128,128] in [0,1] | `output` [N,3,128,128] in [0,1] |
| `classifier.onnx` | `input` | `probs` [N,4] softmax, CLASSES order |
| `specialist_salt.onnx`, `specialist_blur.onnx`, `specialist_occlusion.onnx` | `input` | `output` |
| `moe.onnx` | `input` | `output` [N,3,128,128], `weights` [N,4], `branch_outputs` [N,4,3,128,128] (identity, salt, blur, occlusion) |
| `generator.onnx` | `photo` [N,3,128,128] in [-1,1], `style` int64 [N] | `sketch` [N,C,128,128] in [-1,1], C is 1 or 3 |

Every ONNX file has a sidecar `<name>.json` written by `src.common.onnx_utils.write_model_card`
(task, model config, I/O spec, preprocessing, test metrics, hyperparameters, parity result,
W&B run URL, git commit, export time). Parity: ONNX Runtime vs PyTorch (eval mode) on at
least 32 real inputs; require max absolute difference below 1e-4 and record it.

## Scripts and testing

- Scripts are argparse CLIs run as modules: `python -m src.task1.train --config ...`.
- Every training/search/eval script supports `--smoke`: tiny synthetic data, 1-2 epochs or
  trials, W&B disabled, runs on CPU in under ~2 minutes. Run it before handing off to Colab.
- Unit tests for every component; keep each test file fast (< 1 minute on CPU).
- Match the existing style: module docstring explaining purpose, short comments only where
  the reason is not obvious, no dead code.

## Colab task notebooks

Each `notebooks/task{n}_colab.ipynb` starts with the same two cells:
```python
# 1. code
REPO_URL = 'https://github.com/fsdev87/generative-ai-A1.git'
import os
if os.path.exists('/content/repo'):
    !git -C /content/repo pull
else:
    !git clone {REPO_URL} /content/repo
%cd /content/repo
```
```python
# 2. setup: Drive, packages, data cache, W&B, env vars
from src.common.colab import setup
setup(fs2k=False)   # True for Task 4
```
followed by the task's Optuna, final training, evaluation and ONNX export cells, each a
single `!python -m ...` command with a short markdown explanation.
