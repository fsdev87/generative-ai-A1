# GenAI Studio — Assignment 1

Four generative-AI systems for image restoration and paired image-to-image generation,
trained with PyTorch, tuned with Optuna, tracked with Weights & Biases, exported to ONNX and
served through a single browser application (React + Tailwind frontend, FastAPI backend, Docker Compose).

| Task | System | Dataset |
|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder | Oxford-IIIT Pet |
| 2 | Corruption classifier + hard-routed specialist autoencoders | Oxford-IIIT Pet |
| 3 | Jointly trained soft mixture-of-experts restoration | Oxford-IIIT Pet |
| 4 | Style-conditioned face-to-sketch conditional GAN | FS2K |

- **Technical report:** `report/main.tex` (IEEE format)
- **Demonstration video:** TODO
- **Trained models (ONNX):** TODO — download link
- **Project status:** [docs/STATUS.md](docs/STATUS.md)

## Quick start — run the application

Prerequisites: Docker Desktop (or Docker Engine with Compose v2). No Python, Node or VS Code needed.

```bash
git clone https://github.com/fsdev87/generative-ai-A1.git
cd generative-ai-A1
# 1. Fetch the trained ONNX models into ./models (see "Trained models" above)
# 2. Start everything
docker compose up --build
```

Then open <http://localhost:5173>. The API is on <http://localhost:8000>, with interactive
documentation at <http://localhost:8000/api/docs> and a health check at
<http://localhost:8000/api/health> that lists which models are loaded.

The four workspaces are: **Universal Restoration**, **Hard-Routed Restoration**,
**Soft Mixture-of-Experts Restoration** and **Face-to-Sketch Generator**.

## Repository layout

```
src/common/        Shared training utilities (paths, losses, metrics, checkpoints,
                   W&B, Optuna, ONNX export/parity, evaluation, Colab setup)
src/models/        ConvAutoencoder, CorruptionClassifier (shared by Tasks 1-3)
src/data/          Corruption pipeline, Oxford-IIIT Pet and FS2K datasets
src/task1..task4/  Per-task training, Optuna search, evaluation and ONNX export
scripts/           Dataset download and preparation, preview figures
manifests/         Committed splits and deterministic corruption manifests
notebooks/         Colab runners: setup + one per task
backend/           FastAPI application and its Dockerfile
frontend/          React + Tailwind application
report/            IEEE LaTeX technical report
docs/              Conventions, per-task design notes, research notes, API reference
tests/             pytest suites
```

## Reproducing the models

Training runs on a free Google Colab T4 GPU; every script resumes after a disconnect
(checkpoints and Optuna studies live on Google Drive).

1. Open `notebooks/colab_setup.ipynb` in Colab, select **Runtime → Change runtime type → T4 GPU**,
   add a Colab secret `WANDB_API_KEY` (from <https://wandb.ai/authorize>), and run all cells.
   This mounts Drive, installs dependencies, downloads both datasets and builds the caches.
2. Run the task notebooks in order: `task1_colab.ipynb`, `task2_colab.ipynb`,
   `task3_colab.ipynb` (requires Task 2's checkpoints), `task4_colab.ipynb` (independent).

Each notebook runs its Optuna search, trains the selected configuration, evaluates on the
official test split and exports ONNX models with a PyTorch parity check.

### Running locally instead

```bash
pip install -r requirements.txt
python scripts/download_data.py                      # both datasets
python scripts/prepare_pets.py                       # split, manifests, 128x128 cache
python scripts/prepare_fs2k.py                       # FS2K split and cache
python -m src.task1.optuna_search --n-trials 40      # example: Task 1
python -m src.task1.train --config outputs/task1/best_config.yaml
python -m src.task1.evaluate
python -m src.task1.export_onnx
```

Directories are configurable through the environment variables `DATA_DIR`, `CACHE_DIR`,
`CKPT_DIR`, `OPTUNA_DIR`, `ONNX_DIR` and `OUTPUT_DIR` (defaults are inside the repository).
Every training, search and evaluation script accepts `--smoke` for a fast CPU check on
synthetic data.

## Tests

```bash
pip install -r requirements.txt -r backend/requirements-dev.txt
python -m pytest tests backend/tests -q
```

## Datasets

Both are downloaded by `scripts/download_data.py` and are **not** stored in this repository.

- **Oxford-IIIT Pet** (Parkhi et al., 2012) — <https://www.robots.ox.ac.uk/~vgg/data/pets/>.
  The official `trainval` list is split 80/20 into training and validation with seed 42;
  the official test list is untouched until final evaluation. Corruptions are generated at
  runtime during training and from committed deterministic manifests for validation and test.
- **FS2K** (Fan et al., 2022) — <https://github.com/DengPingFan/FS2K>. 2,104 photo-sketch pairs
  in three styles; 15% of the official training portion is held out for validation
  (style-stratified, seed 42).

## Documentation

| Document | Contents |
|---|---|
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | Shared contract: paths, tensors, checkpoints, ONNX, testing |
| [docs/api.md](docs/api.md) | Backend endpoint reference |
| [docs/task1_notes.md](docs/task1_notes.md), [task2](docs/task2_notes.md), [task3](docs/task3_notes.md), [fs2k_notes.md](docs/fs2k_notes.md) | Per-task design decisions and justifications |
| [docs/research_notes.md](docs/research_notes.md) | Literature behind the design decisions |
| [docs/stitch_prompt.md](docs/stitch_prompt.md) | Google Stitch interface design prompts |

## Acknowledgements

Oxford-IIIT Pet and FS2K are used under their respective licences. Development used
AI coding assistance; the AI-use appendix of the technical report lists the tools, what
they were used for and how their output was verified.
