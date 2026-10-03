# Project status

Living checklist of what is done, what is in progress and what is left.
Update it whenever a piece lands, so work can always be resumed from here.

## Pipeline order

Code is written in parallel, but GPU training is sequential on one free Colab T4:
**Task 1 → Task 2 → Task 3 (needs Task 2's checkpoints) → Task 4** (independent, can go any time).
Estimated total GPU time ≈ 10 h, realistically spread over several days of free-tier quota.

## Done (committed)

| Piece | Notes |
|---|---|
| Data pipeline (Tasks 1-3) | Oxford-IIIT Pet, 2944/736/3669 split (seed 42), runtime corruptions, deterministic val/test manifests (36,690 test entries), balanced batch sampler |
| Shared foundation | `src/common/` (paths, losses, metrics, checkpoints, W&B, Optuna with Drive sync, ONNX export/parity/model cards, restoration evaluation, Colab setup), `src/models/` (ConvAutoencoder, CorruptionClassifier) |
| Backend | FastAPI, all required endpoints, upload validation, ONNX registry, dummy-model generator, Docker image, `docs/api.md`, 88 tests |
| Task 2 | Classifier + 3 specialists + hard routing (oracle/predicted), Optuna studies, evaluation incl. misrouting analysis, ONNX export, notebook, 13 tests |
| Task 4 stage 1 | FS2K pairing/split/cache, preprocessing (matches the backend exactly), U-Net generator with FiLM style conditioning, projection PatchGAN, 23 tests |
| Task 1 | Universal autoencoder, Optuna, evaluation with classical baselines, limited-skip ablation, ONNX export, notebook, 18 tests |
| Task 3 | Soft MoE from Task 2 checkpoints, warm-up + joint fine-tuning, collapse-aware Optuna, gating analysis, cross-task comparison, ONNX export, notebook, 17 tests |
| Task 4 stage 2 | cGAN training with separate loss logging, Optuna, test evaluation per style and photo source, generator ONNX export, notebook, 35 tests in total |
| Report skeleton | `report/main.tex` (IEEE, 18 pages, builds clean), `references.bib` (114 verified entries), `docs/research_notes.md` |
| Google Stitch design | `design/stitch/` (exported HTML + screenshots of all four workspaces) |
| Frontend + Docker Compose | React + Tailwind from the Stitch code, real data only, self-hosted fonts; `docker compose up --build`; 54 tests; verified end to end with placeholder models |
| SSIM in the app | Backend returns SSIM next to PSNR (NumPy, matches training definition to 3.2e-7) |
| Colab setup | `notebooks/colab_setup.ipynb`, Drive layout `MyDrive/GenAI_A1/{archives,cache,checkpoints,optuna,onnx,outputs}` |

## In progress

Nothing; all model code, the backend and the report skeleton are committed.

## Not started

| Piece | Blocked on |
|---|---|
| Report author block + YouTube link | User (`report/main.tex`, two places) |
| Optional: LPIPS as a perceptual metric for Task 4 | Decision; the report's abstract placeholder currently mentions it |
| Sample pet images in `backend/app/samples/pets/` | Streaming 8 official test images (in progress); faces left out for licence reasons (upload/webcam instead) |
| Trained ONNX models in `./models` + download link | Training runs |
| Training runs | Task 1 running on Colab (Optuna started 2026-10-03); then Task 2, Task 3, Task 4 |
| Demonstration video (5-7 min, YouTube) | Working app + trained models |
| AI-use appendix | Report skeleton |

## Assignment requirements checklist

- [x] Runtime corruption pipeline with the exact specified ranges and fixed test severities
- [x] Deterministic validation and test corruption manifests
- [x] Task 1 universal autoencoder with a genuine bottleneck (+ limited-skip ablation)
- [x] Task 2 classifier (balanced batches) + 3 independent specialists + identity bypass
- [x] Task 2 oracle vs predicted routing comparison
- [x] Task 3 soft MoE initialised from Task 2, warm-up then joint fine-tuning
- [x] Task 3 gating analysis (weights per type/severity, heatmap, collapse checks)
- [x] Task 4 cGAN: U-Net generator, PatchGAN discriminator, style embedding in both — *stage 2 in progress*
- [x] Optuna in all four tasks
- [x] W&B tracking (hyperparameters, losses, metrics, checkpoints, visual outputs)
- [x] ONNX export + PyTorch parity checks for every inference model
- [x] FastAPI backend with health + the four task endpoints
- [x] Google Stitch design (evidence in `design/stitch/`; figure still to be added to the report)
- [x] React + Tailwind frontend, four workspaces
- [x] Docker Compose, one documented command
- [ ] IEEE LaTeX report with all required figures/tables and interpretation
- [ ] GitHub repository with README and execution instructions
- [ ] 5-7 minute YouTube demonstration video
- [ ] AI-use appendix

## Known issues / decisions to revisit

- FS2K style is confounded with photo source in the official training split; test metrics are
  reported per style **and** per photo source (`docs/fs2k_notes.md`).
- All three test "high" severities sit at the top edge of the training ranges, and no training
  blur ever reaches the strength of the (7, 2.5) test level — expect the weakest scores there.
- **FID/LPIPS deliberately not used for Task 4.** FID over the 46 style-2 test pairs would be
  statistically meaningless, and LPIPS is trained on natural images rather than line drawings.
  Sketch quality is reported with L1, PSNR, SSIM and an `edge_ratio` sharpness diagnostic
  (generated edge energy / ground truth), which directly measures the blur an L1 term induces.
- Task 4 mixed-precision stability could not be verified on a CPU-only machine; watch the first
  Colab epoch and fall back to `--no-amp` if the losses are not finite.
- `src/common/onnx_utils.py` exports with `dynamo=False` to get opset 17; torch 2.12 defaults to
  `dynamo=True`, which produces opset 18 and can emit invalid `Split` nodes.
