# Experiment log

Observations and interpretations from the real training runs, written down as they happen
so the report can use them. Raw numbers, tables and figures are produced by the scripts
(Google Drive `MyDrive/GenAI_A1/outputs/` and W&B project `mustafafaraz87-fast-nuces/genai-a1`);
this file records what they mean and anything that is not in those outputs.

Hardware for every run: Google Colab free tier, NVIDIA Tesla T4, mixed precision (fp16 autocast).

---

## Task 1 — Universal denoising autoencoder

### Optuna search (2026-10-03, study `task1_udae`)

- Command: `optuna_search --n-trials 40 --epochs 15 --timeout-min 90`.
- 33 trials (0–32) finished inside the 90-minute cap; the cap, not the trial count, ended the search.
- **Best: trial 32 — validation score 0.6263** (score = 0.5·SSIM + 0.5·PSNR/40 on the val manifest).
  The best trial was the last one, so the search was still improving when it stopped.
- Selected configuration:

  | Hyperparameter | Range searched | Selected |
  |---|---|---|
  | learning rate | 1e-4 – 3e-3 (log) | 8.28e-4 |
  | batch size | 16 / 32 / 64 | **16** (lower edge) |
  | latent channels (latent = C×8×8) | 8 / 16 / 32 / 64 | **64** (upper edge) → 4,096 values, 12× compression |
  | base channels | 16 / 32 / 48 / 64 | 48 |
  | dropout | 0 – 0.3 | 0.016 |
  | alpha (L1 weight) | 0.1 – 0.95 | **0.285** |

- Trial 0 = the brief's starting point (alpha 0.8, base 32, latent 32, batch 32, lr 1e-3): its
  score was 0.533 at epoch 8 of 15 (final value in `trials.csv`).

**Interpretations for the report**

1. *Loss weighting.* Optuna moved alpha from the brief's 0.8 to 0.285, i.e. most of the weight on
   the (1 − SSIM) term. This agrees with Zhao et al.'s L1 + MS-SSIM finding, whose 0.84 weights
   the MS-SSIM term (≈ 0.16 in our parameterisation) — not with a literal reading of 0.8.
   Caveat to state: the objective itself is half SSIM, which favours SSIM-heavy losses.
2. *Bottleneck at the edge of the range.* The largest latent (64 channels, 12× compression) won:
   a wider bottleneck passes more detail, so reconstruction improves monotonically with it.
   The range was capped deliberately — much larger latents approach an identity mapping, which
   the brief rules out ("meaningful bottleneck"). This is a quality-vs-bottleneck trade-off, not
   an oversight of the search.
3. *Short-budget bias.* The smallest batch size won. With a fixed 15-epoch budget per trial,
   smaller batches give more optimiser steps (184 vs 46 per epoch), so short trials favour
   configurations that learn fast. The final run trains 4× longer, which mitigates this.
4. Possible improvement given more GPU time: more trials (the best was the last trial).

### Final training (2026-10-03, W&B run `udae`, id `i066ij8w`)

- 9.31 M parameters; latent 64×8×8 = 4,096 (12× compression); no skip connections.
- 184 steps/epoch, ≈ 19 s/epoch on the T4; 60 epochs ≈ 20 min; early stopping (patience 12) never triggered.
- **Best validation score 0.7023 at epoch 59** (of 60); `best.pt` uploaded to W&B as an artifact.
- Validation per condition at the best epoch:

  | Condition | PSNR (dB) | SSIM |
  |---|---|---|
  | clean | 25.59 | 0.833 |
  | blur | 25.31 | 0.816 |
  | occlusion | 21.02 | — (score 0.620) |
  | salt-and-pepper | see W&B summary | |

**Interpretations for the report**

1. *Learning-rate schedule.* The score was 0.629 at epoch 17 and 0.702 at epoch 59: most of the
   gain came in the low-learning-rate tail of the warm-up + cosine schedule. Validation PSNR
   oscillated by ±0.7 dB while the learning rate was high and settled as it decayed.
2. *Occlusion is the hardest condition* (≈ 4 dB below the others): pixels behind a black mask carry
   no information, so the model must synthesise plausible content, whereas noise and blur leave
   evidence in the corrupted pixels.
3. *Cost of universality on clean inputs.* Clean images come back at 25.6 dB, not unchanged:
   everything must pass the 4,096-value bottleneck, so fine detail is always lost. Task 2's
   identity bypass returns correctly classified clean images exactly — a direct point for the
   cross-task comparison.

### Test evaluation (official test manifest, 36,690 entries; `best.pt`, epoch 59)

- All corrupted test inputs (salt + blur + occlusion, all severities, 33,021 entries):
  **PSNR 23.82 dB, SSIM 0.7845**. Per type × severity, with the identity baseline (no
  restoration) and the oracle classical baselines (told the corruption type and parameters):

  | Input | Identity PSNR / SSIM | Oracle classical PSNR / SSIM | **UDAE PSNR / SSIM** |
  |---|---|---|---|
  | clean | 100 (exact) / 1.000 | 100 / 1.000 | **25.12 / 0.828** |
  | salt low | 20.13 / 0.602 | 30.39 / 0.893 | **25.13 / 0.828** |
  | salt medium | 15.87 / 0.339 | 29.51 / 0.884 | **25.13 / 0.826** |
  | salt high | 13.14 / 0.204 | 27.78 / 0.863 | **25.04 / 0.821** |
  | blur low | 32.34 / 0.944 | 36.38 / 0.976 | **25.28 / 0.827** |
  | blur medium | 26.77 / 0.806 | 28.32 / 0.854 | **25.14 / 0.819** |
  | blur high | 24.35 / 0.692 | 25.41 / 0.735 | **24.34 / 0.756** |
  | occlusion low | 16.67 / 0.862 | 26.74 / 0.930 | **23.03 / 0.782** |
  | occlusion medium | 13.26 / 0.726 | 23.03 / 0.860 | **21.52 / 0.736** |
  | occlusion high | 10.76 / 0.536 | 20.25 / 0.756 | **19.80 / 0.665** |
  | all corrupted, low | 23.05 / 0.803 | 31.17 / 0.933 | **24.48 / 0.812** |
  | all corrupted, medium | 18.63 / 0.624 | 26.95 / 0.866 | **23.93 / 0.794** |
  | all corrupted, high | 16.08 / 0.477 | 24.48 / 0.785 | **23.06 / 0.747** |

  (3,669 entries per type × level row; source `outputs/task1/tables/comparison_by_type_level.csv`.)
- Sanity check of the exported model in the real backend (`/api/restore/universal`, sample
  Abyssinian_201, seed 7): salt high 12.87 → 24.02 dB, blur low 30.94 → 24.42 dB, occlusion medium
  14.10 → 19.47 dB, clean 100 → 24.30 dB — the same pattern as the test set.

**Interpretation — the ~25 dB ceiling.** The UDAE's output quality is nearly constant (≈ 25 dB,
SSIM ≈ 0.82) for clean, salt and low/medium blur inputs: the 12× bottleneck caps how faithfully
any image can be reproduced (clean detail ratio 0.50). Consequently it helps most where the
corruption is severe (salt high +11.9 dB, occlusion high +9.0 dB over the input) but *hurts* mild
corruptions (blur low −7.1 dB) and clean images (exact → 25 dB). This is the direct motivation for
Task 2's identity bypass and specialists, and for Task 3. The oracle classical baselines beat the
UDAE in PSNR almost everywhere, but they are given the corruption type, its parameters and the
occlusion mask; the UDAE is blind. Only for high blur does the UDAE reach a higher SSIM (0.756 vs
0.735) than unsharp masking with the known kernel.
- **Salt-and-pepper impulse survival: 0.0023** — of the pixels hit by an impulse, 0.23 % are still
  closer to the impulse than to the clean value after restoration (identity = 1, perfect = 0).
- **Clean detail ratio: 0.498** — on clean inputs the output has about half the high-frequency
  (Laplacian) energy of the clean image (1 = equally detailed).

**Interpretations for the report**

1. The two diagnostics quantify both sides of the bottleneck: no shortcut for the corruption
   (99.8 % of impulses removed) and a real price in detail (half the fine texture lost even when
   nothing needed fixing). Limited skips would trade the second for the first — that is what the
   ablation (`--skip-resolutions 16/32/128`) measures; not run yet (optional, GPU quota).

### ONNX export

- `udae.onnx` + model card `udae.json`; parity with PyTorch: **max |diff| 8.64e-7** on 64 real test
  inputs (single image 3.58e-7), far below the 1e-4 threshold.

---

## Task 2 — Hard routing

### Classifier Optuna search (2026-10-03, study `task2_classifier`)

- Command: `optuna_classifier --n-trials 30 --epochs 12 --timeout-min 45`; 30 trials.
- **Best: trial 1 — validation macro-F1 0.9959.** Selected: lr 1.70e-3, batch 32, channels
  32-64-128-256, 1 conv per stage, dropout 0.146, weight decay 2.9e-4.
- Pruned trials still reached macro-F1 ≈ 0.98 by epoch 3 (e.g. trial 29: 0.9783).

**Interpretation:** the best configuration appeared at trial 1 and many configurations score
≈ 0.98–0.996, i.e. the objective is nearly saturated: detecting the corruption type is easy for a
CNN on this data. The remaining errors are expected among the mildest blurs vs naturally soft
clean photos (to be confirmed by the blur-strength analysis in the classifier evaluation).

### Final classifier and test evaluation (after the checkpoint fix)

- Final run: best validation macro-F1 **1.0000 at epoch 47** (60-epoch schedule).
- **Test (36,690 entries): accuracy 0.9989, macro-F1 0.9981, macro-precision 0.9980,
  macro-recall 0.9982** — about 40 errors in total. Validation (736): 1.0000 on every metric.
- Error analysis: 0.41 % of clean test images predicted "blur" (≈ 15 images), 0.08 % predicted
  "occlusion" (≈ 3); only 0.055 % of the *low*-severity blur entries predicted "clean" (≈ 2).
  Full tables/figures: `outputs/task2/tables`, `outputs/task2/figures`.

**Interpretations for the report**

1. Contrary to the expectation that the mildest blur would be confused with clean photos, the
   classifier detects even the (3, 0.7) blur level almost perfectly; its dominant error is the
   reverse — flagging naturally soft clean photos as blurred.
2. With a near-perfect classifier, predicted routing should almost equal oracle routing: the
   brief's "classifier errors cause restoration failures" analysis will contain few cases, to be
   discussed individually.
3. Task 3's gate is initialised from this classifier, so on the standard test set the soft MoE is
   likely to route almost one-hot; the mixed-corruption experiment is where soft routing can differ.

### Incident: quick check contaminated the real checkpoints (found and fixed 2026-10-03)

- Symptom: the classifier evaluation reported test accuracy 0.300, macro-recall exactly 0.250 and
  validation macro-F1 0.1000 — i.e. one class predicted for every input, and the validation score
  identical to the `--smoke` quick check's.
- Cause: Task 2's `--smoke` runs saved into the real checkpoint folders (Tasks 1, 3 and 4 use a
  separate `smoke/` subfolder). A quick check run before the real training left a *finished* tiny
  model there; the final run with `--resume` then printed "nothing to do", so the real classifier
  (and later the salt specialist) was never trained, and evaluation used the tiny model. A second,
  independent bug crashed the evaluation's blur-detection plot (an error bar of −1e-17 at p = 0/1).
- Fix (commit `bba804b`): a real run ignores a smoke `last.pt` and trains from scratch;
  evaluation and export refuse smoke checkpoints; error bars clipped at 0; regression test added.
- Unaffected: the Optuna searches (they never write checkpoints) and the blur and occlusion
  specialists (no smoke checkpoint existed for them).
- For the AI-use appendix: an example of AI-generated code passing its own tests but failing in the
  real workflow, caught by checking the evaluation numbers against what a trained model must produce.

### Specialist Optuna search (2026-10-03, study `task2_specialists`)

- Command: `optuna_specialists --n-trials 20 --epochs 6 --timeout-min 80`; each trial trains all
  three specialists for 6 epochs; objective = mean of their validation restoration scores.
- First session stopped manually about 11 minutes in (while the checkpoint bug was being fixed):
  trials 0–2 had finished and were synced to Drive; the interrupted trial was never synced, so the
  resumed study continued from trial 3. Result: exactly **20 finished trials (0–19)**, none failed,
  all within the 80-minute cap.
- **Best: trial 10 — mean validation score 0.6819** (salt 0.7006, blur 0.7295, occlusion 0.6155).
- Selected (shared by the three specialists): lr 2.37e-3, batch 16, base channels 48,
  latent **16×16×32 = 8,192 values (6× compression)**, alpha 0.516.

**Interpretations for the report**

1. *Bottleneck confound in the Task 1 vs Task 2 comparison.* The specialists' search space
   included 16×16 latents and selected one twice as large as Task 1's (8×8×64 = 4,096). Part of any
   advantage of hard routing over the universal model can therefore come from the larger
   bottleneck rather than from specialisation alone. State this as a limitation in the cross-task
   comparison (both are genuine bottlenecks: 6× and 12× compression).
2. *Loss weighting again below the brief's 0.8:* alpha 0.52 (Task 1 chose 0.29) — the searches
   consistently move weight toward the SSIM term.
3. *Short-budget bias again:* the smallest batch size (16) won again, as in Task 1.
4. After only 6 epochs the specialists' validation scores (blur 0.73) are already close to the
   fully trained universal model's (blur 0.724 after 60 epochs).

### Move to Kaggle (2026-10-04)

Colab's free GPU quota ran out before the specialists' final runs completed, on the deadline day.
The remaining work moved to Kaggle (2× T4) with `scripts/kaggle_run.py`: Task 2's final classifier
and specialists were retrained there from scratch with the configurations the Colab searches
selected (`configs/task2_*.yaml`); Kaggle has no access to the Colab files, so no partial Colab
checkpoint could be reused. The deployed Task 2 models and their test numbers therefore come from
the Kaggle run (the Colab classifier numbers above are superseded). Tasks 3 and 4 used reduced
Optuna budgets because of the deadline (Task 3: 6 trials of 1 warm-up + 3 joint epochs, final
2 + 10; Task 4: 10 trials of 8 epochs, final 120 epochs).

On Kaggle the Task 2 classifier stopped early after 2.4 min (≈ 3.5 s/epoch); the specialists took
≈ 12–13 s/epoch. All 17 pipeline steps finished at 11:12 UTC (16:12 local), ≈ 1 h after starting.

## Task 3 — Soft mixture of experts

### Kaggle run (2026-10-04)

- Optuna study `task3_moe`: 6 trials (1 warm-up + 3 joint epochs each), 15.9 min.
- Final run: 2 warm-up + 10 joint epochs, 9.3 min (≈ 24 s per warm-up epoch, ≈ 47 s per joint
  epoch on a T4); best at epoch 10, **validation score 0.8260** (0.8251 after warm-up).
  Selected temperature **tau = 2.678** (search range 0.3–3.0).
- Routing during training: argmax routing accuracy 0.995–0.996 on validation; mean branch usage
  0.25 / 0.25 / 0.25 / 0.25 (identity, salt, blur, occlusion) — no collapse.
- Test evaluation 6.1 min over 36,690 entries; `moe.onnx` 49.2 MB.

**Interpretation (to confirm with the test outputs):** the search selected a temperature near the
soft end of the range, so the gate blends experts even though its top choice is almost always the
right one; usage shows no collapse. Per-type numbers, routing heatmap and the mixed-corruption
experiment: `outputs/task3` (pending download).

## Task 4 — Face-to-sketch cGAN

### Kaggle run (2026-10-04, GPU 1, finished before Task 3)

- Optuna study `task4_cgan`: 10 trials × 8 epochs; final run 120 epochs (details pending from
  `outputs/task4`).
- `generator.onnx` parity with PyTorch: max |diff| **2.86e-6** on 64 test photos (single image
  1.36e-6).

---

## Still needed for the report

- Test-set results and figures for every task (scripts write them to Drive `outputs/`).
- At the end: download `MyDrive/GenAI_A1/outputs/` into the repository's `outputs/` folder
  (gitignored) so the figures and tables can be placed in `report/figures/` and the LaTeX tables.
- Screenshots of the running application with the trained models (all four workspaces).
- The YouTube link of the demonstration video.
