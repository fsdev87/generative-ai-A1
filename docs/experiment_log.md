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

### Test evaluation

*Pending (cell 6).*

---

## Task 2 — Hard routing

*Not started.*

## Task 3 — Soft mixture of experts

*Not started.*

## Task 4 — Face-to-sketch cGAN

*Not started.*

---

## Still needed for the report

- Test-set results and figures for every task (scripts write them to Drive `outputs/`).
- At the end: download `MyDrive/GenAI_A1/outputs/` into the repository's `outputs/` folder
  (gitignored) so the figures and tables can be placed in `report/figures/` and the LaTeX tables.
- Screenshots of the running application with the trained models (all four workspaces).
- The YouTube link of the demonstration video.
