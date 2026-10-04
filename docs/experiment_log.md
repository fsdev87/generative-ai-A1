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

### Final models — retrained (2026-10-04 evening, Kaggle)

The first final model (no skip, trained on Colab) showed an output-quality ceiling of about
25 dB and lost most fine detail even on clean inputs. Task 1 was therefore retrained on Kaggle
from the same Optuna-selected configuration (`configs/task1.yaml`) as a skip-connection
ablation: no skip (`udae`), one limited skip at 16×16 (`udae_skip16`) and at 32×32
(`udae_skip32`), each 60 epochs, evaluated on the full test manifest, compared, and exported.
The first run's numbers were removed from this log and the report; results below come from the
retrain only.

*Results: pending (Kaggle retrain in progress).*

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

### Final models — retrained (2026-10-04 evening, Kaggle)

The first Kaggle run's Task 2 models worked (classifier ≈ 99.6 % test accuracy) but the pure-
bottleneck specialists over-smoothed and filled occlusions with blurry smears. The final run
retrains the classifier (same configuration) and the three specialists with one limited skip at
32×32 (`configs/task2_specialists_skip.yaml`). Hyperparameters come from the Colab searches,
which were run without skips (limitation). The first run's numbers were removed.

*Results: pending (Kaggle retrain in progress).*

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

### Retrain (2026-10-04 evening, Kaggle)

Rebuilt from the retrained Task 2 models: new Optuna study (6 trials of 1 warm-up + 3 joint
epochs), final 2 warm-up + 10 joint epochs. Lesson kept from the first run: a high selected
temperature did not make routing soft — with the cross-entropy term on the raw gate logits, the
logits grow and routing stays near one-hot, so check the routing entropy rather than tau.

*Results: pending (Kaggle retrain in progress).*

## Task 4 — Face-to-sketch cGAN

### Retrain (2026-10-04 evening, Kaggle)

The first run (10 trials × 8 epochs, final 120 epochs) produced soft strokes; its search picked a
discriminator learning rate 4.5× below the generator's. Task 4 was rerun with a larger search
(18 trials × 12 epochs) whose discriminator learning-rate range starts at 1.5e-4, and a 200-epoch
final run. The first run's numbers were removed; results below come from the rerun only.

*Results: pending (Kaggle retrain in progress).*

## Still needed for the report

- Test-set results and figures for every task (scripts write them to Drive `outputs/`).
- At the end: download `MyDrive/GenAI_A1/outputs/` into the repository's `outputs/` folder
  (gitignored) so the figures and tables can be placed in `report/figures/` and the LaTeX tables.
- Screenshots of the running application with the trained models (all four workspaces).
- The YouTube link of the demonstration video.
