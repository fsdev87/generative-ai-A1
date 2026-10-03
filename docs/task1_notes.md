# Task 1 — Universal Multi-Corruption Denoising Autoencoder (UDAE): design notes

One convolutional autoencoder restores 128x128 pet images that are clean, salt-and-pepper
corrupted, Gaussian-blurred or occluded, **without being told which corruption was applied**.
This document explains every design decision (so it can be defended), what the scripts
produce, and the exact commands. Shared conventions: `docs/CONVENTIONS.md`.

| File | Purpose |
|---|---|
| `src/task1/config.py` | Default hyperparameters, YAML/CLI merging, variant names, output folders, bottleneck size |
| `src/task1/train.py` | `train_model()` (shared with Optuna) + CLI for the final run and the ablation variants |
| `src/task1/optuna_search.py` | Optuna study `task1_udae`, report, `best_config.yaml` |
| `src/task1/baselines.py` | Identity and classical *oracle* baselines (median / unsharp mask / Telea) |
| `src/task1/evaluate.py` | Test-set evaluation, tables, figures, diagnostics, ablation comparison |
| `src/task1/export_onnx.py` | `best.pt` -> `ONNX_DIR/udae.onnx` + parity check + model card |
| `notebooks/task1_colab.ipynb` | Colab runner (everything resumable) |
| `tests/test_task1.py` | CPU smoke/unit tests |

## 1. Data pipeline (shared foundation, used unchanged)

- Training: `RuntimeCorruptionDataset(images["train"], conditions=CLASSES)` draws a **new**
  condition (clean / salt / blur / occlusion, uniform, as the brief requires) and severity
  every time an image is loaded, plus a random horizontal flip. Nothing corrupted is stored.
  2,944 training images; `drop_last=True` keeps every batch the same size (BatchNorm).
- Validation: the fixed val manifest (736 entries, balanced over the four conditions,
  severities from the training ranges). Because it is deterministic, the corrupted
  validation inputs are computed **once** per process and kept as tensors (`load_data`);
  the Optuna search shares one copy across all trials instead of re-corrupting 736 images
  every epoch of every trial.
- Test: the official manifest, 3,669 images x 10 variants (clean + 3 corruptions x 3 fixed
  severities) = 36,690 entries, streamed through a DataLoader (too large to keep in memory).

## 2. Architecture and bottleneck

`src.models.autoencoder.ConvAutoencoder` (shared with Tasks 2-3): a stem of two 3x3
conv-BN-ReLU layers, then `depth` stride-2 blocks that halve the resolution and double the
channels (capped at 8 x `base_channels`), a 1x1 convolution to `latent_channels`, and a
mirrored decoder using bilinear upsampling + convolution (avoids the checkerboard artefacts
of transposed convolutions, Odena et al. 2016) ending in a sigmoid (outputs in [0, 1]).

**Depth is fixed at 4 (128 -> 8x8 latent).** Measured receptive fields (gradient of one
latent cell / one output pixel w.r.t. the input):

| depth | latent | one latent cell sees | one output pixel sees |
|---|---|---|---|
| 3 | 16x16 | 47 px | 87 px |
| **4** | **8x8** | **95 px** | **175 px (the whole image)** |

The largest test occlusion is a single ~76x76 box (35% of the image). To fill the centre
of that hole the network must see past its border on both sides; with depth 4 every
output pixel sees the entire 128x128 image, with depth 3 it barely does.

**Bottleneck dimension.** The latent is `latent_channels x 8 x 8`, the only path from
input to output in the main model (verified by `tests/test_models.py::test_autoencoder_is_a_real_bottleneck`).
The input has 3 x 128 x 128 = 49,152 values:

| latent_channels | latent dimension | compression ratio |
|---|---|---|
| 8 | 512 | 96x |
| 16 | 1,024 | 48x |
| 32 | 2,048 | 24x |
| 64 | 4,096 | 12x |

The search range stops at 64 channels (12x compression). Wider latents would score
higher on PSNR (more capacity always helps a pixel metric), but the brief requires a
*meaningful* bottleneck; 12x is still a strong compression of the image.

**Encoder width** (`base_channels`, channels per level = base x [1, 2, 4, 8, 8]):

| base_channels | parameters (latent 32) | forward cost per 128x128 image |
|---|---|---|
| 16 | 1.04 M | 0.66 GMAC |
| 32 | 4.13 M | 2.60 GMAC |
| 48 | 9.28 M | 5.82 GMAC |
| 64 | 16.49 M | 10.33 GMAC |

## 3. No skip connections, and the skip-connection ablation

The main model has **no skip connections**: unrestricted skips (U-Net, Ronneberger et al.
2015) would let the decoder copy the input around the bottleneck, which the brief rules out,
and for this task they are actively harmful because the input *is* the corruption: a
full-resolution skip hands salt-and-pepper pixels and black occlusion boxes straight to
the decoder. Skips are popular in restoration (e.g. RED-Net, Mao et al. 2016) because they
pass high-frequency detail that a bottleneck loses, so their effect is measured instead of assumed.

**Ablation.** Three variants add *one* limited skip (encoder features concatenated into
the decoder at that resolution), with the same hyperparameters and epochs as the main model:

| variant | skip at | values carried per image (base 32) | relative to the input |
|---|---|---|---|
| `udae` (main) | — | 0 | 0 |
| `udae_skip16` | 16x16 | 256 x 16 x 16 = 65,536 | 1.33x |
| `udae_skip32` | 32x32 | 128 x 32 x 32 = 131,072 | 2.67x |
| `udae_skip128` | 128x128 | 32 x 128 x 128 = 524,288 | 10.7x |

Even the 16x16 skip carries more numbers than the input itself, so a "limited" skip is
limited in *resolution*, not capacity. Deep features have passed through many nonlinear
layers (and see a 47-95 px neighbourhood), so they carry context rather than raw pixels; a
128x128 skip is only two convolutions away from the raw input. Expected trade-off, which
the evaluation quantifies: sharper outputs and higher PSNR on clean and mildly corrupted
inputs (detail bypasses the bottleneck), versus corruption leaking through (impulses and
box edges copied to the output).

Two diagnostics (computed by `evaluate.py`, saved in `eval/diagnostics.json`):
- **Impulse survival** (salt-and-pepper entries): among pixels hit by an impulse (input
  pure black/white in all channels and > 0.25 away from the clean value), the fraction
  where the output is still closer to the impulse than to the clean value. Identity = 1,
  perfect restoration = 0 (the tests check identity = 1 and a constant grey output = 0).
- **Detail ratio** (clean entries): mean |Laplacian| of the output's luminance divided
  by that of the target. Below 1 means the output is smoother than the clean image
  (detail lost in the bottleneck); identity = 1.

`evaluate.py --compare` puts the variants side by side (`tables/skip_ablation.{csv,tex}`:
parameters, skip capacity, PSNR per condition, pooled SSIM, impulse survival, detail
ratio) and shows the same test entries (clean, salt high, blur medium, occlusion high)
restored by every variant (`figures/skip_ablation_examples.png`).

## 4. Loss

`RestorationLoss(alpha) = alpha * L1 + (1 - alpha) * (1 - SSIM)`, as in the brief (SSIM:
Wang et al. 2004, 11x11 Gaussian window, computed in float32 even under mixed precision).
L1 keeps colours and brightness right; SSIM rewards local structure and contrast. Zhao et
al. (2017) found that this kind of mix (they used MS-SSIM) beats either term alone, and that
SSIM-type losses alone allow colour and brightness shifts because they are insensitive to
uniform biases. Note that their alpha weights the MS-SSIM term
(alpha = 0.84, chosen so that both terms contribute roughly equally), whereas the brief's
alpha weights L1 — their setting corresponds to alpha ≈ 0.16 here. The two terms also have
different scales (L1 ≈ 0.02-0.05 vs 1 - SSIM ≈ 0.05-0.2 for good restorations), so the
best alpha is an empirical question: the search covers 0.1-0.95, which includes both the
brief's 0.8 and Zhao et al.'s balance point.

## 5. Optimisation

- **AdamW** (Loshchilov & Hutter 2019): Adam's per-parameter step sizes converge quickly on
  conv autoencoders without tuning a momentum schedule; AdamW applies weight decay
  directly to the weights instead of mixing it into the adaptive gradient, where Adam would
  rescale it. Weight decay is fixed at 1e-4 (a mild regulariser; overfitting is not the
  main risk because every epoch sees new corruptions) to keep the search six-dimensional.
- **Schedule: linear warm-up (1 epoch) then cosine decay to 1% of the peak LR, per step.**
  Warm-up avoids large early Adam updates while its second-moment estimates and the
  BatchNorm statistics are still noisy (Goyal et al. 2017). Cosine annealing (Loshchilov
  & Hutter 2017) decays smoothly and needs no validation-based trigger. Importantly, every
  Optuna trial anneals fully within its own short budget, so trials are compared at the end
  of a complete schedule rather than mid-training at a high learning rate.
- **Mixed precision on CUDA** (fp16 autocast + GradScaler, Micikevicius et al. 2018). The
  T4 has fp16 tensor cores but no bf16 support, so fp16 with loss scaling is the right
  choice; SSIM runs in fp32 (its variance terms lose all precision in fp16). Gradients are
  unscaled before clipping, and steps whose fp16 gradients overflowed are skipped by the scaler.
- **Gradient clipping** at norm 1.0, a safety net against rare fp16 or high-LR spikes;
  the mean gradient norm per epoch is logged (`train/grad_norm`) to check how often it
  matters. With Adam a uniform rescaling of a gradient barely changes the update, so
  clipping does not slow normal training.
- **Efficiency:** `cudnn.benchmark` (fixed input size), pinned memory + non-blocking copies,
  2 DataLoader workers with persistent workers on CUDA (0 on Windows/CPU), loss/metric sums
  kept on the GPU and read once per epoch (no per-step host synchronisation), validation
  metrics computed on the GPU in one pass.
- **Early stopping** on the validation score with patience 12 epochs (a safety net: with
  cosine decay the score usually improves until the end). Disabled inside Optuna trials,
  where the pruner decides.
- **Diverged runs:** a non-finite epoch loss or validation score raises
  `NonFiniteLossError` (in Optuna: trial pruned with reason `non_finite_loss`).

## 6. Validation, checkpoints, tracking

- Every epoch: mean loss, PSNR, SSIM and `restoration_score = 0.5 * SSIM + 0.5 * PSNR / 40`
  on the 736 validation entries, overall and per condition. Validation runs under the
  same autocast as training (it runs in every epoch of every trial); test metrics and the
  ONNX export use float32.
- `CKPT_DIR/task1/<variant>/last.pt` every epoch: model, optimizer, scheduler, GradScaler,
  early-stopping state, epoch, per-epoch history, best epoch/metrics, config, W&B run id.
  `best.pt` (model + config + val metrics) whenever the score improves; it is written
  *before* `last.pt`, so a crash between the two never leaves `last.pt` ahead of `best.pt`.
  Writes are atomic (`save_checkpoint`).
- **Resume** (`--resume`): continues from `last.pt` at the next epoch with all states
  restored (the LR schedule and Adam step counts continue; verified by
  `test_resume_continues_at_the_next_epoch`). The checkpoint's configuration is
  authoritative (changed CLI values are reported and ignored). A finished run returns
  immediately, so re-running a notebook cell is safe. The W&B run is resumed too: the
  run id is stored in the checkpoint and passed via `WANDB_RUN_ID`/`WANDB_RESUME=allow`.
  The random stream is re-seeded with `seed + completed epochs`, so a resumed run does not
  replay the first epochs' corruptions.
- **W&B:** one `run.log` per epoch with `epoch` as x-axis (`train/*`, `val/*` incl.
  per-condition metrics, `lr`, `time/*` throughput); every 5 epochs (and the first/last)
  an image grid input | output | target for the **same** 8 validation entries (the first
  2 of each condition); at the end `best.pt` is uploaded as model artifact `task1-<variant>`.
  Groups: `task1-final` (final + ablation runs, tagged `main` / `ablation`), `task1-optuna`.

## 7. Optuna search

Study `task1_udae` (TPE sampler, seed 42, 10 random start-up trials, SQLite snapshot copied
to Drive after every trial, so the search survives disconnects; `run_study` only runs the
remaining trials). Objective: **the best validation restoration_score of the trial**
(maximise). It does not depend on alpha, so trials with different loss weights are
compared on the same scale; it rewards both structure (SSIM) and pixel fidelity (PSNR).

| Parameter | Range | Scale | Why this range |
|---|---|---|---|
| `lr` | 1e-4 – 3e-3 | log-uniform | brackets Adam's usual 1e-3 by ~1 decade below and ~0.5 above; above ~3e-3 BN conv nets tend to become unstable, below 1e-4 a 15-epoch trial cannot converge |
| `batch_size` | 16, 32, 64 | categorical | 184 / 92 / 46 updates per epoch on 2,944 images; 128 would leave only 23 updates per epoch (≈ 345 per trial), too few in a fixed epoch budget; < 16 gives noisy BatchNorm statistics |
| `latent_channels` (bottleneck) | 8, 16, 32, 64 | categorical | latent 512 – 4,096 values = 96x – 12x compression (§2); capped to keep a meaningful bottleneck |
| `base_channels` (encoder width) | 16, 32, 48, 64 | categorical | 1 M – 16.5 M parameters, 0.7 – 10.3 GMAC per image; wider than 64 would make trials too slow for the T4 budget and the app's CPU inference |
| `dropout` | 0.0 – 0.3 | uniform | spatial dropout (Dropout2d, Tompson et al. 2015) after every conv block; low values because the runtime corruption is already a strong regulariser and dropout in a restoration decoder adds noise to the output; the result shows whether it helps at all |
| `alpha` | 0.1 – 0.95 | uniform | from SSIM-dominated to L1-dominated; contains the brief's 0.8 and Zhao et al.'s balance point (≈ 0.16 in this parameterisation), §4 |

Categorical powers of two/multiples of 16 are used for channel counts and batch sizes
(standard, tensor-core friendly sizes; TPE handles small categorical sets well).

- **Reference trial:** trial 0 is enqueued with the brief's starting point (alpha 0.8, lr
  1e-3, batch 32, latent 32, base 32, dropout 0), so the report can state how much the search
  gained over it. It is re-enqueued only if a previous attempt failed (disconnect).
- **Budget:** 15 epochs per trial (a quarter of the 60-epoch final run), 40 trials, at most
  90 minutes per session (`--timeout-min`). Short trials are standard practice (as in
  successive halving); the schedule anneals completely within a trial (§5), which makes the
  short-budget ranking more reliable. Caveat for the report: settings that pay off only
  late (lower LR, dropout) can be under-rated by short trials.
- **Pruning:** `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`: no pruning during the
  first 5 trials or the first 2 epochs of a trial (warm-up and the first high-LR epochs are
  noisy); from epoch 3 a trial stops if its best score so far is below the median of earlier
  trials at the same epoch. Pruned trials record `pruned_reason` (`pruner`, `cuda_oom`,
  `non_finite_loss`) and their W&B runs are always finished.
- **CUDA out of memory** (very unlikely in this space: the largest configuration needs a
  few GB at batch 64 on a 15 GB T4): the exception is caught, memory released
  (`gc.collect()` + `torch.cuda.empty_cache()` after leaving the `except` block, so the
  traceback no longer holds tensors), and the trial is pruned with reason `cuda_oom`.
- **Outputs:** `OUTPUT_DIR/task1/optuna/task1_udae/` (`trials.csv` incl. user attributes:
  latent dimension, compression ratio, epochs run, best validation metrics per condition,
  pruning reason; `summary.json`; `search_space.json`; optimisation history, parameter
  importances, parallel coordinates, slice and intermediate-value plots) and
  `OUTPUT_DIR/task1/best_config.yaml` (best trial's parameters + the fixed settings, with the
  latent dimension and compression ratio in the header; `epochs` is deliberately left out so
  the final run uses its own 60).

## 8. Evaluation (`evaluate.py`)

All metrics on the official test manifest in float32. Main model -> `OUTPUT_DIR/task1/`,
ablation variants -> `OUTPUT_DIR/task1/variants/<variant>/`:

- `eval/test_records.csv`: one row per test entry (entry, image, type, level, PSNR, SSIM,
  MSE) — the file Task 3's cross-task comparison reads.
- `tables/by_type`, `by_type_level`, `by_level` (`.csv` + `.tex`): results per condition
  (clean / salt / blur / occlusion), per condition x severity, and per severity.
- `tables/comparison_by_type`, `comparison_by_type_level`: the UDAE next to
  (a) **identity** — the corrupted input itself (how bad each corruption is), and
  (b) **classical oracle baselines that are told the corruption**, i.e. use information the
  UDAE never gets: 3x3 median filter for salt-and-pepper; unsharp masking with the *known*
  blur kernel and amount 1, x̂ = y + (y − h∗y), which is one iteration of Van Cittert's
  deconvolution (1931) — the error spectrum (1 − H)X becomes (1 − H)²X, smaller wherever
  the kernel's frequency response is between 0 and 1 — followed by clipping to [0, 1]; and
  OpenCV Telea inpainting (Telea 2004) of the *known* occlusion mask. Both baselines return
  clean inputs unchanged (exact; PSNR = ∞, stored as the 100 dB cap in the CSV and shown as
  ∞ in LaTeX), so the pooled row ("corrupted") covers the three corruptions only.
  Baseline records are computed once and cached in `OUTPUT_DIR/task1/eval/baselines/`.
- `figures/representative_examples_{1,2}.png`: 12 typical examples (3 clean + one per
  corruption x severity), each the entry closest to its group's median SSIM (not
  cherry-picked), as Target | Input | Output | |Error| rows.
- `figures/failure_cases.png`: six different failure modes — the lowest-SSIM output for
  each input condition (clean: detail lost in the bottleneck; salt: residual noise / texture
  confusion; blur: unrecoverable detail; occlusion: implausible fill), plus the two corrupted
  entries where the model gains least PSNR over (or even loses to) its own input. Row labels:
  `mode: type severity` and `PSNR dB (input PSNR) / SSIM`.
- `figures/severity_curves.png`: PSNR and SSIM vs severity per corruption type for the UDAE,
  the oracle classical baseline and the corrupted input, with the UDAE's clean-input score
  as a reference line.
- `figures/training_curves.png`: train/validation loss and per-condition validation
  PSNR/SSIM per epoch (from `last.pt`, best epoch marked).
- `eval/diagnostics.json` (impulse survival, detail ratio; §3) and `eval/summary.json`
  (all headline numbers, model/bottleneck info, selected example entries, W&B URL).

## 9. ONNX export (`export_onnx.py`)

`best.pt` -> `ONNX_DIR/udae.onnx`: opset 17, float32, input `input` [N,3,128,128] and output
`output` [N,3,128,128] in [0,1], dynamic batch. Parity: ONNX Runtime vs PyTorch (eval, CPU) on
64 real test inputs evenly spread over the test manifest (all conditions and severities) as one
batch, and on a single image (the app's case); max |difference| must be ≤ 1e-4, otherwise
the file is renamed to `udae.onnx.failed` and the script raises. The model card `udae.json`
contains model config, latent shape/dimension/compression, I/O spec, preprocessing ("RGB,
resized to 128x128, float32 in [0,1], NCHW"), test metrics from `eval/summary.json` (if the
evaluation has run), validation metrics, training hyperparameters, parity result, W&B run URL,
git commit and export time.

## 10. Commands

Colab (after the two setup cells of `notebooks/task1_colab.ipynb`; `$OUTPUT_DIR` is set by `setup()`):

```bash
python -m src.task1.train --smoke                                   # 1-minute sanity check
python -m src.task1.optuna_search --n-trials 40 --epochs 15 --timeout-min 90
python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --resume
python -m src.task1.evaluate
python -m src.task1.export_onnx
# skip-connection ablation (same config and epochs; own folders, never overwrite udae)
python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --skip-resolutions 16 --resume
python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --skip-resolutions 32 --resume
python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --skip-resolutions 128 --resume
python -m src.task1.evaluate --variant udae_skip16
python -m src.task1.evaluate --variant udae_skip32
python -m src.task1.evaluate --variant udae_skip128
python -m src.task1.evaluate --compare udae udae_skip16 udae_skip32 udae_skip128
```

Every hyperparameter is also a flag (`python -m src.task1.train --help`), overriding the YAML.
`--output-subdir` / `--run-name` / `--wandb-group` name further variants. Every script
accepts `--smoke` (synthetic data, W&B disabled, outputs under a separate `smoke/`
subfolder so real results are never overwritten). Tests: `python -m pytest tests/test_task1.py -q`.

## 11. Runtime budget (estimates, not measured on a T4)

Forward cost is 2.6 GMAC per image at base 32 (10.3 at base 64), so one training epoch is
≈ 46 TFLOP (≈ 180 at base 64). Assuming 10-15 effective TFLOPS with fp16 on a T4, plus a
data-loading floor of ≈ 2-3 s per epoch (runtime corruption measured at 0.9 ms per image
on a desktop CPU; Colab has 2 slower vCPUs), an epoch should take ≈ 5-8 s at base 32 and
≈ 12-20 s at base 64, plus up to a few seconds to write `last.pt` to Drive.

| Step | Estimate |
|---|---|
| Optuna, 40 trials x 15 epochs, about half pruned around epoch 3-6 | 60-90 min (capped at 90 per session) |
| Final run, 60 epochs | 6-20 min depending on the selected width |
| Test evaluation (first run computes the cached baselines) | ≈ 5 min, ≈ 2-3 min per variant |
| ONNX export | ≈ 1 min |
| Ablation, 3 runs x 60 epochs | 20-60 min |

Total ≈ 2-2.5 h. The real epoch time is printed in every log line and logged to W&B
(`time/epoch_s`); if needed, reduce `--n-trials` or the ablation epochs — the ablation runs
must all use the same epochs as the main model for a fair comparison.

## 12. Limitations and points to check after the Colab run

- The GPU throughput above is estimated from FLOP counts; only CPU smoke runs were executed
  here. The CUDA-only paths (fp16 autocast, GradScaler, pinned memory, workers) follow the
  standard PyTorch recipes but run for the first time on Colab — the notebook's smoke cell
  exercises them in about a minute.
- Hyperparameters chosen with 15-epoch trials are applied to a 60-epoch run; the best LR for
  a 4x longer schedule can be slightly lower.
- Validation metrics are computed under fp16 autocast (§6); the reported test metrics are
  float32.
- `restoration_score` averages per-image PSNR; on clean inputs the identity baselines are
  exact (∞ dB) and are shown as such rather than averaged.

## References

- Vincent, Larochelle, Bengio, Manzagol. Extracting and composing robust features with denoising autoencoders. ICML 2008.
- Wang, Bovik, Sheikh, Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP 13(4), 2004.
- Zhao, Gallo, Frosio, Kautz. Loss functions for image restoration with neural networks. IEEE Trans. Computational Imaging 3(1), 2017 (Eq. 14, alpha = 0.84 on the MS-SSIM term).
- Ronneberger, Fischer, Brox. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.
- Mao, Shen, Yang. Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections. NeurIPS 2016.
- Odena, Dumoulin, Olah. Deconvolution and checkerboard artifacts. Distill, 2016.
- Tompson, Goroshin, Jain, LeCun, Bregler. Efficient object localization using convolutional networks. CVPR 2015 (spatial dropout).
- Loshchilov, Hutter. Decoupled weight decay regularization. ICLR 2019 (AdamW).
- Loshchilov, Hutter. SGDR: stochastic gradient descent with warm restarts. ICLR 2017 (cosine annealing).
- Goyal et al. Accurate, large minibatch SGD: training ImageNet in 1 hour. arXiv:1706.02677, 2017 (warm-up).
- Micikevicius et al. Mixed precision training. ICLR 2018.
- Bergstra, Bardenet, Bengio, Kégl. Algorithms for hyper-parameter optimization. NeurIPS 2011 (TPE).
- Akiba, Sano, Yanase, Ohta, Koyama. Optuna: a next-generation hyperparameter optimization framework. KDD 2019.
- Telea. An image inpainting technique based on the fast marching method. Journal of Graphics Tools 9(1), 2004.
- van Cittert. Zum Einfluß der Spaltbreite auf die Intensitätsverteilung in Spektrallinien. II. Zeitschrift für Physik 69, 298–308, 1931.
