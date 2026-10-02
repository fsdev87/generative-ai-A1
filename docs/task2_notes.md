# Task 2: corruption classifier and hard-routed specialist autoencoders

Implementation notes: what each script does, why it is built that way, the Optuna search spaces,
what the evaluation produces, and the exact commands. The brief is `GenAI_Assignment#1.pdf`
(Task 2, pages 4-5). Shared conventions: `docs/CONVENTIONS.md`.

## Files

| File | Purpose |
|---|---|
| `src/task2/common.py` | Shared helpers: config files, warm-up + cosine schedule, AMP, loaders, cached validation set, W&B runs that resume, classification metrics, confusion-matrix figure |
| `src/task2/train_classifier.py` | Classifier training (`train_classifier(config, trial=None)`), CLI `--config/--smoke/--resume` |
| `src/task2/optuna_classifier.py` | Study `task2_classifier` |
| `src/task2/evaluate_classifier.py` | Test/val metrics and error analysis of the classifier |
| `src/task2/train_specialist.py` | One specialist autoencoder (`SpecialistTrainer`, `train_specialist(config, type)`), CLI `--type --config --smoke --resume` |
| `src/task2/optuna_specialists.py` | Shared study `task2_specialists` (all three specialists per trial) |
| `src/task2/routing.py` | `HardRoutedRestorer` (predicted / oracle routing, identity bypass) |
| `src/task2/evaluate_routing.py` | Oracle vs predicted routing on the full test manifest, failure analysis |
| `src/task2/export_onnx.py` | `classifier.onnx` + `specialist_{salt,blur,occlusion}.onnx`, parity checks, model cards |
| `notebooks/task2_colab.ipynb` | Colab runner (top to bottom, resumable) |
| `tests/test_task2.py` | CPU tests: end-to-end smoke pipeline, balanced batches, routing, resume, ONNX parity |

## Pipeline and commands

Colab (each command is one notebook cell; `$OUTPUT_DIR` is set by `setup()`):

```bash
python -m src.task2.optuna_classifier --n-trials 30 --epochs 12
python -m src.task2.train_classifier --config "$OUTPUT_DIR/task2/task2_classifier_best_config.yaml" --resume
python -m src.task2.evaluate_classifier
python -m src.task2.optuna_specialists --n-trials 20 --epochs 6
python -m src.task2.train_specialist --type salt      --config "$OUTPUT_DIR/task2/task2_specialists_best_config.yaml" --resume
python -m src.task2.train_specialist --type blur      --config "$OUTPUT_DIR/task2/task2_specialists_best_config.yaml" --resume
python -m src.task2.train_specialist --type occlusion --config "$OUTPUT_DIR/task2/task2_specialists_best_config.yaml" --resume
python -m src.task2.evaluate_routing
python -m src.task2.export_onnx
```

Local smoke test (tiny synthetic data, tiny models, W&B disabled, CPU):

```bash
python -m src.task2.optuna_classifier --smoke
python -m src.task2.train_classifier --smoke
python -m src.task2.evaluate_classifier --smoke
python -m src.task2.optuna_specialists --smoke
python -m src.task2.train_specialist --type salt --smoke     # also blur, occlusion
python -m src.task2.evaluate_routing --smoke
python -m src.task2.export_onnx --smoke
python -m pytest tests/test_task2.py -q
```

Every Optuna and training script also accepts `--epochs`, `--num-workers`, and the studies
`--n-trials` and `--timeout-min` (stop starting new trials after a wall-clock budget).

### Disconnects and smoke safety

- Optuna studies continue after a disconnect (`create_study`/`run_study` from `src.common.optuna_utils`:
  the SQLite study is restored from Drive and only the remaining trials run; an interrupted trial
  is marked failed and replaced).
- Final training saves `last.pt` every epoch (model, optimiser, scheduler, GradScaler, early-stopping
  state, best-so-far metrics, W&B run id). `--resume` continues from it, using the configuration
  stored in the checkpoint so that a resumed run cannot silently change its schedule. A finished run
  is marked `finished` and returns immediately, so re-running a notebook cell is always safe. The W&B
  run is resumed under the same id (`wandb.init(id=..., resume="allow")`), so the curves stay in one run.
- The batch order of epoch `e` comes from an RNG seeded with `(seed, e)`, so it is the same with or without a resume.
- `--smoke` uses separate study names (`task2_classifier_smoke`, `task2_specialists_smoke`). `create_study(fresh=True)` deletes the study it opens, so this keeps a smoke run from deleting the real study. A smoke run also refuses to overwrite `last.pt`/`best.pt` written by a real run.

## Design decisions (with justification)

### Classifier

| Decision | Justification |
|---|---|
| Training data: `RuntimeCorruptionDataset` + `BalancedBatchSampler` | The brief requires runtime corruption and balanced batches. The sampler assigns the condition, so every batch holds exactly batch_size/4 images per class (tested) and the label is the one the corruption pipeline applied. Batch sizes are multiples of 4. |
| Validation: the fixed validation manifest (184 entries per class) | Deterministic and balanced by construction. The corrupted images are generated once per process (`materialize`) instead of every epoch. |
| Loss: plain multiclass cross-entropy | Required by the brief. No label smoothing: the probabilities are shown in the app and initialise the Task 3 gate, so they should be the model's own estimates. |
| Model: `CorruptionClassifier` (shared) | First stage at full resolution (salt pixels and mild blur would be destroyed by early downsampling). The head concatenates global average and global max pooling, because the cues are sparse. |
| AdamW, decay only on conv/linear weights | Decoupled weight decay (Loshchilov & Hutter, 2019). Weight decay is not applied to BatchNorm γ/β or biases ("no bias decay", He et al., 2019): decaying them does not regularise what the BN-normalised layers compute. |
| Schedule: linear warm-up (2 epochs, 1 in trials) then per-iteration cosine decay to 1% of the peak LR | Warm-up avoids unstable first updates at a high learning rate (Goyal et al., 2017). Cosine annealing (Loshchilov & Hutter, 2017) has no milestones or decay factors to tune. Each Optuna trial gets a complete schedule over its own budget, so trials compare fully annealed models. ReduceLROnPlateau was rejected: in a 12-epoch trial it rarely fires, so trials would be judged at a schedule different from the final run's. |
| Selection, early stopping and Optuna objective: validation **macro-F1** | Routing uses the argmax, so the decision metric matters, not the loss. Macro-F1 weighs the four classes equally and also penalises a class that is over-predicted (low precision), which accuracy on a balanced set would hide. Validation CE is logged but not used: it can rise through over-confidence while accuracy still improves (Guo et al., 2017). Early stopping (patience 15 of 60 epochs) is a safety net; with cosine decay the best epoch is usually near the end. |
| AMP (float16 autocast + GradScaler) on CUDA, channels-last, `cudnn.benchmark` | About 2x faster on the T4's tensor cores (Micikevicius et al., 2018). Validation and all evaluation run in float32, so metrics match the float32 ONNX models. `torch.compile` was not used: compiling once per Optuna trial would cost more than it saves. |
| Checkpoints: `best.pt` = `{model_config, model_state, ...}` with a CPU, NCHW-contiguous state dict | The contract Task 3 relies on (`build_model(CorruptionClassifier, path)`), independent of the channels-last training layout. |

### Specialists

| Decision | Justification |
|---|---|
| One `ConvAutoencoder` per corruption, trained only on `RuntimeCorruptionDataset(train, conditions=(type,))` | The brief requires that each specialist sees only its own corruption, with a fresh severity on every load and the clean image as target. The three runs are separate (different seeds `seed + class index`, separate optimisers, W&B groups and checkpoints), so their parameters are trained independently. |
| Pure bottleneck: `skip_resolutions = ()` (configurable, not searched) | The specialists are "denoising autoencoders". Task 1's rule that the compressed latent must be meaningful applies, so the latent is the only path from input to output. The bottleneck makes the network project the corrupted input onto clean-image structure (Vincent et al., 2008). Skip connections recover more detail (Mao et al., 2016) but also pass corrupted high-frequency content (salt pixels, rectangle edges) past the latent. They can be enabled in the YAML config to match Task 1 if Task 1 uses limited skips; the report should then justify it as the brief asks. |
| Bottleneck searched as the latent **shape** (8x8 or 16x16 x channels) | "Bottleneck size" is tuned through `latent_channels` and the latent resolution (`depth` 4 or 3). For restoration, a higher-resolution latent with fewer channels keeps more spatial layout than a low-resolution latent of the same size. Every preset compresses (3x-24x fewer values than the 49,152 inputs). |
| Loss `RestorationLoss(alpha)`; selection on `restoration_score = 0.5 SSIM + 0.5 PSNR/40` | The brief's L1/SSIM mix (Zhao et al., 2017 motivate mixing L1 with SSIM-type terms). The selection score does not depend on alpha, so trials with different alpha are comparable (shared convention). |
| AdamW, weight decay 1e-4 (fixed), no dropout | Light regularisation: runtime corruption already gives a new training pair every load. Dropout removes whole feature maps and injects noise into reconstructions. Neither is in the brief's list of hyperparameters to tune, so the search budget goes to those. |
| Warm-up 1 epoch + cosine, AMP, float32 validation, early stopping (patience 10 of 40) | As for the classifier. SSIM always runs in float32 (shared `ssim`). |
| W&B: per-epoch train/val losses, PSNR/SSIM overall and per severity, identity baseline, sample panels (target, input, output, error x4) at fixed validation entries every 5 epochs, `best.pt` artifact | Shared conventions. The identity baseline (input vs target) shows how much each specialist actually improves the input. |

### Shared specialist search

The brief allows one shared search for a common architecture. Each trial trains all three specialists
with the trial's hyperparameters, interleaved epoch by epoch:

```
for epoch e: train salt, blur, occlusion one epoch each -> validate each
             report mean(score_salt, score_blur, score_occlusion) at step e -> MedianPruner
objective = mean of the three best validation scores (per-type scores stored as user attributes)
```

Interleaving makes the reported value the running mean over all three types at every epoch. Training
the specialists one after another would let the pruner see only the salt score for the first third of
a trial, so a configuration that suits salt but fails occlusion would survive longer. After the study,
the three specialists are retrained independently with the selected configuration and the full schedule.

### Hard routing

`HardRoutedRestorer` computes `p = softmax(C(x))` and `r = argmax p` (predicted mode), or takes `r` from
the manifest label (oracle mode; the classifier still runs, so its prediction is recorded). It then
starts from `output = x.clone()` and overwrites only the samples with `r > 0` using
`A_r(x[r == k])`. Consequences:
- clean inputs never run an expert, and their output is bit-identical to the input (tested with `torch.equal`);
- each expert runs at most once per batch on its sub-batch, never per sample (tested);
- all models are in eval mode, so a sample's output does not depend on the rest of the batch (BatchNorm uses running statistics). Correctly routed entries give identical metrics in both modes (tested).

## Optuna search spaces

TPE sampler (seed 42), MedianPruner, SQLite study mirrored to Drive (`OPTUNA_DIR`). Every trial is its
own W&B run (groups `task2-classifier-optuna`, `task2-specialists-optuna`). After the study:
`OUTPUT_DIR/task2/optuna/<study>/` (trials.csv, summary.json, history, importances, parallel
coordinates, slice, intermediate values, `search_space.json`) and
`OUTPUT_DIR/task2/<study>_best_config.yaml` (best hyperparameters + the final-run schedule).

### `task2_classifier` (30 trials x 12 epochs; objective: best validation macro-F1)

| Parameter | Space | Why |
|---|---|---|
| learning rate | log-uniform [1e-4, 3e-3] | Typical AdamW range for small CNNs with BN; log scale because its effect is multiplicative |
| batch size | {32, 64, 128} | Multiples of 4 for exact balance; trades gradient noise against steps per epoch (92 / 46 / 23) |
| channel configuration | `16-32-64-128` (0.21 GMAC), `32-64-128-256` (0.84), `48-96-192-384` (1.89), `32-64-128-256-256` (0.92, one more stage) | Width vs depth/receptive field at roughly matched cost, as categorical presets of `channels` |
| convolutions per stage | {1, 2} | Depth per stage. The second full-resolution conv is the most expensive layer, and a smaller classifier is also a cheaper Task 3 gate |
| dropout (head) | uniform [0, 0.5] | Regularisation of the linear head |
| weight decay | log-uniform [1e-5, 1e-1] | Decoupled decay of conv/linear weights |

Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`, which never prunes during the first 3
epochs (warm-up and early noise). TPE `n_startup_trials=10` random trials before modelling.

### `task2_specialists` (20 trials x 6 epochs per specialist; objective: mean of the three best validation restoration_scores)

| Parameter | Space | Why |
|---|---|---|
| learning rate | log-uniform [1e-4, 3e-3] | As above |
| batch size | {16, 32, 64} | 184 / 92 / 46 steps per epoch |
| channel configuration (`base_channels`) | {16, 32, 48}; stages use base x (1, 2, 4, 8, 8) | 0.66 / 2.6 / 5.8 GMAC per image: capacity vs time |
| bottleneck (latent shape) | `8x8x32`, `8x8x64`, `8x8x128`, `16x16x16`, `16x16x32`, `16x16x64` (2,048-16,384 values; 24x-3x compression) | `latent_channels` and latent resolution (`depth` 4 or 3) |
| alpha (L1 weight) | uniform [0.4, 0.95] | From SSIM-dominated to L1-dominated around the brief's 0.8. With typical values (L1 about 0.03, 1-SSIM about 0.1), alpha about 0.8 balances the two terms |

Fixed: weight decay 1e-4, dropout 0, no skips, warm-up 1 epoch, cosine to 1%. Pruner:
`MedianPruner(n_startup_trials=5, n_warmup_steps=2)` on the running mean. TPE `n_startup_trials=8`.

Known limitation: short trials (low fidelity) favour configurations that learn fast. Rankings at
6 or 12 epochs correlate with, but are not identical to, rankings after the full schedule. This is the usual
trade-off of budgeted searches; the final runs use 40-60 epochs.

## What the evaluation produces

### `evaluate_classifier` (test manifest = 3,669 images x 10 variants; validation for reference)

- `tables/classifier_metrics.json`: accuracy, macro precision/recall/F1, per-class P/R/F1/support,
  raw and row-normalised confusion (test and val), plus the analysis rates below.
- `tables/classifier_{test,val}_per_class.csv/.tex`, `classifier_{test,val}_confusion.csv`,
  `figures/classifier_confusion_{test,val}.png`: normalised 4x4 confusion, rows = true class.
- `tables/classifier_test_by_type_level.csv/.tex`: accuracy and the share of each predicted class per type x severity.
- **Clean vs blur.** `tables/classifier_blur_strength.csv` and `figures/classifier_blur_detection.png` show
  P(pred = blur) and P(pred = clean) against the actual blur strength (`blur_strength`, the std of the
  applied truncated kernel). Test: the three fixed levels. Validation: training-range blurs in quantile
  bins. Strength 0: clean photos (the false-alarm rate). All with 95% Wilson intervals (Wilson, 1927).
  `tables/classifier_clean_sharpness.csv` and `figures/classifier_sharpness.png` measure the natural
  sharpness of every input as the variance of the Laplacian, a standard focus measure
  (Pech-Pacheco et al., 2000). They show how often clean photos in each sharpness quartile are called
  blurred, and how the sharpness of clean photos overlaps with mild blur. Expected finding: the
  mildest blurs (strength 0.46-0.65 px) look like naturally soft or low-resolution photos, so most
  clean↔blur errors come from the softest clean photos and the low blur level.
- **Clean vs occlusion and dark content.** `tables/classifier_dark_regions.csv` covers two directions:
  clean photos binned by `dark_fraction` (share of near-black pixels, all channels <= 0.1) with
  P(pred = occlusion), which tests whether black cats and dogs look like occlusion; and occlusion entries
  binned by `hidden_fraction` (share of the occluded area that was already near-black) with the
  detection rate, which tests whether a black rectangle on black fur goes unnoticed.
- `figures/classifier_misclassified.png`: the most confident errors, round-robin over (true, predicted) pairs, with probability bars.
- `tables/classifier_test_predictions.csv`: one row per test entry (probabilities, blur_strength,
  sharpness, dark_fraction, hidden_fraction) for further analysis.

### `evaluate_routing` (full test manifest, both modes)

- `eval/test_records_{oracle,predicted}.csv` from `evaluate_restoration`: standard columns plus
  `true_label, pred_label, pred_class, classifier_correct, route, expert, prob_*`.
- `tables/routing_{mode}_{by_type,by_type_level,by_level}.csv/.tex`: the standard tables, plus
  `n_exact` (outputs identical to the target) and `psnr_finite` (mean PSNR over the other outputs),
  and a `corrupted` row (mean over the three corruptions).
- `tables/routing_oracle_vs_predicted.csv/.tex`: per type x severity (and per type, and all corrupted):
  routing accuracy, PSNR/SSIM in both modes, deltas, exact counts, misrouted/harmful/beneficial counts.
- `tables/routing_misrouting.csv/.tex`: the failure analysis, one row per (true, routed-to) confusion:
  count, share of the true type, counts per severity, mean PSNR/SSIM change, and how many were
  harmful (predicted routing at least 1 dB worse than oracle), neutral, or beneficial (at least 1 dB better).
  Beneficial misroutings are possible. For example, a mild blur routed to the identity can score
  higher than the blur specialist's reconstruction if the bottleneck loses more detail than the blur did.
  This is reported, not hidden.
- `figures/routing_failures.png`: worst harmful cases per confusion pair, showing target, input, oracle
  output, predicted output and the classifier probabilities. `figures/routing_examples_{mode}.png` shows
  representative entries (median SSIM per type x severity) as target, input, output and error.
- `eval/summary.json`: routing accuracy, outcome counts, and per-type metrics in both modes. The oracle
  per-type metrics are each specialist's results on its own corruption and are read by `export_onnx`.

**Clean inputs and the PSNR cap.** With oracle routing every clean input passes through the identity
bypass: output = target exactly, MSE 0, SSIM 1, PSNR capped at 100 dB. These are reported as
"exact (n/n)" via `n_exact`, and as SSIM, rather than averaged as 100 dB. In predicted mode the share of
clean inputs that still pass through exactly equals the classifier's clean recall. Misrouted clean
photos get a finite PSNR, reported in `psnr_finite`. The standard `psnr` column is kept for
cross-task comparability. The "all" row mixes capped and finite values, so use the `corrupted` row
when comparing restoration quality.

### `export_onnx`

`ONNX_DIR/classifier.onnx` (`input` -> `probs`, a Softmax wrapper, CLASSES order) and
`ONNX_DIR/specialist_{type}.onnx` (`input` -> `output`). Opset 17, float32, dynamic batch. Parity is
checked against PyTorch on 64 real test inputs spread over the test manifest (for a specialist, over
its own corruption); max |diff| must be at most 1e-4. On failure the file is deleted and the script
stops. Model cards (`.json`) record: config, I/O spec, preprocessing, the routing rule, test metrics
(classifier: accuracy, macro P/R/F1, per class; specialist: PSNR/SSIM/MSE on its own corruption, per
severity), training hyperparameters, best epoch, parity result and W&B run URL.

## GPU budget (free Colab T4, **estimates**)

Measured on CPU: the corruption pipeline costs 0.2-0.5 ms per sample (about 1-2 s per 2,944-image epoch
on one worker, so data loading is not the bottleneck). Model cost per image (forward): classifier
0.21-1.89 GMAC; autoencoder 0.66 (base 16) / 2.6 (base 32) / 5.8 (base 48) GMAC. Assuming about
5-6 TFLOPS effective on a T4 with AMP gives these estimates (actual times are printed and logged per epoch):

| Step | Default | Estimate |
|---|---|---|
| Classifier Optuna | 30 trials x 12 epochs (about 3-5 s per epoch), pruning | 30-40 min |
| Final classifier | 60 epochs | 5-15 min |
| Classifier evaluation | test + val | 2-4 min |
| Specialist Optuna | 20 trials x 3 specialists x 6 epochs (about 3-20 s per epoch depending on width) | 55-75 min |
| Three final specialists | 40 epochs each | 5-15 min each |
| Routing evaluation | 2 x 36,690 entries | 4-8 min |
| ONNX export | 4 models | 1-2 min |

If the T4 turns out slower than estimated, use `--timeout-min` on the studies or fewer `--n-trials`.

## References (checked)

- E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *JASA* 22(158), 1927.
- J. L. Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study," *ICPR* 2000 (variance of the Laplacian as a focus measure).
- P. Vincent et al., "Extracting and composing robust features with denoising autoencoders," *ICML* 2008.
- X.-J. Mao, C. Shen, Y.-B. Yang, "Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections," *NIPS* 2016 (arXiv:1603.09056).
- I. Loshchilov, F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," *ICLR* 2017.
- I. Loshchilov, F. Hutter, "Decoupled weight decay regularization," *ICLR* 2019.
- P. Goyal et al., "Accurate, large minibatch SGD: training ImageNet in 1 hour," arXiv:1706.02677, 2017 (gradual warm-up).
- T. He et al., "Bag of tricks for image classification with convolutional neural networks," *CVPR* 2019 (no bias decay).
- H. Zhao, O. Gallo, I. Frosio, J. Kautz, "Loss functions for image restoration with neural networks," *IEEE TCI* 3(1), 2017.
- P. Micikevicius et al., "Mixed precision training," *ICLR* 2018.
- C. Guo et al., "On calibration of modern neural networks," *ICML* 2017.
- T. Akiba et al., "Optuna: a next-generation hyperparameter optimization framework," *KDD* 2019; J. Bergstra et al., "Algorithms for hyper-parameter optimization," *NeurIPS* 2011 (TPE).
- Z. Wang et al., "Image quality assessment: from error visibility to structural similarity," *IEEE TIP* 2004 (SSIM).
