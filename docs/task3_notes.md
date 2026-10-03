# Task 3 - Jointly trained soft mixture-of-experts restoration: design notes

Everything in `src/task3/`, the Colab runner `notebooks/task3_colab.ipynb` and the tests in
`tests/test_task3.py`. This file records *why* each decision was made, so every one of them can be
defended in the live evaluation.

```
w     = softmax(G(x) / tau)                       G = the Task 2 corruption classifier (the gate)
x_hat = w0 * x + w1 * A_salt(x) + w2 * A_blur(x) + w3 * A_occlusion(x)
L     = l1 * L1 + ls * (1 - SSIM) + lc * CE + lb * L_balance
```

Branch order is `CLASSES = (clean, salt, blur, occlusion)` everywhere: branch 0 is the identity,
branches 1-3 are the Task 2 specialists.

## Files

| File | Purpose |
|---|---|
| `model.py` | `SoftMoERestorer`, `from_task2`, the stand-in checkpoint helper |
| `routing.py` | balance loss, full MoE loss, routing statistics, collapse criterion, hard routing |
| `train.py` | the two-stage training loop (`train_moe`), resumable, W&B logging, CLI |
| `optuna_search.py` | study `task3_moe`, pruning on poor score and on routing collapse |
| `evaluate.py` | test-set restoration metrics **and** the gating analysis the brief requires |
| `compare.py` | universal AE vs hard routing (oracle / predicted) vs soft MoE |
| `make_mixed_manifest.py`, `evaluate_mixed.py` | the optional two-corruption experiment (extra) |
| `export_onnx.py` | the whole pipeline as one ONNX graph + parity check + model card |
| `plots.py`, `config.py` | figures and configuration/paths |

## Model

**Initialisation (required by the brief).** `SoftMoERestorer.from_task2(ckpt_dir, tau)` loads
`CKPT_DIR/task2/classifier/best.pt` as the gate and the three `specialist_*/best.pt` as the experts.
Each sub-model is rebuilt from **its own** `model_config` stored in its checkpoint, so whatever
architecture Task 2's Optuna search selected is used unchanged (for example specialists with a 16x16
latent, i.e. `depth=3`, or different widths per expert). Nothing is assumed about those configs; a
missing checkpoint raises a `FileNotFoundError` naming the files and telling the user to train Task 2
first. Starting from random components would throw away the Task 2 training and, worse, the gate
would initially route at chance level, so the experts would be trained on inputs they do not match -
exactly the self-reinforcing mis-assignment that MoE work warns about.

**The identity branch is a real identity**, not a learned "clean expert": `branch_outputs[:, 0] = x`.
A clean input therefore only needs the gate to put its weight on branch 0 to be reproduced perfectly;
there is no reconstruction floor for clean images, as there would be with a fourth autoencoder.

**Temperature `tau` is a registered buffer**, not a parameter and not a Python float. As a buffer it
(a) is saved in the `state_dict` and travels with the checkpoint, (b) is never updated by the
optimiser, and (c) is traced into the exported ONNX graph as a constant, so the deployed model is
guaranteed to use the temperature it was trained with. Temperature-scaled softmax follows Hinton et
al. (2015), *Distilling the Knowledge in a Neural Network*, §2: `q_i = exp(z_i/T) / sum_j exp(z_j/T)`,
where "using a higher value for T produces a softer probability distribution over classes". Here a
small `tau` approaches hard routing (Task 2) and a large `tau` approaches uniform blending, so `tau`
is the one knob that interpolates between Tasks 2 and 3 - which is why it is searched by Optuna.

**Signatures.** `forward(x)` returns exactly `(output, weights, branch_outputs)` with shapes
`[N,3,128,128]`, `[N,4]`, `[N,4,3,128,128]` - the ONNX contract in `docs/CONVENTIONS.md`. The gate
logits needed for the cross-entropy term are returned by a **separate method**,
`forward_with_logits(x)`, rather than by a boolean flag on `forward`: a `if flag:` branch inside
`forward` makes the TorchScript exporter emit a `TracerWarning` (the flag is baked in as a constant),
and keeping the exported signature free of optional arguments means the graph cannot accidentally be
exported with the wrong output set.

**Softmax and gate logits are computed in float32** even under AMP (`.float()` in `route`). The
softmax of fp16 logits and the cross-entropy on them lose precision exactly where the routing
decision is made, and the resulting weights multiply *every* branch output.

## Staged training

The brief prescribes the two stages; the details below are the choices made inside them.

**Stage 1, warm-up (2 epochs, gate only, `lr_warmup = 3e-4`).** The experts are frozen in two senses
at once, and the tests check both:

1. `requires_grad = False` for every expert parameter, and the warm-up optimiser is constructed over
   `model.gate.parameters()` only - so no expert tensor can be touched even by weight decay or
   optimiser state.
2. The experts are kept in **eval mode**, and `SoftMoERestorer.train()` is overridden so that a later
   `model.train()` call cannot silently undo it. Without this, `requires_grad=False` alone would still
   let BatchNorm update its running mean/variance on every forward pass and dropout stay active: the
   experts would drift although "frozen", and the warm-up would no longer be a clean measurement of
   the gate.

The purpose of the warm-up is that the gate must first adapt to its new job. In Task 2 it was trained
as a classifier with cross-entropy only; here its output is multiplied into a reconstruction. Letting
the experts move while the gate is still mis-routing would train each expert on inputs of the wrong
type. This is the standard argument for staged unfreezing in transfer learning (Howard and Ruder,
2018, *ULMFiT*, §3.3: "Rather than fine-tuning all layers at once, which risks catastrophic
forgetting, we propose to gradually unfreeze the model").

**Stage 2, joint fine-tuning (up to 20 epochs).** All parameters train, with two optimiser groups:
the gate at `lr` and the experts at `lr * expert_lr_scale` (default 0.1). Justification:

- The experts are already good at their own corruption; they only need a small correction for the fact
  that they now also see a weighted mixture of inputs. The gate, by contrast, has a genuinely new
  objective. Using a lower learning rate for the part of the network that carries the most pre-trained
  knowledge is discriminative fine-tuning (Howard and Ruder, 2018, §3.2: "discriminative fine-tuning
  allows us to tune *each* layer with different learning rates").
- The gradient reaching an expert is scaled by its routing weight `w_k`, so an expert that is
  currently rarely selected receives little signal while a dominant expert receives a lot; a smaller
  expert learning rate limits how fast that asymmetry can damage the specialists.
- The searched range for the joint `lr` (1e-5 .. 3e-4) is capped at the warm-up learning rate, so the
  joint stage can never be more aggressive than the warm-up, as the brief requires ("unfreeze the
  experts and jointly fine-tune the complete system using a smaller learning rate").

**BatchNorm in the joint stage (`freeze_expert_bn = True`, default).** The experts' BatchNorm *affine*
parameters train, but the running statistics stay at the values estimated during Task 2 training. The
input distribution of an expert changes during joint training (it now sees all four conditions, not
only its own), so re-estimating the statistics on MoE batches would move every expert's normalisation
away from the regime it was trained in, for inputs it is supposed to handle. Keeping pre-training
statistics fixed when fine-tuning is standard practice (He, Girshick and Dollár, 2018, *Rethinking
ImageNet Pre-training*, §3.1: "fine-tuning can adopt the pre-training batch statistics as fixed
parameters"). `--expert-bn train` switches to re-estimation for an ablation.

**Schedules and optimiser.** AdamW; constant LR during the warm-up (it is only two epochs, a schedule
would add nothing); cosine decay to 1 % of the peak over the joint epochs, stepped per batch.
Gradient clipping at norm 1.0 and fp16 AMP on CUDA, with the losses evaluated in float32 (SSIM is
numerically unusable in fp16 - see `src/common/losses.py`). `channels_last` memory format on CUDA,
which suits the 3x3 convolutions of four CNNs on a T4.

**Early stopping** uses the validation objective and a patience of 6 **joint** epochs; the counter is
reset at the stage boundary so that the (deliberately short) warm-up cannot consume the patience of
the joint stage.

**Validation objective.** `restoration_score = 0.5 * SSIM + 0.5 * PSNR / 40` from
`src/common/metrics.py`, but with the per-image PSNR capped at 40 dB *inside the objective*. Reason:
`restoration_score` treats 40 dB as equivalent to a perfect SSIM of 1, and a clean input whose weight
sits on the identity branch reaches 60-100 dB. Without the cap, model selection would be dominated by
how exactly the clean quarter of the validation set is passed through, instead of by restoration
quality. The uncapped score is logged as well (`val/restoration_score`), and the cap applies only to
the selection objective, never to the reported test metrics.

## Loss

Start values from the brief: `l1 = 0.8`, `ls = 0.2`, `lc = 0.1`, `lb = 0.01`.

**Cross-entropy is applied to the raw gate logits `G(x)`, not to `G(x) / tau`** (`ce_input: logits`,
the default; `routing` is available as an option). Two reasons:

- With CE on `G(x)/tau`, the *same* parameter `tau` controls both how sharp the routing is and how
  strong the classification gradient is: a small `tau` multiplies the CE gradient by `1/tau` and makes
  the term dominate the loss, so the Optuna search over `tau` would be confounded with the search over
  `lc`. Keeping CE on the raw logits makes `tau` purely a routing-sharpness knob and `lc` purely a
  weight, so the two search dimensions are separable.
- CE on `G(x)` is exactly the Task 2 classifier objective, so the term keeps the gate a calibrated
  corruption classifier - which is what makes the routing weights interpretable in the gating analysis
  and in the application's "which experts contributed" display.

**The balance regulariser is the brief's** `L_balance = sum_k (mean_batch(w_k) - 1/4)^2`, implemented
in `routing.balance_loss`.

**Why it does not fight the cross-entropy - the key property to explain.** Training batches come from
`BalancedBatchSampler`, which puts exactly `batch_size/4` images of each condition in every batch.
Under perfect routing, `w` is one-hot on the true class, so each branch's batch-mean weight is exactly
`1/4` and `L_balance = 0`. In other words, with exactly balanced batches the regulariser's optimum
*coincides* with the cross-entropy's optimum: the term can only penalise deviations from per-branch
usage of 1/4, which is precisely what routing collapse looks like, and it is zero for the behaviour CE
is pushing towards. Total collapse onto one branch costs `(1 - 1/4)^2 + 3 * (1/4)^2 = 0.75`. This
property depends on the balanced sampler: with an unbalanced batch (say three clean images and one
salt), perfect routing would give mean weights of (0.75, 0.25, 0, 0) and the "balance" term would
actively punish correct routing. A test asserts both halves of this (zero for balanced perfect
routing, non-zero for an unbalanced batch). Its only inputs are the batch-mean weights, so it is
blind to *which* input got which weight - it prevents collapse and leaves the assignment to CE.

**Why a soft constraint on the batch-wise average is the right family of regulariser.** Shazeer et al.
(2017), *Outrageously Large Neural Networks*, §4 state the failure mode this term exists for: "the
gating network tends to converge to a state where it always produces large weights for the same few
experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus
are selected even more by the gating network", and they note that Bengio et al. (2015),
*Conditional Computation in Neural Networks*, §3.2 eq. 5-6, "include a soft constraint on the
batch-wise average of each gate" - which is exactly the form the brief prescribes. Shazeer et al.
themselves use `L_importance = w * CV(Importance(X))^2`, the squared coefficient of variation of the
batch-wise sum of gate values (eq. 6-7); the brief's term is the same idea with a fixed target of
`1/K` instead of a scale-free dispersion measure, which is appropriate here because the balanced
sampler fixes the correct target.

**Alternatives considered, not substituted.** The Switch Transformer load-balancing loss
(Fedus, Zoph and Shazeer, JMLR 2022, §2.2, eq. 4-6: `loss = alpha * N * sum_i f_i * P_i` with `f_i`
the fraction of tokens dispatched to expert `i`, `P_i` the mean router probability, `alpha = 1e-2`)
is "minimized under a uniform distribution". It is designed for *top-1 sparse* routing, where `f_i`
counts hard dispatches; here all four branches always run densely, so its hard-assignment factor
`f_i` adds nothing the brief's term does not already cover, and - importantly - it also targets
uniform usage, so it would behave like the brief's term rather than replace it. An entropy
regulariser was rejected on principle: maximising routing entropy would push *against* the
cross-entropy term (it rewards spreading weight over wrong branches) and minimising it would push
towards hard routing, which is Task 2. Entropy is therefore used here as a **diagnostic**, reported
per class in the gating analysis, not as a loss. The brief's term is implemented and used; no
replacement is claimed.

## Routing-collapse criterion

Defined once in `routing.detect_collapse` and used by the Optuna pruner, the per-epoch W&B log and the
final evaluation, so "collapse" means the same thing everywhere. It is computed on the **validation
manifest**, which is balanced across the four classes, from the matrix

`R[c, k] = mean weight of branch k over validation entries of true class c` (rows sum to 1),

with `usage_k = mean_c R[c, k]` (class-balanced share of branch `k`; `1/4` under balanced perfect
routing) and `off_k = mean_{c != k} R[c, k]` (its weight on inputs it is *not* responsible for).
A configuration has collapsed if, for any branch `k`:

- **dead branch**: `usage_k < 0.05` - a fifth of its fair share of 0.25. Since `usage_k >= R[k,k]/4`,
  a dead branch receives less than 0.2 of the weight even on the inputs it exists for, i.e. the MoE
  has effectively lost that expert; or
- **dominating branch**: `off_k > 0.5` - that branch takes more than half of the weight, on average,
  on inputs of the other three classes, so routing no longer follows the corruption type.

Both thresholds are configurable (`collapse_min_usage`, `collapse_max_off_class`). The criterion is
deliberately *not* "a branch below some weight on its own class": a model can be uncertain (weights
near 0.25 everywhere) without having collapsed, and that case is informative, not useless - the test
suite asserts that uniform routing is **not** flagged, while total collapse onto one branch is flagged
as both dead and dominating, and the two one-sided cases trigger exactly the expected reason. Total
collapse onto one branch is the failure mode Shazeer et al. describe, and it is what the balance term
is there to prevent; the criterion stops a trial that gets there anyway from burning GPU time.

## Optuna search space

Study `task3_moe`, direction maximise, TPE (seed 42) + `MedianPruner`, objective = the validation
score above. Trial 0 is enqueued with the brief's starting values, so the search always contains the
reference point. Each trial runs a short schedule (1 warm-up + 4 joint epochs) and is its own W&B run.

| Parameter | Range | Scale | Why |
|---|---|---|---|
| `lr` (joint) | 1e-5 .. 3e-4 | log | The brief's "joint fine-tuning learning rate". Capped at the warm-up LR so the joint stage stays the gentler one; experts follow at `0.1 * lr`. |
| `tau` | 0.3 .. 3.0 | log | The brief's temperature. 0.3 is near-hard routing (Task 2), 1.0 is the classifier's own probabilities, 3.0 is close to uniform blending - the whole hard-to-soft spectrum. |
| `lambda_ce` | 0.01 .. 1.0 | log | The brief's classification weight; one decade either side of its 0.1. |
| `lambda_balance` | 1e-3 .. 1e-1 | log | The brief's balance weight; one decade either side of its 0.01. Includes the Switch Transformer's `alpha = 1e-2`. |
| `alpha` | 0.5 .. 0.95 | linear | The brief's "reconstruction-loss weighting": `lambda_l1 = alpha`, `lambda_ssim = 1 - alpha`. Tying them to one parameter keeps the reconstruction scale constant, so the comparison with `lambda_ce` and `lambda_balance` stays meaningful; the range brackets the brief's 0.8 and matches Task 1. |

Two pruning paths, both recorded in the trial attribute `prune_reason` (and in `trials.csv`): the
median pruner for weak configurations, and the collapse criterion above. A diverged loss (NaN/inf) is
also converted into a prune with its reason rather than a crashed trial, so one bad learning rate
cannot end the session. `save_study_report` writes the history, importance, parallel-coordinate and
slice plots plus `trials.csv`; the winning configuration is written to
`OUTPUT_DIR/task3/best_config.yaml` with the full-length schedule substituted in.

## What the evaluation produces

`evaluate.py` runs `best.pt` in float32 over all 36,690 test entries:

- **Restoration**: `eval/test_records.csv` (one row per entry: PSNR, SSIM, MSE, the four weights, the
  dominant branch, the normalised routing entropy), `eval/summary.json`, and the standard tables per
  type, per type x severity and per severity, as CSV and LaTeX. Because clean inputs can be reproduced
  (nearly) exactly, PSNR is capped at 100 dB by `src/common/metrics.py`; the tables therefore also
  carry `n_exact` (entries at the cap) and `psnr_finite` (mean over the rest), as Task 2 does, and a
  `corrupted` row that excludes clean entries entirely.
- **Figures**: representative examples and failure cases as Target | Input | Output | |Error| rows via
  the shared `save_restoration_figure`, including a dedicated figure of the worst *mis-routed* entries.
- **The gating analysis the brief requires**:
  - average expert weights for **every true corruption type and severity level**:
    `tables/routing_by_type_level.{csv,tex}` and the heatmap `figures/routing_heatmap_type_level.png`
    (plus the compact 4x4 `figures/routing_matrix.png`);
  - **weight-distribution diagram**: `figures/weight_distributions.png`, the distribution of each
    branch weight within each true class;
  - **examples where one expert dominates and where the weights are spread**:
    `figures/gating_dominant.png` (lowest routing entropy per class) and `figures/gating_spread.png`
    (highest routing entropy), each showing every branch output with its weight, the mixed output and
    the target - selected by entropy rather than by hand, so they are reproducible;
  - **is an expert inactive or dominating unrelated inputs**: `tables/expert_activity.{csv,tex}` gives,
    per branch, its class-balanced usage, its mean weight on its own class, its mean and worst-case
    mean weight on the other classes, and how often it is the dominant branch on other-class inputs;
    `tables/routing_entropy.{csv,tex}` and `figures/routing_entropy.png` give the entropy statistics.
    `eval/summary.json` records the collapse verdict for the final model.

`compare.py` builds the cross-task table (type x severity) from the per-entry CSVs of Tasks 1-3,
skipping whatever is not there yet. Since all systems are evaluated on the same manifest in the same
order, it also reports **paired** per-entry PSNR/SSIM gains and win rates of the soft MoE, split by
whether the Task 2 classifier was correct - the case where soft routing should pay off. Hard routing
passes clean inputs through unchanged, so its clean PSNR sits at the cap; PSNR columns exclude capped
entries and `n_exact` columns count them.

## Optional extra experiment: two corruptions per image

Clearly **beyond the brief's corruption definitions** and labelled as such everywhere it appears. The
brief motivates soft routing with images containing more than one corruption, but its test manifest
has exactly one corruption per entry, so that claim cannot be tested on it. `make_mixed_manifest.py`
writes `manifests/pets_mixed_manifest.json`: 300 fixed test images (seed 3003) x {salt+blur,
blur+occlusion, salt+occlusion}, each component at its fixed **medium** test severity, built from the
existing corruption functions with stored specs so it is exactly reproducible. Components are applied
in physical order - blur (optics), then occlusion (an object in front), then salt-and-pepper
(sensor/transmission noise) - so salt pixels are never smeared by the blur that precedes them.
`evaluate_mixed.py` compares the soft MoE, the same MoE forced to argmax routing, Task 2 hard routing
and the Task 1 universal model on these inputs, and reports the mean routing weights per combination
plus the share of entries whose two largest weights are exactly the two corruptions present.

## ONNX export

`export_onnx.py` exports the **complete pipeline as one graph** (gate, temperature constant, softmax,
identity branch, three experts, weighted sum) to `ONNX_DIR/moe.onnx`, opset 17, dynamic batch axis,
input `input`, outputs `output`, `weights`, `branch_outputs` in `CLASSES` order - the contract in
`docs/CONVENTIONS.md` and what `docs/api.md` expects. Parity against ONNX Runtime is checked on **40
real test inputs** (4 per type x severity group, not random noise, so the gate sees realistic logits),
and the script exits with an error if any output exceeds 1e-4 or if fewer than 32 inputs were used.
Measured on the development stand-ins: 1.8e-7 maximum absolute difference. It additionally checks
batch sizes 1 and 3 against PyTorch, asserts the three output shapes and that the exported weights sum
to 1, and writes the model card `moe.json` (tau, model configs, I/O spec, preprocessing, test metrics
and routing summary from `evaluate.py`, hyperparameters, parity result, W&B run, git commit).

## Resuming after a Colab disconnect

`last.pt` is written **every** epoch (atomically) and holds the model, both optimiser groups, the
scheduler, the AMP scaler, the early-stopping state, the stage, the full history and the W&B run id;
`best.pt` is written before `last.pt` so a crash in between can never leave `best.pt` behind. Resuming
restores the stage, rebuilds the right optimiser and continues the schedule; data order and runtime
corruptions depend only on `(seed, epoch)`, so a resumed run sees the same data as an uninterrupted
one. A test interrupts a 3-epoch run inside epoch 3 and asserts that the resumed weights match the
uninterrupted run's exactly. The W&B run is continued by passing `id=` and `resume="allow"` to
`wandb.init` (never through `WANDB_RUN_ID`, which wandb reads once per process). A finished run
reports that and exits instead of retraining. The Optuna study is resumed by `src.common.optuna_utils`
(SQLite on local disk, snapshotted to Drive after every finished trial).

## Costs and measured times

Per image, one MoE forward pass runs the gate plus three autoencoders. With the default Task 1/2
architectures (gate 1.18 M parameters, three experts 12.45 M total) that is **17.3 GFLOP per image**
forward (the gate is only 1.7 of it) and about **51.8 GFLOP per image** for a full joint training step
- roughly four times a single-autoencoder task, as the budget assumed.

Measured on this development machine (4-thread CPU, PyTorch 2.12, no GPU), default architectures:
1.1 training images/s in the joint stage, 3.2 images/s in the warm-up (only the gate has gradients),
4.8 images/s inference. The CPU smoke runs (tiny synthetic data, stand-in Task 2 models) take
14 s for `train --smoke` (both stages), 3 s for `train --smoke --resume` (already finished),
24 s for `optuna_search --smoke` (2 trials), 14 s for `evaluate --smoke`, 4 s for `compare --smoke`,
6 s for `evaluate_mixed --smoke` and 6 s for `export_onnx --smoke` - about 70 s for the whole
sequence; the test suite takes 47-76 s depending on the filesystem cache.

Extrapolating the measured FLOPs to a T4 (and assuming the Task 2 default architectures): an Optuna
trial of 1 + 4 epochs over the 2,944 training images is roughly 4-6 min, so 20 trials fit in about
1-1.5 h with the 90-minute cut-off as the hard stop; the final 2 + 20 epochs are roughly 20-35 min;
evaluation over 36,690 test entries is roughly 5-8 min. Total about 2-2.5 h, i.e. the task's budget.
If the Task 2 specialists turn out larger than assumed, lower `--n-trials` or
`--trial-joint-epochs` rather than the final run's epochs.

## Commands

```bash
# development (CPU, synthetic data, W&B disabled; stand-in Task 2 models if Task 2 is not trained)
python -m src.task3.train --smoke
python -m src.task3.optuna_search --smoke
python -m src.task3.evaluate --smoke
python -m src.task3.evaluate_mixed --smoke
python -m src.task3.compare --smoke
python -m src.task3.export_onnx --smoke
python -m pytest tests/test_task3.py -q

# Colab T4 (needs the trained Task 2 checkpoints; see notebooks/task3_colab.ipynb)
python -m src.task3.optuna_search --n-trials 20 --timeout-min 90      # approx. 1-1.5 h
python -m src.task3.train --config $OUTPUT_DIR/task3/best_config.yaml --resume   # approx. 20-35 min
python -m src.task3.evaluate                                          # approx. 5-8 min
python -m src.task3.compare                                           # < 1 min
python -m src.task3.evaluate_mixed                                    # approx. 1-2 min
python -m src.task3.export_onnx                                       # approx. 1 min

# one-off, already committed: the mixed-corruption manifest
python -m src.task3.make_mixed_manifest
```

## References

- Jacobs, Jordan, Nowlan and Hinton (1991). *Adaptive Mixtures of Local Experts*. Neural Computation
  3(1), 79-87. The original mixture-of-experts formulation with a gating network.
- Bengio, Bacon, Pineau and Precup (2015). *Conditional Computation in Neural Networks for Faster
  Models*. §3.2, eq. 5-6: a soft constraint on the batch-wise average activation of each gate - the
  family the brief's balance term belongs to.
- Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton and Dean (2017). *Outrageously Large Neural
  Networks: The Sparsely-Gated Mixture-of-Experts Layer*. §4: self-reinforcing expert imbalance;
  `L_importance = w * CV(Importance(X))^2` (eq. 6-7).
- Fedus, Zoph and Shazeer (2022). *Switch Transformers*. JMLR. §2.2, eq. 4-6: the differentiable load
  balancing loss for top-1 routing, `alpha = 1e-2`, minimised under a uniform distribution.
- Hinton, Vinyals and Dean (2015). *Distilling the Knowledge in a Neural Network*. §2: softmax with
  temperature; higher T gives a softer distribution.
- Howard and Ruder (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL.
  §3.2 discriminative fine-tuning (per-layer learning rates), §3.3 gradual unfreezing.
- He, Girshick and Dollár (2018). *Rethinking ImageNet Pre-training*. §3.1: fine-tuning may keep
  pre-training batch statistics fixed.
- Wang, Bovik, Sheikh and Simoncelli (2004). *Image Quality Assessment: From Error Visibility to
  Structural Similarity*. IEEE TIP 13(4). The SSIM used in the loss and the metrics.
