# Research notes

Evidence base for the design decisions of Assignment 1. Every entry is a source that
was verified online against an authoritative page (publisher/DOI, arXiv, CVF open
access, PMLR/JMLR, NeurIPS/ICLR proceedings or official documentation); the BibTeX key
is the one used in `report/references.bib`, where each entry carries the verification
URL as a comment. Nothing here is cited from memory: claims attributed to a paper were
read in that paper's own text.

Conventions: "recommendation" is what the report should argue for; it matches the
implementation in `src/` unless flagged otherwise.

---

## Task 1 — Universal multi-corruption denoising autoencoder

### 1.1 Denoising autoencoders

| Alternative | Source | What it says |
|---|---|---|
| Denoising autoencoder (DAE) as a representation learner | `vincent2008denoising`, `vincent2010stacked` | Training an autoencoder to reconstruct a clean input from a stochastically corrupted one forces the model to capture structure in the data rather than learn the identity. The corruption is applied *at training time only*, exactly the runtime-corruption scheme the brief prescribes. |
| Autoencoder with a genuine compressed code | `hinton2006reducing` | A narrow central layer ("bottleneck") is what makes an autoencoder learn a compact representation; without it the network can learn the identity. |
| Convolutional autoencoders | `masci2011stacked` | Convolutional encoder/decoder stacks preserve 2-D locality and share weights, which is what makes AE training feasible on images. |
| CNN denoising/inpainting | `xie2012image` | Early demonstration that a single network trained on corrupted/clean pairs handles both denoising and (blind) inpainting. |
| Residual denoiser (DnCNN) | `zhang2017beyond` | A single CNN can cover a *range* of noise levels ("blind" denoising) instead of one model per level; this is the precedent for one model covering our three corruption families. |

**Relevance to our setting.** Our model must restore clean, salt-and-pepper, blurred and
occluded inputs with one shared representation, and is never told which corruption was
applied — a blind, multi-corruption version of the DnCNN setting.

### 1.2 Bottleneck vs skip connections — the central Task 1 decision

The brief requires "a genuine compressed latent representation" and explicitly says
"simply copying the input through unrestricted skip connections will not satisfy the
autoencoder requirement". The literature explains exactly why unrestricted skips defeat
a bottleneck:

- **U-Net** (`ronneberger2015unet`) concatenates every encoder stage into the mirrored
  decoder stage, including the highest-resolution one. The full-resolution skip carries
  the input's own high-frequency content around the bottleneck, so the deepest code is
  no longer a sufficient statistic for the output: the network can reconstruct detail it
  never encoded. That is desirable for segmentation and for pix2pix, and fatal for the
  assignment's "genuine bottleneck" requirement.
- **RED-Net** (`mao2016image`) uses symmetric skip connections between convolution and
  deconvolution layers and reports two distinct benefits in its own abstract: the skips
  (i) let the gradient reach the bottom layers directly, so very deep restoration
  networks train faster and reach a better optimum, and (ii) "pass image details from
  convolutional layers to de-convolutional layers", which helps recover the original
  image. Benefit (ii) is precisely the information leak the brief forbids when it is
  unrestricted.
- **pix2pix** (`isola2017image`) motivates the U-Net generator for image translation by
  the observation that input and output share low-level structure that should be able to
  bypass the bottleneck — the clearest statement that a skip-connected encoder-decoder is
  *not* an information bottleneck.
- **Context Encoders** (`pathak2016context`) solve inpainting with a *channel-wise fully
  connected* bottleneck and no skips, because occluded regions must be *hallucinated*
  from context rather than copied. This is the opposite regime from denoising: for our
  occlusion class the bottleneck is an asset, not a cost.

**Where the tension bites.** The four conditions pull in opposite directions. Salt-and-pepper
and mild blur are *local* problems where a high-resolution skip would be nearly free and
very helpful; occlusion is a *semantic* problem where the missing pixels must be
synthesised and a high-resolution skip would only forward black pixels. A single
universal model must satisfy both.

**Recommendation (implemented in `src/models/autoencoder.py`).** Keep a real
bottleneck — a 1x1 convolution projects the deepest encoder features to
`latent_channels`, and that tensor is the only path from input to output by default
(`skip_resolutions=()`). Expose *limited, low-resolution* skips as a tunable
(`skip_resolutions`), restricted to encoder resolutions strictly below the input
resolution (e.g. 16x16), never the 128x128 stem. Report the bottleneck size explicitly
as `latent_channels x (128 / 2**depth)^2` values versus the 3 x 128 x 128 = 49,152
values of the input, so the compression ratio is a stated number rather than a claim,
and justify any enabled skip by its effect on the per-corruption results (the brief
requires exactly this justification). A skip at 16x16 adds at most
`base_channels * 16 * 16` values of side information at 1/64 of the input's spatial
resolution — it cannot carry salt-and-pepper pixel values or sharp edges, so the
bottleneck still determines the output's high-frequency content.

### 1.3 Loss functions for image restoration

| Loss | Source | Finding |
|---|---|---|
| L2 / MSE | `zhao2017loss` | The default for image-processing networks produces "splotchy" artifacts in flat regions, because L2 punishes large errors but tolerates small ones regardless of local structure, while the human visual system is most sensitive to variation in texture-less regions. |
| L1 | `zhao2017loss`, `isola2017image` | L1 weights all errors equally, preserves colour and luminance, and converges to a better optimum than L2 in their experiments. pix2pix uses L1 rather than L2 explicitly because "L1 encourages less blurring". |
| SSIM / MS-SSIM | `wang2004image`, `wang2003multiscale`, `zhao2017loss` | SSIM compares local luminance, contrast and structure instead of per-pixel error. MS-SSIM evaluates it over a scale pyramid. Both are differentiable and usable as losses; MS-SSIM preserves contrast in high-frequency regions better than any other loss Zhao et al. tried. |
| Mixed MS-SSIM + L1 | `zhao2017loss` | Their recommended loss is `L_mix = alpha * L_MS-SSIM + (1 - alpha) * G_sigma . L_L1` with **alpha = 0.84**, set so the two contributions are roughly balanced; they note results were "not significantly sensitive to small variations of alpha". **Caveat for honesty in the report: their alpha weights the SSIM term, while the brief's alpha weights the L1 term, so the brief's suggested 0.8 and their 0.84 are not the same quantity.** |
| Perceptual / VGG loss | `johnson2016perceptual` | Feature-space losses give perceptually sharper results for super-resolution and style transfer, but they optimise a different objective than PSNR/SSIM and need a pretrained VGG at training time. |

**Recommendation.** Use the brief's `L = alpha * L1 + (1 - alpha) * (1 - SSIM)`
(`src/common/losses.py: RestorationLoss`) and tune `alpha` with Optuna rather than
accepting 0.8. Note in the report that (i) the pure-L2 baseline is rejected on the
evidence of `zhao2017loss`, (ii) we use single-scale SSIM rather than MS-SSIM because
at 128x128 a five-level MS-SSIM pyramid reduces the smallest scale to 8x8, below the
11x11 SSIM window, and (iii) perceptual loss was considered and rejected: it would
optimise a quantity different from the reported PSNR/SSIM and adds a VGG forward pass
per step on a time-limited T4.

**Why the validation objective is not the loss.** `restoration_score = 0.5 * SSIM + 0.5 * PSNR/40`
(`docs/CONVENTIONS.md`) is independent of the tuned `alpha`, so trials with different
loss weights remain comparable — otherwise Optuna would partly be optimising its own
yardstick.

### 1.4 Upsampling: transposed convolution vs resize-convolution

- `odena2016deconvolution` (Distill) shows transposed convolution has "uneven overlap"
  whenever the kernel size is not divisible by the stride; the unevenness multiplies
  across the two spatial axes and produces the characteristic checkerboard pattern. The
  recommended fix is **resize-convolution**: upsample (nearest-neighbour or bilinear)
  and then convolve, which is "implicitly weight-tying in a way that discourages high
  frequency artifacts". They report best results with nearest-neighbour.
- `aitken2017checkerboard` shows the artifact can also be removed by *initialising*
  sub-pixel convolution to be equivalent to nearest-neighbour resize-convolution
  ("ICNR"), i.e. the problem is one of initialisation/parameterisation rather than of
  transposed convolution per se.
- `shi2016real` introduces sub-pixel convolution (pixel shuffle) as the efficient
  alternative for super-resolution.

**Recommendation (implemented).** Bilinear `nn.Upsample` + conv, as in
`src/models/autoencoder.py`. Honest caveat for the report: Odena et al. had their best
results with nearest-neighbour and "difficulty making bilinear resize work" in their
setting; our decoder convolves after every upsample at every scale, which is the
structure they recommend, and our outputs are evaluated against a clean target, not a
style objective.

### 1.5 Metrics

- **PSNR**: `huynh2008scope` shows experimentally that PSNR is a valid quality measure
  as long as content and codec are fixed, but correlation with subjective quality drops
  sharply across different contents — so PSNR must be compared *within* our fixed test
  set, never quoted as an absolute quality level.
- **SSIM**: `wang2004image`, the canonical reference. Our implementation follows its
  settings exactly: 11x11 circularly-symmetric Gaussian window with standard deviation
  1.5 samples, K1 = 0.01, K2 = 0.03.
- `hore2010image` compares PSNR and SSIM directly and relates them analytically, useful
  for the sentence explaining why we report both.
- `zhang2018unreasonable` (LPIPS) is the reference for the claim that PSNR and SSIM are
  "simple, shallow functions" that miss nuances of human perception — the justification
  for also showing visual results and error maps, as the brief requires.

**PSNR cap.** Our `psnr` is capped at 100 dB for identical images
(`src/common/metrics.py`); for Task 2/3 clean inputs the identity bypass makes outputs
*exact*, so the report must state this rather than silently averaging capped values.
`restoration_score` uses PSNR/40, which caps the PSNR contribution at the score level.

### 1.6 Classical baselines (oracle comparators)

| Corruption | Classical method | Source |
|---|---|---|
| Salt-and-pepper | Median filter; adaptive and detail-preserving variants | `hwang1995adaptive`, `chan2005salt` — the median filter is the textbook impulse-noise filter; Chan et al. show median-type *detectors* plus edge-preserving regularisation beat plain median filtering at high impulse rates. |
| Blur | Unsharp masking / iterative deconvolution; Richardson–Lucy | `richardson1972bayesian`, `lucy1974iterative` — the classical iterative deconvolution pair. Our `unsharp_mask` with the *known* kernel is one step of Van Cittert-style iterative deconvolution. |
| Occlusion | Inpainting: Telea's fast-marching method; Navier–Stokes; variational inpainting | `telea2004image` (the method behind `cv2.INPAINT_TELEA`), `bertalmio2001navier`, `bertalmio2000image`. |
| Occlusion (learned) | Context Encoders | `pathak2016context` |

**Recommendation (implemented in `src/task1/baselines.py`).** Report two baselines:
*identity* (the corrupted input itself, which quantifies how damaging each corruption
is) and *oracle classical* (median / unsharp with the true kernel / Telea with the true
mask). The oracle baselines are deliberately given information the network never gets —
the corruption type and its exact parameters — so beating them is a strong result and
losing to them on one corruption is a legitimate, explainable finding rather than a
failure to hide.

---

## Task 2 — Corruption classification and hard routing

### 2.1 Classifier design

- `lecun1998gradient` and `he2016deep` are the standard architectural references for a
  convolutional classifier; `ioffe2015batch` for BatchNorm; `srivastava2014dropout` for
  dropout as the regulariser Optuna tunes.
- Our cues are unusual: salt-and-pepper is a *high-frequency, sparse* signal and blur is
  the *absence* of high frequencies. This motivates two choices in
  `src/models/classifier.py` that should be justified in the report: the first stage
  runs at full 128x128 resolution (early downsampling would destroy exactly the evidence
  the classifier needs), and the head concatenates global average and global max pooling
  (average pooling alone dilutes a few isolated bright pixels or one black rectangle).

### 2.2 Class-balanced sampling

- `buda2018systematic` systematically compares remedies for class imbalance in CNNs on
  MNIST/CIFAR-10/ImageNet and concludes that **oversampling is dominant in almost all
  scenarios**, should be applied to the level that *completely eliminates* the
  imbalance, and — unlike in classical ML — does not cause overfitting in CNNs.
- In our pipeline imbalance is not a property of the data but of the sampler: the
  corruption is drawn at load time, so exact balance is free. `BalancedBatchSampler`
  (`src/data/pets.py`) yields batches with exactly `batch_size / 4` of each condition,
  which is the strongest form of the remedy Buda et al. recommend and also makes the
  routing-balance statistics of Task 3 interpretable (a balanced batch means a perfectly
  routed MoE has mean weight exactly 1/4 per branch).

### 2.3 Metrics for the four-class problem

- `sokolova2009systematic` is the reference for *why* macro-averaged precision/recall/F1
  are the right summary for a multi-class problem where every class matters equally:
  macro-averaging treats classes as equally important regardless of their frequency,
  whereas micro-averaging/accuracy is dominated by the largest class.
- Because our test manifest is balanced *by construction* (see dataset section),
  accuracy and micro-F1 coincide; the report should say so rather than presenting them
  as independent evidence.
- **Normalised confusion matrices**: row-normalise by true class, so each row is the
  distribution of predictions for that class and reads as a recall profile. State the
  normalisation in the caption — an unnormalised matrix on a balanced test set differs
  only by a constant factor, but the normalised one is directly comparable with the
  Task 3 routing matrix, which is also row-stochastic.
- `guo2017calibration` shows modern networks are badly *calibrated* (over-confident),
  which matters here because Task 3 turns this classifier's logits into routing weights:
  the confidence of the gate is not a probability we should trust blindly, which is one
  more reason for the temperature `tau` in the gate.

### 2.4 Routing and specialist-tool approaches (related work)

| Approach | Source | Relation to our system |
|---|---|---|
| RL-Restore | `yu2018crafting` | Builds a *toolbox* of small CNNs specialised to different distortions and learns, by reinforcement learning, a policy that selects tools step by step for a corrupted image. The clearest precedent for "classifier picks specialist": they too find a toolchain of small specialists more parameter-efficient than one large network. |
| Path-Restore | `yu2022path` | A multi-path CNN with a *pathfinder* that selects a route per image region, trained with RL and a difficulty-regulated reward; achieves comparable performance at lower cost. Shows routing can be made spatial and content-adaptive — a limitation of our image-level routing worth naming. |
| AirNet | `li2022allinone` | All-in-one restoration without any prior on corruption type: a contrastive degradation encoder infers the degradation, which then guides restoration. The "no explicit classifier" alternative to our hard router. |
| PromptIR | `potlapalli2023promptir` | Prompt-based all-in-one blind restoration: lightweight learned prompts encode degradation-specific information and dynamically guide the network; state of the art on denoising/deraining/dehazing. The modern form of "one model, many corruptions" — the Task 1 philosophy taken to its conclusion. |
| Conditional computation | `bengio2015conditional`, `eigen2013learning` | The general framing: route each input through a subset of the network. Eigen et al. already observed the gating-collapse failure mode (see Task 3). |

**Recommendation.** Frame the four tasks as a ladder along a single axis: *one shared
model* (Task 1, cf. DnCNN/AirNet/PromptIR) → *discrete routing to specialists* (Task 2,
cf. RL-Restore) → *differentiable soft routing* (Task 3, cf. MoE). Cite AirNet and
PromptIR as the state of the art we are *not* claiming to match — they are trained on
far larger data with far larger networks — and RL-Restore/Path-Restore as the direct
antecedents of the routing idea.

---

## Task 3 — Soft mixture of experts

### 3.1 Foundations

- **`jacobs1991adaptive`** (Adaptive Mixtures of Local Experts, *Neural Computation*
  3(1):79–87). Read in the original. It distinguishes two error functions:
  - the *cooperative* one, `E^c = || d^c - sum_i p_i^c o_i^c ||^2` (their Eq. 1.1), where
    the target is compared with a *blend* of expert outputs. They argue this "does not
    encourage localization": each expert must cancel the residual left by all the
    others, the experts are strongly coupled, and it "tends to lead to solutions in
    which many experts are used for each case";
  - the *competitive* one, `E^c = sum_i p_i^c || d^c - o_i^c ||^2` (Eq. 1.2), where each
    expert must produce the *whole* output, so the system "tends to devote a single
    expert to each training case";
  - and the one actually used in their simulations,
    `E^c = -log sum_i p_i^c exp(-1/2 ||d^c - o_i^c||^2)` (Eq. 1.3), the negative log
    probability under a mixture-of-Gaussians model, whose gradient (Eq. 1.5) weights
    each expert by how well it does *relative to the others*.
  - The gating network is a softmax: `p_j = exp(x_j) / sum_i exp(x_i)`.

  **This is directly relevant and should be stated honestly in the report:** the brief's
  architecture, `x_hat = sum_k w_k A_k(x)` trained on the error of the blend, is exactly
  Jacobs et al.'s *cooperative* Eq. 1.1 — the formulation they identify as discouraging
  specialisation. Two things save it in our setting: the experts are *pre-trained* to
  specialise (Task 2) rather than learned from scratch, and the cross-entropy term
  `L_CE` on the gate supervises the routing directly with the known corruption label.
  Without those, blend-error training would be expected to drift towards many-experts-
  per-case.
- **`jordan1994hierarchical`** (Hierarchical Mixtures of Experts and the EM Algorithm,
  *Neural Computation* 6(2):181–214): the statistical formulation of MoE as a mixture
  model with an EM training algorithm; the source of the "softmax gating" terminology
  cited by Shazeer et al.
- **`yuksel2012twenty`** is a survey ("Twenty Years of Mixture of Experts", *IEEE TNNLS*
  23(8):1177–1193), useful for a single related-work sentence covering the gating and
  expert variants we did not try.

### 3.2 Balance regularisers — exact formulas from the papers

This is the decision the brief asks us to justify with research. Candidates, with the
formulas as published:

**(a) The brief's own term.**
```
L_balance = sum_{k=0..3} (mean_batch(w_k) - 1/4)^2
```
Only sees the batch-mean weights.

**(b) Shazeer et al. 2017 (`shazeer2017outrageously`), importance loss.** Verified
verbatim from the paper (Eqs. 6–7):
```
Importance(X) = sum_{x in X} G(x)
L_importance(X) = w_importance * CV(Importance(X))^2
```
i.e. the squared *coefficient of variation* of the per-expert importance (the batchwise
sum of gate values), times a hand-tuned scale. Its stated purpose: "This additional loss
encourages all experts to have equal importance."

**(c) Shazeer et al. 2017, load loss (their Appendix A).** Because equal importance does
not imply equal *numbers* of examples, they add a second term over a smooth estimator
`Load(X)_i = sum_{x in X} P(x, i)` of the number of examples routed to expert `i`:
```
L_load(X) = w_load * CV(Load(X))^2
```
`P(x,i)` (their Eqs. 8–9) is the probability that `G(x)_i` is non-zero under a fresh
draw of the gating noise — it exists only because their gate is *noisy top-k*.

Their Table 6 (10 epochs, MoE-256) is worth quoting in the report because it shows how
little the exact weights matter once *some* balance loss is present:

| w_importance | w_load | Test perplexity | CV(Importance) | CV(Load) | max/mean Load |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 39.8 | 3.04 | 3.01 | 17.80 |
| 0.2 | 0.0 | 35.6 | 0.06 | 0.17 | 1.47 |
| 0.0 | 0.2 | 35.7 | 0.22 | 0.04 | 1.15 |
| 0.1 | 0.1 | 35.6 | 0.06 | 0.05 | 1.14 |
| 0.01 | 0.01 | 35.7 | 0.48 | 0.11 | 1.37 |
| 1.0 | 1.0 | 35.7 | 0.03 | 0.02 | 1.07 |

Their conclusion in the text: "All the combinations containing at least one of the two
losses led to very similar model quality, where having no loss was much worse."

**(d) Switch Transformer (`fedus2022switch`), the modern simplification.** Verified
verbatim (their Eqs. 4–6). For `N` experts and a batch `B` of `T` tokens:
```
loss = alpha * N * sum_{i=1..N} f_i * P_i
f_i  = (1/T) * sum_{x in B} 1{argmax p(x) = i}      (fraction of tokens dispatched to i)
P_i  = (1/T) * sum_{x in B} p_i(x)                   (mean router probability for i)
```
They note it "simplifies the original design in Shazeer et al. (2017) which had separate
load-balancing and importance-weighting losses"; the `f` vector is *not* differentiable,
the `P` vector is; the factor `N` keeps the loss scale constant in `N` (under uniform
routing the sum is `1/N`); and they use **alpha = 1e-2**, chosen after sweeping
`1e-1 … 1e-5` in powers of ten as "sufficiently large to ensure load balancing while
small enough to not overwhelm the primary cross-entropy objective".

**(e) GShard (`lepikhin2021gshard`)** is the intermediate step Switch cites for the same
simplified auxiliary loss.

**(f) Soft MoE (`puigcerver2024soft`).** Takes the *structural* route instead of a
penalty: every expert slot is filled with a weighted average of all tokens, so Soft MoE
is "immune to token dropping and expert unbalance" **by construction and needs no
auxiliary balance loss at all**. It also names the failure modes of top-k routing it
avoids: token dropping and expert unbalance.

**(g) Entropy-based regularisers.**
- Per-example entropy *minimisation*, `-sum_k w_k log w_k` averaged over examples
  (`grandvalet2004semi`), makes individual routing decisions *confident*.
- Batch-marginal entropy *maximisation*, `H(mean_batch w)`, is the "class balance" term
  of Regularized Information Maximization (`krause2010discriminative`), which states
  plainly that entropy of the empirical label distribution is "a natural way to encode
  our preference towards class balance, because it is maximized when the labels are
  uniformly distributed". Their combined objective is exactly mutual information:
  `I{y;x} = H{mean_i p(y|x_i)} - (1/N) sum_i H{p(y|x_i)}` — maximise marginal entropy
  (balance) *and* minimise conditional entropy (confidence).
- `pereyra2017regularizing` goes the other way (penalising *low* output entropy to
  prevent over-confidence) — relevant if the gate saturates.

**Recommendation.** Keep the brief's quadratic term (a) as the primary regulariser and
justify it on three grounds, all defensible from the sources above:

1. **It is the right simplification for our architecture.** Shazeer's `L_load` and
   Switch's `f_i` exist to count *discretely dispatched* tokens in a top-k router. Our
   gate is fully dense and soft (all four branches always run), so there are no dropped
   tokens and no load to count — only importance. With `N = 4` fixed, the Switch loss
   reduces to a dot product of `f` and `P`; with soft routing `f` and `P` coincide, and
   `N * sum_k P_k^2` is minimised at the uniform `P` exactly like `sum_k (P_k - 1/4)^2`.
   The two differ only by a constant and a factor: `sum_k (P_k - 1/N)^2 = sum_k P_k^2 - 1/N`.
   **So the brief's term is, for a dense gate, the Switch load-balancing loss up to an
   affine transformation** — a statement worth making in the report, with the algebra.
2. **It does not fight the supervised objective.** Because `BalancedBatchSampler`
   guarantees exactly `batch_size/4` examples of each condition, *perfect* routing
   (one-hot on the true class) gives mean weight exactly 1/4 per branch, so `L_balance = 0`
   at the optimum of `L_CE`. The two terms are compatible by construction. This is a
   property of our pipeline that the MoE literature does not have (they have no labels),
   and it should be stated as the reason we can afford a plain balance term.
   Total collapse onto one branch costs `(1 - 1/4)^2 + 3 * (1/4)^2 = 0.75`.
3. **The exact form matters less than its presence.** Shazeer's Table 6 above: any of
   the losses, over two orders of magnitude of weight, gives nearly identical quality,
   while omitting them is much worse.

Report the alternatives (b)–(f) as considered-and-rejected with the reasons above, and
state the one honest caveat: because `L_balance` only sees *batch-mean* weights, it
cannot by itself detect a gate that routes every input to one branch per class in the
wrong assignment (e.g. a permutation). That is what the cross-entropy term and the
routing-matrix diagnostics in `src/task3/routing.py` are for: `R[c,k]` (mean weight of
branch `k` on true class `c`), `usage_k`, `own_k`, `off_k`, routing entropy normalised
by log 4, and `detect_collapse` (a branch is *dead* below 5% class-balanced usage,
*dominant* above 50% off-class weight). If Optuna's search drives the balance weight to
near zero without collapse, say so — that is a result, not a failure.

Should a stronger regulariser be wanted, the principled upgrade is the RIM-style
mutual-information term (`krause2010discriminative`): add
`-H(mean_batch w) + (1/N) sum H(w_i)`, which balances marginals *and* sharpens
individual decisions. It is differentiable, needs no noise, and subsumes both the
balance and the "don't route everything half-way" concerns.

### 3.3 Temperature-scaled softmax gating and routing collapse

- The brief's gate is `w = softmax(G(x) / tau)`. `guo2017calibration` is the reference
  for temperature scaling as the standard single-parameter method for sharpening or
  softening a softmax (there, for calibration). `hinton2015distilling` uses the same
  `softmax(z/T)` to control how much probability mass sits on non-maximal classes.
- `jang2017categorical` (Gumbel-Softmax) is the reference for the continuous relaxation
  of a *categorical* choice with a temperature — the alternative we did not take
  (stochastic hard routing with a differentiable estimator).
- **Routing collapse.** Shazeer et al. state the mechanism in their own words: "the
  gating network tends to converge to a state where it always produces large weights for
  the same few experts. This imbalance is self-reinforcing, as the favored experts are
  trained more rapidly and thus are selected even more by the gating network." They cite
  `eigen2013learning` for the same phenomenon (addressed there with a hard constraint
  early in training) and `bengio2015conditional` for a soft constraint on the batchwise
  average of each gate. Soft MoE (`puigcerver2024soft`) names the two symptoms of sparse
  routing it removes: token dropping and expert unbalance.

**Recommendation.** Fix `tau` per run as a buffer baked into the ONNX graph (as
implemented), tune it with Optuna, and report the routing entropy alongside the weights
so the chosen `tau` is interpretable: entropy near 0 means hard routing, near 1 means a
uniform blend. Prune trials that collapse (`detect_collapse`), and record the prune
reason — the brief explicitly permits pruning on routing collapse.

### 3.4 Staged training: freeze, warm up the gate, then fine-tune jointly

- `kumar2022finetuning` is the strongest evidence for the *shape* of our schedule. They
  show that full fine-tuning from a randomly initialised head *distorts* good pretrained
  features, because the lower layers move while the head is still wrong; the simple
  two-step remedy — **linear probing then full fine-tuning (LP-FT)** — "combines the
  benefits of both" (1% better in-distribution, 10% better out-of-distribution than full
  fine-tuning in their experiments). Our warm-up stage (experts frozen, only the gate
  trains) is exactly the probe stage, with the gate playing the role of the head: it
  prevents a mis-calibrated gate from pushing destructive gradients into three
  specialists that are already good.
- `howard2018universal` is the standard reference for discriminative (per-layer) learning
  rates and gradual unfreezing when fine-tuning a pretrained model — our
  `expert_lr_scale`, which keeps the experts at a fraction of the gate's learning rate
  during the joint stage, is that idea.
- `shazeer2017outrageously` initialise `W_g` and `W_noise` to zeros so that routing
  starts approximately balanced, "since the soft constraints need some time to work" —
  the same concern our warm-up addresses from the other direction (we start from a
  *trained* gate instead of a neutral one).

**Recommendation (matches `src/task3/train.py`).** Stage 1: experts frozen with
eval-mode BatchNorm (so their running statistics are not corrupted by gradient-free
forward passes), gate only. Stage 2: unfreeze, gate at `lr`, experts at
`lr * expert_lr_scale`, cosine decay. Initialise the gate from the Task 2 classifier and
the experts from the Task 2 specialists, never randomly — the brief requires this and
`kumar2022finetuning` explains why it matters.

---

## Task 4 — Style-conditioned face-to-sketch cGAN

### 4.1 Conditional GANs and pix2pix

- `goodfellow2014generative` — the GAN framework; also the source of the non-saturating
  generator trick (maximise `log D(G(z))` instead of minimising `log(1 - D(G(z)))`) that
  pix2pix adopts.
- `mirza2014conditional` — conditional GANs: both `G` and `D` receive the condition `y`,
  so the model learns `p(x | y)` rather than `p(x)`.
- `isola2017image` / `isola2016image` (CVPR version / arXiv) — pix2pix. Verified details
  from the paper itself:
  - objective `G* = argmin_G max_D L_cGAN(G,D) + lambda * L_L1(G)`, with L1 chosen over
    L2 because "L1 encourages less blurring";
  - **lambda = 100**;
  - **70x70 PatchGAN** is the default; a 16x16 patch is enough for sharpness but tiles,
    70x70 removes the tiling, and the full 286x286 ImageGAN is *worse*;
  - U-Net generator with skips between mirrored layers;
  - Adam, **learning rate 2e-4, beta1 = 0.5, beta2 = 0.999**; the `D` objective is
    divided by 2 to slow `D` relative to `G`;
  - noise is supplied only as **dropout**, applied at train *and* test time, because a
    noise input `z` was simply ignored by the generator;
  - removing the conditioning from `D` made the generator collapse to nearly the same
    output regardless of input — direct evidence for conditioning the discriminator.
- `radford2016unsupervised` (DCGAN) — the conv-BN-ReLU module design and the N(0, 0.02)
  initialisation pix2pix inherits.
- `heusel2017gans` (TTUR) — separate learning rates for `G` and `D`; also the source of FID.

**Note on the Task 1 / Task 4 contrast worth making explicitly in the report:** the same
U-Net skip connections that are *forbidden* in Task 1 (they would defeat the bottleneck)
are *mandated* in Task 4, because here the goal is a translation that preserves the
input's spatial layout rather than a compressed representation. The brief's own wording
makes this contrast, and it is a good illustration that architectural choices follow the
objective, not fashion.

### 4.2 Injecting a categorical condition

| Method | Source | Mechanism | Trade-off |
|---|---|---|---|
| Input concatenation | `mirza2014conditional`, `isola2017image`, `choi2018stargan` | Spatially replicate the (embedded) label and concatenate it to the input image (StarGAN: "The target domain label is spatially replicated and concatenated with the input image"). | Simplest. The signal must survive the whole encoder; easy for the network to ignore, and costs input channels. |
| Conditional instance / batch norm | `dumoulin2017learned`, `devries2017modulating` | Learn a per-condition scale and shift (`gamma_s`, `beta_s`) for the normalisation layers. Dumoulin et al. show a *single* style-transfer network captures 32 styles this way, and that interpolating `gamma`/`beta` interpolates styles. | Touches every layer, so the condition cannot be ignored; very few parameters. De Vries et al. modulate visual processing by language with the same mechanism. |
| FiLM | `perez2018film` | The general form: a feature-wise affine transform `gamma . h + beta` predicted from the conditioning input, applied throughout the network. | The umbrella formulation of the above. |
| AdaIN | `huang2017arbitrary` | Replace the normalised features' statistics with those of the style — used for *arbitrary* (unseen) styles. | More than we need: our three styles are fixed categories. |
| Auxiliary classifier (AC-GAN) | `odena2017conditional` | `D` also predicts the class; `G` is trained to produce classifiable images. | Adds a classification head; known to encourage mode-dropping within a class. |
| Projection discriminator | `miyato2018cgans` | `f(x,y) = y^T V phi(x) + psi(phi(x))`: an inner product between the embedded condition and the discriminator's own feature vector, plus the usual unconditional term. Motivated by the role of the condition in the underlying probabilistic model; improved class-conditional ImageNet generation over concatenation. | The principled way to condition `D`. |

**Recommendation (matches `src/task4/models.py`).** Condition the **generator** by FiLM /
conditional instance norm on the decoder: a learned embedding of the style index
produces per-layer scale and shift, so the condition modulates every decoder stage and
cannot be ignored, and the encoder stays style-agnostic (it only needs to read the
photograph). Condition the **discriminator** by projection (`miyato2018cgans`): each
patch logit is the unconditional logit plus the inner product of the patch features with
D's own style embedding. Justify against the simpler input-concatenation baseline:
the brief requires that "the embedding must be incorporated into the generator and
discriminator rather than being used only as an interface label", and concatenation at
the input is the weakest form of that.

### 4.3 Face sketch synthesis and FS2K

- `wang2009face` — the classical reference for face photo–sketch synthesis (multiscale
  Markov random field) and the CUHK datasets; the pre-deep-learning baseline for this task.
- **`fan2022facial`** — the FS2K dataset paper (Fan et al., *Machine Intelligence
  Research* 19(4):257–287, 2022). Verified from the paper itself:
  - **2,104 photo–sketch pairs**, split into **1,058 for training and 1,046 for testing**;
  - **three sketch styles** drawn by three senior artists, distinguished by the cheek
    region: simple lines (style 1), long strokes (style 2), repeated wispy details (style 3);
  - the *training* set is deliberately style-balanced (357 / 351 / 350) but the **test set
    is not: 619 / 381 / 46**;
  - extra attribute annotations (gender, smile, hair style/colour, earring, skin, pose);
  - photos largely from CASIA-WebFace plus actors, children and stock photography;
  - **metrics it reports for image-to-sketch: SSIM and SCOOT** (`fan2019scoot`, the
    structure co-occurrence texture metric), with SCOOT adopted because SSIM "ignores the
    perceptual similarity between a prediction and the reference"; a benchmark of 19
    state-of-the-art models is evaluated with them.
- **Consequence for our experiment design.** FS2K's own evaluation uses SSIM and SCOOT,
  not FID. SCOOT is purpose-built for facial sketches; if we report only SSIM/FID/LPIPS
  we are *not* comparable to the FS2K benchmark numbers, and the report must say so
  rather than implying a comparison. The style imbalance in the official test set (only
  46 style-3 images, 4.4%) means **per-style metrics are mandatory**: an aggregate number
  is dominated by style 1, and style 3 is estimated from 46 images.
- See `docs/fs2k_notes.md` for the repository's own data investigation, including the
  finding that style is confounded with the photo source folder in the official training
  split — which further restricts what a per-style comparison can claim.

### 4.4 GAN evaluation metrics and their caveats on small test sets

- **FID** (`heusel2017gans`): the Fréchet (Wasserstein-2) distance between Gaussians
  fitted to Inception-v3 activations of real and generated images.
- **Caveat 1 — bias.** `chong2020effectively` prove FID and the Inception Score are
  *biased*: the expected value at a finite sample size is not the true value, **and the
  bias term depends on the model being evaluated**, so model A can beat model B purely
  through a smaller bias. They state this "means all comparisons using FID or IS as
  currently computed are unreliable" and propose extrapolated `FID_inf` / `IS_inf`.
  With a 1,046-image test set — and only 46 images of style 3 — this is not a footnote
  but the dominant source of error; a per-style FID on 46 images is meaningless.
- **Caveat 2 — preprocessing.** `parmar2022aliased` show that low-level choices,
  specifically image resizing (aliasing from fixed-width prefilters) and JPEG
  compression, induce *large* variations in FID, and that compressing real training
  images can even *improve* FID if generated images are compressed too. Our images are
  resized to 128x128 with bicubic resampling, and Inception expects 299x299, so the
  resize path must be stated explicitly and kept identical for real and generated images.
- **Caveat 3 — FID's normality assumption.** `binkowski2018demystifying` propose KID
  (Kernel Inception Distance), noting that unlike FID it "does not assume a parametric
  form for the distribution of activations" — which matters because Inception features
  are ReLU outputs with ~2% exact zeros and no density — and that **KID has a simple
  unbiased estimator**, which FID does not.
- **Caveat 4 — metrics disagree.** `borji2019pros` surveys GAN evaluation measures;
  `lucic2018gans` show in a large-scale study that no GAN clearly dominates once the
  computational budget is matched, i.e. small metric differences are noise.
- **LPIPS** (`zhang2018unreasonable`): a learned perceptual distance that correlates with
  human judgement far better than PSNR/SSIM; it is a *paired* metric, so unlike FID it is
  well defined per image and usable on 46 images.
- **SSIM** (`wang2004image`): paired, interpretable, and what the FS2K benchmark reports.

**Recommendation.** Report, per style and overall: **SSIM** (comparable to FS2K's own
protocol), **LPIPS** (perceptual, paired, valid on small samples) and **FID** on the full
test set only — never per style, given 46 style-3 images — with an explicit statement of
the sample size, the resize path and the Chong–Forsyth bias caveat. Mention KID as the
better-behaved alternative at small `n`, and SCOOT as the FS2K-native metric we did not
implement. Do not compare our FID with numbers from other papers computed on other
preprocessing.

---

## Hyperparameter optimisation

- **Optuna** (`akiba2019optuna`): its three stated design criteria are (1) a
  **define-by-run** API that builds the search space dynamically, (2) efficient searching
  *and* pruning, (3) a versatile, easy-to-deploy architecture. Define-by-run is what lets
  the Task 1 study make `skip_resolutions` conditional on `depth` without declaring the
  whole space up front.
- **TPE** (`bergstra2011algorithms`): models `p(x | y)` as two densities — `l(x)` from
  trials whose objective beat a quantile threshold `y*` and `g(x)` from the rest — and
  maximises expected improvement by picking candidates with high `l(x)/g(x)`. Optuna's
  own documentation describes `TPESampler` in exactly these terms and defaults to
  `n_startup_trials = 10` random trials before the model takes over, `n_ei_candidates = 24`.
  `watanabe2023tree` dissects which TPE components actually matter.
  `bergstra2012random` is the baseline result that random search already beats grid
  search in high dimensions.
- **Median pruning** (`golovin2017google`, Google Vizier): the median stopping rule stops
  a trial at step `s` if its best objective so far is strictly worse than the median of
  the running averages of all completed trials at `s`. Vizier reports a consistent
  "factor two to three speedup over random search, while always finding the best
  performing Trial", and notes the rule is model-free, unlike parametric learning-curve
  stopping. Optuna implements it as `MedianPruner(n_startup_trials=5, n_warmup_steps=0,
  interval_steps=1, n_min_trials=1)`.
- **Successive halving / Hyperband** (`jamieson2016non`, `li2018hyperband`,
  `li2020system`): the bandit alternatives. Hyperband treats the budget-vs-number-of-
  configurations trade-off by running successive halving at several values of `n`;
  ASHA (`li2020system`) is the asynchronous version. Optuna provides
  `SuccessiveHalvingPruner(reduction_factor=4)` and `HyperbandPruner(reduction_factor=3)`.

**Recommendation (matches `docs/CONVENTIONS.md`).** TPE sampler seeded at 42 + MedianPruner.
Justification to state in the report: with 20–40 trials on a Colab T4 we are in the
regime where TPE's model is still useful but Hyperband's aggressive early stopping would
spend most of the budget on very short runs whose ranking is unreliable for GAN and MoE
training; the median rule is model-free and matches our fixed-epoch validation schedule.
For Task 3 the median rule is augmented by a *task-specific* pruning criterion — routing
collapse — which no generic pruner can express, and the brief explicitly allows it.
Record for each study: the full search space, number of completed/pruned trials, the best
trial, the final configuration, and the importance plot.

---

## Deployment

Cited as official documentation (web references, not papers):

- **ONNX** (`onnx`) — the open model-exchange format; **ONNX Runtime** (`onnxruntime`) —
  the cross-platform inference engine; **PyTorch ONNX export** (`pytorchonnx`) — the
  `torch.onnx` exporter documentation, for opset and dynamic-axis semantics.
- **FastAPI** (`fastapi`) — the Python web framework for the backend.
- **Docker Compose** (`dockercompose`) — the one-command multi-container startup the
  brief requires.
- **React** (`react`) and **Tailwind CSS** (`tailwind`) — the frontend stack; **Google
  Stitch** (`stitch`) — the AI design tool the brief mandates for the interface design.
- **Weights & Biases** (`wandb`) — experiment tracking; **Google Colab** (`colab`) and the
  **NVIDIA T4** (`nvidiat4`) — the training environment.
- **Optuna** (`optunadocs`) — the library documentation, alongside the paper.
- **PyTorch** (`paszke2019pytorch`) — cited as the NeurIPS paper, not as documentation.

**Recommendation.** Keep these as `@misc` entries with `howpublished`/`url` and an
`urldate`; do not dress documentation up as a paper. Parity between PyTorch and ONNX
Runtime outputs (max absolute difference below 1e-4 on at least 32 real inputs, recorded
in the model card) is our own verification requirement, not something a citation
establishes.

---

## Datasets

- **Oxford-IIIT Pet** (`parkhi2012cats`, CVPR 2012, pp. 3498–3505): 37 breeds of cats and
  dogs, ~200 images per class, with breed, head bounding box and trimap annotations. The
  dataset's own README states the official `trainval.txt` / `test.txt` are "splits used in
  the paper". We use the official trainval/test division and split trainval 80/20 with
  seed 42, as the brief requires; breed labels are unused (the clean image is the target).
- **FS2K** (`fan2022facial`) — see §4.3.
- **SCOOT** (`fan2019scoot`, ICCV 2019, pp. 5611–5621) — the FS2K-native sketch metric,
  cited but not implemented.

---

## Summary of recommendations that affect the implementation

1. **Task 1 — bottleneck:** keep `skip_resolutions=()` as the default and treat any skip
   as a tunable restricted to low resolutions (<= 16x16). State the compression ratio
   numerically. Justify any enabled skip by its per-corruption effect; the brief requires
   this and `mao2016image` gives the two mechanisms (gradient flow vs detail transfer) to
   attribute the gain to.
2. **Task 1 — alpha:** the brief's `alpha` weights L1; Zhao et al.'s 0.84 weights the
   SSIM term. Do not present 0.8 and 0.84 as agreeing.
3. **Task 3 — balance regulariser:** keep the brief's quadratic term, and justify it with
   the algebra showing it equals the Switch Transformer loss up to an affine transform
   for a dense gate, plus the fact that balanced batches make it zero at the optimum of
   the cross-entropy. Report Shazeer's Table 6 as evidence that the exact form matters
   less than its presence. Name the blind spot (permutations) and point to the routing
   matrix for it.
4. **Task 3 — honest framing:** the brief's blend-error training is Jacobs et al.'s
   *cooperative* error function, the one they argue discourages specialisation; say so,
   and explain why pre-trained experts plus the CE term make it work here.
5. **Task 4 — conditioning:** FiLM/conditional instance norm in the generator decoder +
   projection discriminator, rather than input concatenation.
6. **Task 4 — metrics:** SSIM + LPIPS per style; FID only on the full test set, with the
   bias and resize caveats stated. 46 style-3 test images make per-style FID invalid.
   Note that FS2K's own benchmark uses SSIM and SCOOT, so our numbers are not directly
   comparable to it.
