# FS2K data and Task 4 model notes

Investigation of the FS2K dataset and the design of the Task 4 data pipeline and cGAN
models (`src/data/fs2k.py`, `src/task4/models.py`). Every number below was measured on
the extracted dataset (`data/FS2K`, from `FS2K.zip`, Google Drive id `1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp`)
unless it is attributed to the dataset README.

Reproduce: `python scripts/prepare_fs2k.py` (verifies pairing, writes
`manifests/fs2k_split.json`, builds the cache) and `python scripts/preview_fs2k.py`
(figure `outputs/task4/figures/fs2k_pairs.png`).

## 1. Summary

| Item | Value |
|---|---|
| Pairs | 2,104 (official train 1,058, official test 1,046), every pair verified one-to-one |
| Our split | train 899 / val 159 / test 1,046 (val = 15% of official train, stratified by style, seed 42) |
| Styles (train / val / test) | style 0: 303 / 54 / 619; style 1: 298 / 52 / 381; style 2: 298 / 53 / 46 |
| Sources | photo1 CASIA-WebFace (250x250), photo2 eight invited actors (223x318), photo3 stock photos (475x340) |
| Sketch colour | R = G = B in every pixel of every sketch, so sketches are stored with 1 channel |
| Preprocessing | EXIF-upright, alpha onto white, centre square crop, bicubic (antialiased) resize to 128x128 |
| Augmentation | paired horizontal flip (p = 0.5) + pix2pix jitter (zoom 1 to 286/256, random 128 crop) |
| Main caveat | in the official training set style is confounded with photo source (Section 6.1) |

## 2. Directory structure

```
FS2K/
  README.pdf                dataset README (3.0 MB)
  anno_train.json           official training list + attributes (1,058 entries)
  anno_test.json            official test list + attributes (1,046 entries)
  photo/photo1/  1,529 files  image0001.jpg ...  (source: CASIA-WebFace)
  photo/photo2/     98 files  image0001.jpg ...  (source: eight invited actors)
  photo/photo3/    477 files  476 x .jpg + 1 x .JPG (image0449.JPG)  (source: Unsplash, Pexels, Pngimg, Google)
  sketch/sketch1/ 1,529 files sketch0001.jpg ...
  sketch/sketch2/    98 files sketch0001.png ...
  sketch/sketch3/   477 files sketch0001.jpg ...
```

The pairing rule is `photo/photo{k}/image{n}.*` ↔ `sketch/sketch{k}/sketch{n}.*`. Ids are
not contiguous (photo1 numbers run from 1 to 1,712 for 1,529 files), so pairs must be
matched by number, never by sorted position. Sources are taken from the README.

## 3. Annotation format and official split

Each JSON file is a list of objects; the file an entry appears in defines its official split.

```json
{"image_name": "photo1/image0110", "skin_color": [163, 139],
 "lip_color": [156.98, 82.51, 79.0], "eye_color": [118.65, 72.26, 69.60],
 "hair": 0, "hair_color": 2, "gender": 0, "earring": 1, "smile": 1,
 "frontal_face": 1, "style": 0}
```

- `image_name` is the pair id used everywhere in our code (`photo1/image0110`).
- `style` ∈ {0, 1, 2} is the sketch-style label (shown to users as Style 1/2/3). It is
  **not** the folder index (Section 4).
- The other keys are facial attributes (README: skin point, mean lip/eye RGB, hair visible,
  hair colour, gender, earring, smile, head rotation); Task 4 does not use them.
- All 2,104 entries have the same 11 keys; no id appears twice and train/test do not overlap.
- README discrepancies (data is authoritative): the README calls the key `skin_patch` but
  the files use `skin_color`; the README attribute table gives training counts
  S2 = 351, S3 = 350, while the annotations give style 1 = 350, style 2 = 351.

## 4. Counts per split, style and source

| Official split | Source | Style 0 | Style 1 | Style 2 | Total |
|---|---|---|---|---|---|
| train | photo1 | 357 | 350 | 0 | 707 |
| train | photo2 | 0 | 0 | 98 | 98 |
| train | photo3 | 0 | 0 | 253 | 253 |
| **train** | all | **357** | **350** | **351** | **1,058** |
| test | photo1 | 440 | 381 | 1 | 822 |
| test | photo3 | 179 | 0 | 45 | 224 |
| **test** | all | **619** | **381** | **46** | **1,046** |

After the 15% stratified validation split (Section 8):

| Split | Pairs | Style 0 / 1 / 2 | photo1 / photo2 / photo3 |
|---|---|---|---|
| train | 899 | 303 / 298 / 298 | 601 / 86 / 212 |
| val | 159 | 54 / 52 / 53 | 106 / 12 / 41 |
| test | 1,046 | 619 / 381 / 46 | 822 / 0 / 224 |

## 5. Image properties

| Source | Photos | Sketches | Size (W x H) | Aspect W/H | EXIF orientation |
|---|---|---|---|---|---|
| photo1 | 1,529 JPEG, RGB | 1,529 JPEG, RGB | 250x250 (all) | 1.000 | photos none, sketches 1 |
| photo2 | 98 JPEG, RGB | 98 PNG: 90 RGB + 8 RGBA | 223x318 (97), 205x292 (1: image0003) | 0.701 | all 1 |
| photo3 | 477 JPEG, RGB | 477 JPEG, RGB | 475x340 (all) | 1.397 | photos 1 (299) or none (178), sketches 1 |

- Every photo has exactly the size of its sketch (2,104 / 2,104), including the odd
  205x292 pair, so any geometric operation applied to both keeps them aligned.
- All photos are colour (no grayscale photos: mean per-pixel channel difference ≥ 1 for all).
- The shortest side is ≥ 205 px everywhere, so 128x128 is always a downscale (factor 0.512
  for photo1, 0.574 for photo2, 0.376 for photo3 after cropping).
- No EXIF tag other than 1 occurs, so no FS2K image needs rotating. `ImageOps.exif_transpose`
  is still applied because phone uploads in the app often carry orientation 6 or 8.
- All 4,208 files decode without error.

### Sketches are grayscale

For all 2,104 sketches (RGB and RGBA alike) the maximum of |R−G| and |G−B| over all pixels
is **0**. Converting to one channel (`L`) is therefore lossless: for R = G = B, Pillow's
luma conversion returns R exactly. **Decision: sketches have 1 channel** (generator output
`[N,1,128,128]`, allowed by the ONNX contract). Three channels would triple the output
layer and L1 terms for identical information; the app replicates the channel for display.

### Paper background and ink per style

Measured on the original sketches (grey level 0–255; "paper" = 90th percentile of the
sketch, "ink" = fraction of pixels darker than 128):

| Official split, source, style | n | Paper level | Ink fraction |
|---|---|---|---|
| train photo1 style 0 | 357 | 251 | 0.080 |
| train photo1 style 1 | 350 | 250 | 0.213 |
| train photo2 style 2 | 98 | 252 | 0.171 |
| train photo3 style 2 | 253 | **232** | 0.106 |
| test photo1 style 0 | 440 | 251 | 0.074 |
| test photo1 style 1 | 381 | 249 | 0.266 |
| test photo3 style 0 | 179 | **241** | 0.082 |
| test photo3 style 2 | 45 | **232** | 0.111 |

Style 0 is light contour drawing, style 1 has dense dark strokes and style 2 heavy hatching
(see the preview figure). photo3 sketches were drawn on grey paper (Section 6.1).

## 6. Anomalies and how they are handled

### 6.1 Style is confounded with photo source (most important)

In the official training set, styles 0 and 1 come **only** from photo1 (CASIA-WebFace,
250x250, tightly cropped celebrity faces) and style 2 **only** from photo2/photo3 (studio
cut-outs and stock photos). Consequences:

- G could learn part of "style 2" from photo appearance instead of from the label. Styles 0
  and 1 share a source, so for them the label is the only cue.
- The official test set contains combinations never seen in training: **179 photo3 photos
  with style-0 sketches** and 1 photo1 photo with a style-2 sketch (photo1/image1107). These
  measure how well the style condition generalises to a new photo domain, which is the app's
  situation (any photo, any style). Stage 2 should report test metrics per style and
  separately for seen vs unseen (source, style) combinations.
- Grey paper occurs in training only in photo3/style-2 sketches (paper 232 vs 250–252 elsewhere).
  The test photo3/style-0 ground truth has paper 241, while training taught style 0 white
  paper. This adds roughly a 10-grey-level background offset to those 179 test pairs.
  We keep the original sketches (no paper normalisation) so that test metrics are against
  the official ground truth, and report this effect instead.
- Choices that avoid creating extra shortcuts: centre crop instead of padding (Section 7)
  and FiLM conditioning at every decoder layer, so the label has a strong path (Section 10).

### 6.2 Other anomalies

| Anomaly | Measurement | Handling |
|---|---|---|
| Upper-case extension | `photo3/image0449.JPG` | extensions matched case-insensitively (Colab is case-sensitive) |
| RGBA sketches | 8 PNGs in sketch2; alpha < 255 only on one border column (7 images, 0.45% of pixels) or a 16-px band (image0065, 7.2%); RGB below is near-white (238–254) | alpha composited onto white; changes at most 6 grey levels |
| Odd-size pair | photo2/image0003 is 205x292 (others 223x318) | photo and sketch match; the generic crop handles it |
| Black padding in photos | 163 photos have a pure-black (max ≤ 6) border band ≥ 2 px (157 photo1, 5 photo3, 1 photo2); median width 12 px, 104 ≥ 10 px, 31 ≥ 25 px. In photo1 this is CASIA alignment padding (the sketch shows paper there); in photo3 they are real black studio backgrounds | kept: valid pairs that teach G to ignore black borders, which uploads may also have |
| Off-centre faces | 9 of 477 photo3 images lose > 25% of their sketch ink under the centre crop (32 lose > 10%) | kept: photo and sketch are cropped identically, so the pair stays aligned |
| Few identities in photo2 | README: 98 photos of eight actors | a random split puts the same actors in train and val, so val scores on photo2 (12 pairs) are optimistic. No identity labels exist; the official test set has no photo2 |
| Test style imbalance | style 2 has only 46 test pairs (619 / 381 / 46) | report per-style metrics and a macro average, not only the pooled mean |
| Missing / unannotated / duplicate files | none | `official_pairs` raises if any appear |

## 7. Preprocessing to 128x128: centre crop, not plain resize or padding

27% of the images (photo2 portrait 0.70, photo3 landscape 1.40) are not square. Options considered:

1. **Plain (anisotropic) resize**: distorts faces in opposite directions: photo2 is stretched
   horizontally by 1.43x (scale 0.574 vs 0.403) and photo3 is squeezed to 0.72x (0.269 vs 0.376).
   In training both appear only with style 2, so the distortion would become a style
   shortcut, and webcam frames (landscape) would be squeezed like photo3.
2. **Pad to square**: loses nothing, but adds constant bands to every photo2/photo3 image (all
   style 2 in training, another shortcut), shrinks photo3 heads to roughly half the frame
   height vs most of it for photo1 (visual estimate), and spends a quarter of the 128 px on padding for landscape webcam frames.
3. **Centre square crop then resize** (chosen): a no-op for the 1,529 square photo1 pairs
   (73%). For photo3 it removes 67/68 px of side background; on average 2.4% (median 0.2%)
   of the sketch ink lies there. For photo2 it removes 47/48 px at the top and bottom,
   20.1% of the ink on average (11.5% hair top, 8.6% chin/neck), but the face
   itself is kept (contact sheets inspected). A top-aligned crop would remove 21.9%, so
   it is not better. The aspect ratio is always preserved and no artificial content is added.

The crop is applied with identical geometry to photo and sketch (same size, same box), then
Pillow's bicubic resize, which widens its kernel with the reduction factor and so is
antialiased. The same function serves the app: `preprocess_photo(pil_image)` returns
float32 `(3,128,128)` in [-1, 1]. The test suite checks that it reproduces the training
cache bit-for-bit on real FS2K files. The backend should import it (or mirror these exact
Pillow calls; OpenCV resizing would differ slightly). The model expects a roughly centred
face that fills a good part of the square, as in FS2K.

## 8. Validation split

`make_split`: validation size = round(0.15 x 1,058) = 159. It is allocated to styles by
largest remainder (exact shares 53.55 / 52.5 / 52.65 give 54 / 52 / 53, each within one of
its exact 15%). Within each style the sorted ids are permuted by `np.random.default_rng(42)`
and the first n are taken. The official test list is copied unchanged and never used for
training or selection. The result is committed as `manifests/fs2k_split.json` (ids, style
per id, counts). `prepare_fs2k.py` validates an existing manifest instead of regenerating it,
so a future NumPy change cannot silently alter the split. `check_split` rejects any
manifest that does not partition the official lists.

## 9. Augmentation (training only, paired)

Photo and sketch are stacked into one 4-channel tensor before any spatial transform, so
both always receive the identical transform by construction. Tested by feeding pairs whose
photo channels equal the sketch and checking they stay equal after augmentation.

- **Horizontal flip, p = 0.5.** Faces are roughly mirror-symmetric, and flipping doubles pose
  and lighting variety for only 899 training pairs. Checks for asymmetric content:
  (a) a montage of 96 random sketches shows no text, signatures or watermarks; (b) stroke
  direction, measured by the structure-tensor asymmetry A = Σ gx·gy / Σ ½(gx² + gy²) (a flip
  negates A). Photos: mean A = +0.0007 (51% positive), i.e. symmetric. Sketches at full
  resolution are asymmetric (artist hatching direction): style 0 mean −0.061, only 26% positive,
  t = −16.9. At the 128x128 training resolution this mostly averages out: style 0 −0.004
  (t = −1.0), style 1 +0.023 (t = +5.8), style 2 +0.016 (t = +3.3), against a per-image std
  of 0.10–0.13 (effect size ≤ 0.21). The benefit outweighs this small mirrored-stroke bias;
  `hflip=False` exists for an ablation.
- **pix2pix jitter**: upscale by a factor drawn from [1, 286/256] (128 to at most 143 px,
  bilinear) and crop a random 128 window, i.e. scale ±6% around the mean and shifts of up to
  15 px. pix2pix always resizes 256→286 and then crops. Our factor includes 1.0, so the
  unaugmented test/app framing is part of the training distribution. Bilinear keeps values
  in [-1, 1].
- **Not used**: rotation (needs different fills for the photo and the sketch, and blurs thin
  strokes), and photometric jitter of the photo (allowed, since it is not spatial, but left
  as a stage-2 option for webcam robustness).

## 10. Design decisions (models)

### 10.1 Generator: pix2pix U-Net with style FiLM in the decoder

Architecture: the pix2pix `unet_128` layout (Isola et al., 2017; U-Net of Ronneberger et al.,
2015). Seven 4x4 stride-2 convolutions (widths c, 2c, 4c, 8c, 8c, 8c, 8c; 128 → 1x1),
LeakyReLU(0.2), then six transposed convolutions with skip connections and a tanh output.
Dropout (rate tunable, pix2pix uses 0.5) follows the three innermost decoder layers.
Parameters: 41.89 M at c = 64 (FiLM adds 0.067 M at style_dim 16), 23.58 M at c = 48 and
10.49 M at c = 32.

Style conditioning: one `nn.Embedding(3, style_dim)`. At each of the six decoder layers a
conditional instance norm computes `IN(h)·(1 + γ_l(e)) + β_l(e)`, where γ_l and β_l are
linear projections of the shared embedding e. This is FiLM (Perez et al., 2018) in the form
of conditional instance normalisation, which Dumoulin et al. (2017) introduced to let one
network render several artistic styles. A shared embedding projected to every layer's gains
and biases is BigGAN's "shared embedding" (Brock et al., 2019), and injecting the style by
AdaIN-style modulation in the decoder is what StarGAN v2 does for multi-domain translation
(Choi et al., 2020; AdaIN: Huang & Belongie, 2017).

Alternatives considered:

| Option | Assessment |
|---|---|
| Broadcast and concatenate the embedding at the input (cGAN-style; Mirza & Osindero, 2014; StarGAN, Choi et al., 2018) | A spatially constant input map is only a conditional bias on the first layer (Dumoulin et al., 2018, *Feature-wise transformations*), and the instance norm of the next layer subtracts per-channel offsets, so the signal is largely normalised away |
| Inject at the 1x1 bottleneck | One injection point far from the output; the U-Net skips let the decoder bypass the bottleneck, so the style can only steer the coarse path |
| **FiLM / conditional IN at every decoder layer (chosen)** | Style modulates features at every resolution 2–64 px, where stroke thickness, darkness and hatching are rendered. The encoder stays style-agnostic (content features), the extra cost is tiny, and it exports to plain ONNX ops |

Controlled check (scratch experiment, not part of the repo): one synthetic photo with three
target sketches that differ only in style. Same U-Net (c = 8, style_dim 16), Adam lr 1e-3,
400 steps, 2 seeds; L1 with the true style vs a shifted style. A style-blind model can only
reach the median of the three targets, about 0.054.

| Conditioning | L1 true style | L1 shifted style | Gap |
|---|---|---|---|
| none | 0.0539 | 0.0539 | 0.0000 |
| input concatenation | 0.0558 | 0.0558 | 0.0000 |
| bottleneck injection | 0.0530 | 0.0556 | 0.0026 |
| **decoder FiLM (chosen)** | 0.0452 | 0.0555 | **0.0104** |
| FiLM in encoder and decoder | 0.0445 | 0.0553 | 0.0108 |

Input concatenation was not used at all, as predicted by the bias argument above. Decoder
FiLM learns to use the style 4x faster than bottleneck injection, and also modulating the
encoder adds nothing measurable. The experiment also shows the network learns shared
structure before style, so stage 2 should log a style-sensitivity metric (validation L1
with the true vs a shifted style).

Other generator choices:

- **Instance norm instead of batch norm.** pix2pix's batch norm with batch size 1 and test-batch
  statistics is instance norm (Isola et al., 2017; Ulyanov et al., 2016). IN makes training
  and eval identical, so ONNX inference matches training. It also makes the network
  independent of the batch size, which Optuna tunes. No norm is applied on the 1x1
  bottleneck, where IN would output zeros, nor on the first layer (as in pix2pix). Convs
  followed by a norm have no bias (the norm removes it).
- **FiLM initialisation.** Weights N(0, 0.02²/style_dim) so that γ and β start with std 0.02
  for any style_dim (the embedding is N(0, 1)): the decoder starts as plain IN for every style,
  and Optuna trials with different style_dim start alike. Raising this to 0.2 did not speed
  up style learning in the experiment above. Other weights use pix2pix init: N(0, 0.02),
  norm scales N(1, 0.02), zero biases.
- **γ and β are two separate `Linear` layers.** A fused `Linear` + `chunk()` exports an
  opset-18 `Split(num_outputs)` that becomes invalid when torch 2.12's dynamo exporter
  down-converts to opset 17 (Section 11).
- **Dropout only in train mode.** pix2pix keeps dropout at test time as its noise source. We
  want deterministic sketches and exact ONNX parity, so inference runs in eval mode.
- **ConvTranspose upsampling** as in pix2pix. Resize-convolution avoids checkerboard
  artifacts (Odena et al., 2016). Kernel 4 / stride 2 is divisible, which reduces them; this
  is the fallback if artifacts appear in stage 2.
- **What style_dim can change.** With three categories and affine projections, any
  style_dim ≥ 2 can represent arbitrary per-style γ and β. style_dim mainly changes the
  parameterisation and optimisation, so we expect low importance in the Optuna study.

### 10.2 Discriminator: 70x70 PatchGAN with a projection style term

Body: pix2pix `NLayerDiscriminator` with n_layers = 3 on concat(photo, sketch) (3 + 1
channels): C64(s2, no norm), C128(s2), C256(s2), C512(s1), each with IN and LeakyReLU(0.2).
The output is a 14x14 map of logits for 128x128 inputs (2.90 M parameters). The geometric
receptive field is 70x70 (verified by gradients with norms disabled; n_layers = 2 gives 34x34
and a 30x30 map). Instance-norm statistics make each logit depend weakly on the whole image,
as pix2pix's batch-1 batch norm did. pix2pix found 70x70 best among 1, 16, 70 and 286 px
patches at 256x256. At 128x128 a 70 px patch covers about half a face; `n_layers` is
configurable for an ablation.

Style conditioning, with D's own `nn.Embedding(3, style_dim)`: the output layer is a shared
4x4 kernel plus a style-specific 4x4 kernel `K(s) = reshape(W e(s))`, i.e. per patch p
`logit(p) = w·φ(p) + b + v(s)·φ(p)`, with φ(p) the features under the patch and v(s) = W e(s).
This is the projection discriminator of Miyato & Koyama (2018), applied to every patch:
D(x, y) = ψ(φ(x)) + yᵀVφ(x). It follows from assuming p(y | x) is log-linear in the features.
On ImageNet it beat both concatenation and AC-GAN (intra-FID 103.1 vs 141.2 vs 260.0).
Alternatives:

- **Input concatenation**: the same normalisation problem as in G (only a first-layer bias,
  removed by the IN of layer 2). Pix2pix-style concatenation is still used for the photo,
  which is a spatial condition.
- **AC-GAN auxiliary classifier** (Odena et al., 2017): pushes G toward easily classifiable
  samples and was clearly worse in Miyato & Koyama's comparison.
- **Multi-task heads** (one output per style, StarGAN v2): with one-hot embeddings this
  is equivalent to projection. Our heads share the kernel w and are generated from the
  learned embedding, as the assignment requires.

Implementation: one conv produces a logit map per style and `gather` picks the row's style,
so only the used embedding rows get gradients (tested). The style kernels are initialised
on the same scale as the shared kernel (std 0.02).

Options left for stage 2: spectral normalisation (Miyato et al., 2018) if D overpowers G,
and "wrong-style" negatives (real sketch with a wrong style labelled fake, like GAN-CLS in
Reed et al., 2016) or a StarGAN-like random-style adversarial term for G if the
style/source confound (Section 6.1) shows up as weak style control on photo1 photos.

## 11. ONNX export findings (torch 2.12.0+cpu, onnx 1.23.1, onnxruntime 1.30.0, onnxscript 0.7.2)

- `torch.onnx.export` defaults to `dynamo=True`. When asked for opset 17 it builds opset 18
  and down-converts (warning in the log). A `chunk`/`split` becomes `Split(num_outputs)`,
  which is invalid in opset 17, and ONNX Runtime then refuses the model (`INVALID_GRAPH`). The
  generator avoids `chunk`, so both exporters work.
- The dynamo exporter writes the weights to a separate `generator.onnx.data` file unless
  `external_data=False` is passed (the ~168 MB model fits in one file).
- On Windows consoles (cp1252) the dynamo exporter's progress messages contain an emoji and
  raise `UnicodeEncodeError`; set `PYTHONIOENCODING=utf-8`. Not an issue on Colab or in pytest.
- `dynamo=False` (TorchScript exporter) still works, emits native opset 17 and prints
  deprecation warnings.
- Dynamic batch: `dynamic_axes` for `dynamo=False`; `dynamic_shapes` with one shared
  `torch.export.Dim("batch")` for `photo` and `style` for `dynamo=True`.
- Parity of the default generator (c = 64, random init), 32 real FS2K validation photos
  (11 / 11 / 10 per style), CPUExecutionProvider: max |ORT − PyTorch| = **2.28e-6**
  (batch 32) and 2.03e-6 (batch 1) for both exporters. Graph opset 17, file 167.6 MB.
  `tests/test_task4_models.py` repeats this check (c = 16, both exporters, 32 inputs, < 1e-4).

## 12. Training (stage 2)

Code: `src/task4/{config,train,optuna_search,evaluate,export_onnx,metrics}.py`, notebook
`notebooks/task4_colab.ipynb`, tests `tests/test_task4_train.py`.

### 12.1 Objective and update rule

Per step (`train.gan_step`), exactly the brief's formulation:

1. `fake = G(photo, style)`.
2. **D**: `d_real = BCEWithLogits(D(photo, sketch, style), 1)`,
   `d_fake = BCEWithLogits(D(photo, fake.detach(), style), 0)`, optimise `0.5 * (d_real + d_fake)`.
   The factor 0.5 is pix2pix's ("we divide the objective by 2 while optimizing D, which slows
   down the rate at which D learns relative to G").
3. **G**: `g_adv = BCEWithLogits(D(photo, fake, style), 1)`, `g_l1 = L1(fake, sketch)`,
   optimise `g_adv + lambda_l1 * g_l1`.

The four components `d_real`, `d_fake`, `g_adv`, `g_l1` are kept apart everywhere: averaged per
epoch for the console line and the W&B `train/*` panels, and streamed every `log_every` steps as
`step/*`. They are the standard GAN health check: `d_real` and `d_fake` near 0.69 means D is at
chance, both near 0 with a rising `g_adv` means D is winning.

### 12.2 Decisions

| Decision | Choice | Why |
|---|---|---|
| Optimiser | Adam, beta1 = 0.5, beta2 = 0.999, for G and D | DCGAN (Radford et al., 2016): "leaving the momentum term beta1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training"; pix2pix uses the same with lr 2e-4 |
| LR schedule | constant for the first half of the epochs, then linear decay to 0 (`decay_fraction`) | the pix2pix schedule; the decay quiets the adversarial game at the end, so the best epoch is not a lucky oscillation |
| Separate `lr_g`, `lr_d` | both searched | their ratio, not their absolute size, sets the G/D balance, the main failure mode of GAN training |
| `base_channels` shared by G and D | one hyperparameter | keeps the two capacities balanced (a D much larger than G overpowers it) and halves the search dimension; the brief also lists a single "base channel count" |
| Early stopping | none | GAN validation curves are not monotone; stopping on a reconstruction metric would cut exactly the adversarial refinement the task is about. `best.pt` still tracks the best epoch, and Optuna's pruner handles hopeless trials |
| AMP | fp16 autocast on CUDA, one `GradScaler` per optimiser, `--no-amp` to disable | GANs are the fragile case for fp16, so the risky parts are kept out of it: `binary_cross_entropy_with_logits` is on PyTorch's fp32 autocast list, the L1 term is cast to fp32 explicitly, and validation never runs under autocast. The scaler skips overflowing steps, and an epoch whose mean losses go non-finite raises `NonFiniteLossError` (Optuna turns that into a pruned trial) instead of training on silently. Verified here only that the autocast path runs without dtype errors and gives finite losses (this machine is CPU-only); the first Colab epoch shows whether fp16 is stable, and `--no-amp` is the fallback |
| Batch size | searched over 4/8/16 | with instance norm everywhere (Section 10.1) it is a pure optimisation knob, not an architecture change |

### 12.3 Validation metric and model selection

`metrics.sketch_score = 0.5 * SSIM + 0.5 * (1 - L1)` on the validation split, with sketches mapped
to [0, 1]. It mirrors `restoration_score` for Tasks 1-3 (half structure, half pixel fidelity) and,
like it, excludes the tuned loss weight: it contains neither `lambda_l1` nor the discriminator, so
trials with different weights and different discriminators are comparable. `best.pt` is the
best-scoring epoch.

**Known limitation, and what is done about it.** Reconstruction metrics mildly prefer the blurry
L1-only optimum, so an unbounded search would push `lambda_l1` up and weaken the GAN term. Two
guards: `lambda_l1` is searched in a bounded range (25-200, log-uniform, centred on the brief's
100), and `edge_ratio` (mean gradient magnitude of the generated sketch divided by the ground
truth's; 1 = as sharp as the artist's drawing) is logged every epoch, recorded as an Optuna user
attribute and reported per style, so the sharpness cost of a large `lambda_l1` stays visible even
though it is not optimised. The principled fix is a distribution metric (FID) or a perceptual one
(LPIPS); both need packages this environment does not have - see Section 12.6.

`style_gap` = (mean validation L1 when G is given a wrong style) - (L1 with the true style),
averaged over both wrong styles, is logged every epoch. It is 0 for a model that ignores the
condition and grows as the style embedding starts to matter, which is the direct test of the
"embedding must be incorporated, not just an interface label" requirement.

### 12.4 Fixed validation samples

`train.fixed_sample_indices` picks two validation pairs per style, preferring distinct photo
sources, from the deterministic validation order - the same pairs in every epoch, every trial and
every run. Each grid row is **photo | ground truth | generated Style 1 | Style 2 | Style 3**, so one
figure shows both the development over epochs and whether the style condition changes anything.

### 12.5 Optuna (study `task4_cgan`)

| Hyperparameter | Range | Reasoning |
|---|---|---|
| `lr_g`, `lr_d` | 5e-5 to 5e-4, log | an octave either side of pix2pix's 2e-4, searched separately for the balance |
| `batch_size` | 4 / 8 / 16 | 224 / 112 / 56 updates per epoch on 899 pairs |
| `base_channels` | 32 / 48 / 64 | 10.5M / 23.6M / 41.9M generator parameters; 64 gives a 168 MB ONNX file |
| `dropout` | 0.0 to 0.5 | pix2pix's 0.5 is G's only noise source; 0 tests whether that noise helps at all here |
| `style_dim` | 8 / 16 / 32 | any dim >= 2 can represent three arbitrary (gamma, beta) sets, so this mostly probes optimisation (Section 10.1) |
| `lambda_l1` | 25 to 200, log | the brief's 100 in the middle; bounded for the reason in Section 12.3 |

Trial 0 is enqueued as the brief's/pix2pix's reference point (lr 2e-4/2e-4, batch 8, base 64,
dropout 0.5, style_dim 16, lambda 100), so every other trial is read against a known baseline.
TPE (seed 42), `MedianPruner(n_startup_trials=4, n_warmup_steps=3)`, 10 epochs per trial against
120 for the final run - the brief allows shorter trials, and the cost is that trials are ranked on
early training, which favours configurations that start fast. The selected configuration is then
retrained from scratch for the complete schedule. Trials only ever see the validation split.

### 12.6 Test evaluation

`src/task4/evaluate.py` reports **L1/MAE, SSIM, PSNR, edge_ratio and sketch_score** over the 1,046
official test pairs, as: overall, per style, per photo source, per (source, style) cell, a macro
average over styles (the pooled mean is dominated by style 0, which has 619 of the 1,046 pairs),
and split by whether the (source, style) combination occurs in our training split (Section 6.1).
Sketches are compared against the **original** ground truth with no background normalisation, so
the grey paper of the photo3 sketches counts as error; that is deliberate (metrics stay against the
official data) and is exactly why the per-source table exists. Outputs: per-pair
`eval/test_records.csv`, `eval/summary.json`, `eval/failure_cases.csv` (each with an automatic
pointer: unseen combination, grey-paper source, strokes too soft, too much ink), `tables/*.{csv,tex}`
and the figures `test_examples.png` (median-score examples per style), `style_swap.png` (one photo,
all three styles) and `failure_cases.png`.

**FID / LPIPS are not implemented.** FID through `torchmetrics` needs the `torch-fidelity` package
and LPIPS needs `lpips` (or torchvision backbone weights); both also download ImageNet weights at
runtime. Neither is installed here and the project rules say to report rather than install, so they
are listed as the recommended addition if the packages can be added. Note also that an FID over the
46 style-2 test pairs would be statistically meaningless; it would only be informative pooled or on
style 0.

### 12.7 Measured cost (this machine, CPU) and GPU estimates

CPU (Windows, fp32, batch 8, 112 steps per epoch, 899 training pairs): **0.27 s/step at
`base_channels` 16, 0.79 s at 32, 2.52 s at 64**, i.e. 0.5 / 1.5 / 4.7 min per epoch. The whole CPU
smoke suite (train + Optuna + evaluate + export on synthetic data) runs in about 28 s.
Scaling the base-64 number by a typical T4 speed-up gives **roughly 10-20 s per epoch** there, so
120 epochs is about 25-45 min and 20 trials x 10 epochs about 45-80 min: inside the 2.5 h budget,
but these are *estimates*, not T4 measurements. Every training line prints its real epoch time, and
the notebook says to lower `--epochs` if an epoch takes much more than ~25 s. The Optuna cell is
additionally hard-capped with `--timeout-min 80`, and both stages resume after a disconnect.

### 12.8 ONNX export

`src/task4/export_onnx.py` exports **only the generator** (the discriminator is a training
component) from `best.pt` to `ONNX_DIR/generator.onnx` via `src.common.onnx_utils.export_onnx`
(TorchScript exporter, `dynamo=False`, opset 17 - Section 11 explains why that exporter is the safe
one here). Inputs `photo` [N,3,128,128] in [-1,1] and `style` int64 [N]; output `sketch`
[N,1,128,128] in [-1,1]; dynamic batch. Parity against ONNX Runtime is checked on 64 real test
photos spread over the split and again on a single image; the file is renamed `*.onnx.failed` if
the max absolute difference exceeds 1e-4. The model card `generator.json` records the exact
preprocessing string the backend must mirror (`preprocess_photo`), the test metrics from
`evaluate.py`, the hyperparameters, the parity result and the W&B run.

## References (verified against arXiv / proceedings pages)

- Brock, A., Donahue, J., Simonyan, K. *Large Scale GAN Training for High Fidelity Natural Image Synthesis.* ICLR 2019. arXiv:1809.11096.
- Choi, Y., Choi, M., Kim, M., Ha, J.-W., Kim, S., Choo, J. *StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.* CVPR 2018. arXiv:1711.09020.
- Choi, Y., Uh, Y., Yoo, J., Ha, J.-W. *StarGAN v2: Diverse Image Synthesis for Multiple Domains.* CVPR 2020. arXiv:1912.01865.
- Dumoulin, V., Shlens, J., Kudlur, M. *A Learned Representation for Artistic Style.* ICLR 2017. arXiv:1610.07629.
- Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., Bengio, Y. *Feature-wise transformations.* Distill, 2018. https://distill.pub/2018/feature-wise-transformations/
- Fan, D.-P., Huang, Z., Zheng, P., Liu, H., Qin, X., Van Gool, L. *Facial-Sketch Synthesis: A New Challenge.* Machine Intelligence Research (accepted). arXiv:2112.15439, 2021.
- Huang, X., Belongie, S. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017. arXiv:1703.06868.
- Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR 2017. arXiv:1611.07004.
- Mirza, M., Osindero, S. *Conditional Generative Adversarial Nets.* arXiv:1411.1784, 2014.
- Miyato, T., Koyama, M. *cGANs with Projection Discriminator.* ICLR 2018. arXiv:1802.05637.
- Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. *Spectral Normalization for Generative Adversarial Networks.* ICLR 2018. arXiv:1802.05957.
- Odena, A., Olah, C., Shlens, J. *Conditional Image Synthesis with Auxiliary Classifier GANs.* ICML 2017 (PMLR 70). arXiv:1610.09585.
- Odena, A., Dumoulin, V., Olah, C. *Deconvolution and Checkerboard Artifacts.* Distill, 2016. https://distill.pub/2016/deconv-checkerboard/
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. arXiv:1709.07871.
- Radford, A., Metz, L., Chintala, S. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (DCGAN). ICLR 2016. arXiv:1511.06434.
- Reed, S., Akata, Z., Yan, X., Logeswaran, L., Schiele, B., Lee, H. *Generative Adversarial Text to Image Synthesis.* ICML 2016. arXiv:1605.05396.
- Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. arXiv:1505.04597.
- Ulyanov, D., Vedaldi, A., Lempitsky, V. *Instance Normalization: The Missing Ingredient for Fast Stylization.* arXiv:1607.08022, 2016.
- Yi, D., Lei, Z., Liao, S., Li, S. Z. *Learning Face Representation from Scratch* (CASIA-WebFace). arXiv:1411.7923, 2014.
