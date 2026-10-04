# Demonstration video script (target 6:30, allowed 5–7 minutes)

The brief asks the video to show: the application start-up, image uploading, runtime corruption,
universal restoration, hard routing, soft expert weights, face-to-sketch generation, result
downloading and the experiment-tracking records. Every item is covered below, with the exact
labels used by the interface (`frontend/src`), so the narration matches what is on screen.
Upload the finished video to YouTube (unlisted or public) and put the link in `report/main.tex`
(introduction and Availability section) and in the README.

## Before recording (not part of the video)

- [ ] The seven ONNX models and their model cards are in `./models` (`udae`, `classifier`,
      `specialist_salt`, `specialist_blur`, `specialist_occlusion`, `moe`, `generator`, each
      `.onnx` + `.json`).
- [ ] Docker Desktop is running; no old containers (`docker compose down`).
- [ ] A terminal is open at the repository root, with a large font.
- [ ] On the desktop: one pet photo that is **not** from the dataset (e.g. your own phone
      photo), one face photo you are allowed to show, and one already-corrupted image (for
      example a "Model input" PNG downloaded from the app in an earlier session).
- [ ] Browser tab 2: the Weights & Biases project
      <https://wandb.ai/mustafafaraz87-fast-nuces/genai-a1>, logged in, runs table visible.
- [ ] Webcam allowed in the browser if you want to show the webcam capture (optional).
- [ ] Do a full dry run once: the first `docker compose up --build` can take several minutes;
      record the start-up from a warm build cache or cut the waiting time.

## Timeline

| Time | Section |
|---|---|
| 0:00–0:20 | Introduction |
| 0:20–1:10 | Start-up with Docker Compose, system information |
| 1:10–2:25 | Universal Restoration: upload, runtime corruption, restore, download |
| 2:25–3:40 | Hard-Routed Restoration: predicted vs oracle routing, identity bypass |
| 3:40–4:35 | Soft Mixture-of-Experts Restoration: expert weights |
| 4:35–5:35 | Face-to-Sketch Generator: upload or webcam, three styles, download |
| 5:35–6:25 | Experiment tracking in Weights & Biases |
| 6:25–6:35 | Closing |

---

### 0:00–0:20 — Introduction

**Show:** the repository README or the report title page.

**Say:** "This is GenAI Studio, my Generative AI assignment: four image models — a universal
denoising autoencoder, hard-routed specialists, a soft mixture of experts, and a style-conditioned
face-to-sketch GAN — trained with PyTorch and Optuna, tracked in Weights & Biases, exported to ONNX
and served by a FastAPI backend behind a React interface. I'll start it from scratch with one
command."

### 0:20–1:10 — Application start-up

**Do:** in the terminal, run

```bash
docker compose up --build
```

**Say (while it builds):** "Docker Compose builds two containers. The backend is FastAPI with ONNX
Runtime on the CPU; it mounts the trained models read-only from the `models` folder. The frontend
is the React build served by nginx, which also forwards `/api` to the backend, so the browser talks
to a single origin. The frontend waits until the backend's health check passes."

**Do:** when the logs show both services running, open <http://localhost:5173>.

**Point at:** the status pill **Backend online** in the top bar (green), and the four tabs
**Universal Restoration**, **Hard-Routed Restoration**, **Soft Mixture-of-Experts Restoration**,
**Face-to-Sketch Generator**.

**Do:** click the **System information** button (the info icon at the top right).

**Say:** "The health endpoint reports the runtime — Python, ONNX Runtime and the CPU execution
provider — and all seven models with their model cards, including the measured ONNX-versus-PyTorch
parity." Close the dialog.

### 1:10–2:25 — Universal Restoration (image upload, runtime corruption, restore, download)

**Do:** open **Universal Restoration**. Click **Drop an image or click to upload** and choose the
unseen pet photo.

**Say:** "The image is validated by decoding on the server and resized to 128 × 128, exactly as
in training."

**Do:** under **Corruption** choose **Salt & Pepper**, under **Severity** choose **High**, then
click **Preview corruption**.

**Say:** "The corruption is applied at runtime on the server by the same code that generated the
training data; the seed is shown, so the same corruption can be reproduced."

**Do:** click **Restore**.

**Point at:** the three panels **Original**, **Model input**, **Restored output** with PSNR and
SSIM under them, the **Reconstruction error map** (mean absolute error), and the details card
(model `udae.onnx`, inference time).

**Say:** "One autoencoder, never told which corruption it faces, removes almost all impulses. On
the test set it reaches 23.8 dB over all corrupted inputs, but everything passes through a
4,096-value bottleneck, so even clean images come back at about 25 dB — the limitation that
motivates the next two workspaces."

**Do:** quickly switch **Corruption** to **Occlusion**, tick **Custom parameters**, set 3
rectangles and a larger cover, and click **Restore** again.

**Do:** click **Download result** and show the downloaded file name
(`<photo>_restored-universal.png`) in the browser's download bar.

### 2:25–3:40 — Hard-Routed Restoration (hard routing)

**Do:** open **Hard-Routed Restoration**. Under **Or pick a sample** choose a pet sample. Set
**Corruption** to **Blur**, **Severity** to **Medium**. Keep **Routing** on **Predicted**. Click
**Restore**.

**Point at:** the **Classifier** card: the four probability bars, **Predicted: Blur**,
**Expert used: Blur specialist**, **True: Blur**, the confidence and entropy line, and in the
details card the **Inference time breakdown** (Classifier + Expert).

**Say:** "A CNN classifier predicts the corruption and the argmax selects one independently trained
specialist autoencoder."

**Do:** switch **Routing** to **Oracle** and click **Restore**.

**Say:** "Oracle routing uses the true corruption instead of the prediction. When the classifier is
right both modes give the same output; the report compares them over the whole test set."

**Do:** set **Corruption** to **None** (the sample is sent as clean) and click **Restore**.

**Point at:** the **Identity bypass** note and the output panel badge.

**Say:** "A clean image is routed to the identity branch: no expert runs and the output is exactly
the input, which removes the universal model's 25 dB ceiling for clean images."

**Do (evaluator scenario):** upload the already-corrupted image with **Corruption** = **None** and
click **Restore**.

**Say:** "For an image that arrives already corrupted the true condition is unknown, so only
predicted routing is available — the Oracle button is disabled — and the classifier alone decides."

### 3:40–4:35 — Soft Mixture-of-Experts Restoration (soft expert weights)

**Do:** open **Soft Mixture-of-Experts Restoration**, pick the same sample, set **Corruption** to
**Blur**, **Severity** to **Low**, click **Restore**.

**Point at:** the **Expert weights** card: the four bars **Identity (clean)**, **Salt & Pepper
expert**, **Blur expert**, **Occlusion expert**, the **Main contributor** tag, the sum shown as
"Softmax gate weights, sum = 100.0%", and the row **Output of every branch** with each branch's
image and weight.

**Say:** "Here the gate is a temperature-scaled softmax initialised from the classifier, and the
output is the weighted sum of the identity branch and the three experts, all fine-tuned jointly.
The weights are continuous, so an ambiguous input can be shared between experts instead of being
forced to one."

**Do:** change **Corruption** to **Occlusion**, **High**, click **Restore** and show how the weights
move; then click **Download result**.

### 4:35–5:35 — Face-to-Sketch Generator

**Do:** open **Face-to-Sketch Generator**. Either upload the face photo with **Drop an image or
click to upload**, or click **Use webcam**, then **Capture photo**.

**Say:** "The photo is cropped to a centred square and resized to 128 × 128, as in training."

**Do:** under **Sketch style** choose **Style 1** and click **Generate sketch**. Then choose
**Style 2** and **Style 3** and click **Generate sketch** each time.

**Point at:** **Photo** and **Generated sketch** side by side, the style note under the selector,
and the details card (style index, `generator.onnx`, inference time).

**Say:** "A U-Net generator whose decoder is modulated by a learned style embedding produces the
sketch; the discriminator, a projection PatchGAN, was only needed for training. Only the style
input changes between these three results."

**Do:** click **Download sketch** and show the saved file (`<photo>_sketch-style3.png`).

### 5:35–6:25 — Experiment tracking in Weights & Biases

**Do:** switch to the W&B tab (project `genai-a1`).

**Show, in this order (about 10 s each):**

1. The runs table grouped by **Group**: `task1-optuna`, `task1-final`, `task2-classifier-optuna`,
   `task2-classifier-final`, `task2-specialists-optuna`, `task2-specialist-salt` / `-blur` /
   `-occlusion`, `task3-optuna`, `task3-final`, `task4-optuna`, `task4-final`.
   **Say:** "Every Optuna trial is its own run, with its hyperparameters and per-epoch validation
   metrics."
2. The final Task 1 run `udae`: the loss and per-condition validation PSNR/SSIM charts, then the
   **Overview** with the hyperparameters chosen by Optuna.
3. The logged image panels (input | output | target for the same validation images every few
   epochs) — scroll through the epochs.
4. The **Artifacts** tab of that run: the `best.pt` model checkpoint.
5. The final Task 4 run: the separately logged `train/d_real`, `train/d_fake`, `train/g_adv` and
   `train/g_l1` curves and the validation sample grids.

**Say:** "Hyperparameters, losses, evaluation metrics, checkpoints and visual outputs are all
recorded here, as the brief requires."

### 6:25–6:35 — Closing

**Do:** back in the terminal, press `Ctrl+C` (or run `docker compose down`).

**Say:** "Everything — data preparation, training, Optuna studies, evaluation, ONNX export with
parity checks, the backend, the frontend and the Docker Compose setup — is in the repository linked
in the report. Thank you."

## If something goes wrong while recording

- **Backend degraded** (amber pill) or a workspace shows **Model not available**: a model file is
  missing from `./models`; add it and run `docker compose restart backend`.
- The app shows **Backend offline**: the backend container is still starting; wait for its health
  check, or check `docker compose logs backend`.
- Webcam blocked: say "the webcam is optional" and upload a photo instead.
