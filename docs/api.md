# Backend API reference

FastAPI backend in `backend/` that serves the four workspaces through ONNX Runtime (CPU).
The live OpenAPI schema is at `/api/openapi.json` and interactive docs at `/api/docs`; this
page describes the same API with examples for the React frontend.

- Base URL: `http://localhost:8000` (all paths start with `/api`).
- Every `POST` takes **multipart/form-data** (`FormData` in the browser) and returns JSON.
- Every image in a response is a **PNG data URL** (`data:image/png;base64,...`), usable directly
  as `<img src>` and as a download link (`<a href={url} download="restored.png">`).
- Restoration images are 128x128 RGB; the sketch is 128x128 greyscale (or RGB, see below).
- Class order everywhere is `clean, salt, blur, occlusion` (`clean` = identity branch).
- Examples below were recorded with the placeholder models; base64 strings are shortened and
  timings/PSNR values are only illustrative.

## Contents

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, uptime, runtime versions, every model and its model card |
| GET | `/api/options` | Corruption types, levels, custom ranges, styles, limits (for the UI controls) |
| GET | `/api/samples` | Bundled clean sample images |
| GET | `/api/samples/{id}` | A sample's image file |
| POST | `/api/corrupt` | Apply a corruption (runtime corruption preview) |
| POST | `/api/restore/universal` | Task 1, Universal Restoration (`udae.onnx`) |
| POST | `/api/restore/hard` | Task 2, Hard-Routed Restoration (`classifier.onnx` + specialists) |
| POST | `/api/restore/moe` | Task 3, Soft Mixture-of-Experts Restoration (`moe.onnx`) |
| POST | `/api/sketch` | Task 4, Face-to-Sketch Generator (`generator.onnx`) |

## Errors

Every error is JSON with a human-readable `detail` string; show it to the user as is.

```json
{"detail": "p=0.4 is outside the training range [0.02, 0.15]."}
```

| Status | When |
|---|---|
| 400 | Conflicting or missing fields: both or neither of `file`/`sample_id`, level and custom parameters together, oracle routing without a server-side corruption, empty or corrupt image file |
| 404 | Unknown `sample_id` |
| 413 | File larger than 10 MB, or an image above 50 megapixels (decompression bomb) |
| 415 | Not a JPEG, PNG, WEBP or BMP image (checked by decoding, not by the content type) |
| 422 | Invalid value: unknown enum, wrong type, parameter outside the training range. Also has `errors`: `[{"loc": ["body", "level"], "msg": "...", "type": "..."}]` |
| 500 | A model failed during inference or returned an unexpected output |
| 503 | A model file the request needs is missing or broken; `detail` names the file, e.g. `Model file udae.onnx is missing from /models. Add it and restart the backend.` |

## Common request fields

### Image (every POST): exactly one of

| Field | Type | Description |
|---|---|---|
| `file` | file | JPEG, PNG, WEBP or BMP, at most 10 MB. Webcam frames and phone photos are fine (EXIF rotation is applied; greyscale, palette, 16-bit and transparent images become RGB, transparency on white). |
| `sample_id` | string | Id from `GET /api/samples` |

### Corruption (`/api/corrupt` and the three restore endpoints)

| Field | Type | Description |
|---|---|---|
| `corruption` | `clean` \| `salt` \| `blur` \| `occlusion` | Required for `/api/corrupt`. Optional for restore: when given, the server applies it to the preprocessed clean image before restoring; omit it to restore the image as uploaded (e.g. an already corrupted image). |
| `level` | `low` \| `medium` \| `high` | Fixed test severity (salt p 0.03/0.08/0.15; blur (k, sigma) (3, 0.7)/(5, 1.5)/(7, 2.5); occlusion 1/2/3 rectangles covering 10/20/35 %) |
| `p` | float | Custom salt-and-pepper probability, 0.02 to 0.15 |
| `k`, `sigma` | int, float | Custom blur: kernel 3, 5 or 7 and sigma 0.5 to 2.5 |
| `n`, `cover` | int, float | Custom occlusion: 1 to 3 rectangles covering 0.10 to 0.35 of the image |
| `seed` | int | 0 to 2^31-1. Random when omitted; always returned. The same fields + seed give the same image. |

Rules: `clean` takes nothing else; any other type needs **either** `level` **or** all of its
custom parameters (not both, and only its own). The ranges are the training ranges from
`src/data/corruptions.py`; `GET /api/options` returns them so the UI can build its sliders.

For a **sample with no corruption** send `corruption=clean`: the server then knows the image is
clean (true class known, oracle routing works, PSNR is shown). Omit `corruption` only for an
uploaded image whose true condition is unknown.

### Preprocessing (what the models receive)

- Restoration (Tasks 1-3), as in training: RGB, plain bicubic resize to 128x128 (no crop, the
  aspect ratio is not kept), float32 in [0, 1]. A corruption is then applied with
  `src/data/corruptions.py` and the result is rounded to 8 bits, so the returned `input` PNG
  is exactly what the model saw (re-uploading it gives exactly the same result).
- Sketch (Task 4): centre crop to a square, bicubic resize to 128x128, scaled to [-1, 1]
  (`sketch_photo_input` in `backend/app/services/preprocessing.py`; must match Task 4 training).

## Shared response objects

`source`: where the image came from.
```json
{"kind": "upload", "name": "upload.jpg", "format": "JPEG", "width": 400, "height": 300}
```
`width`/`height` are the original size after EXIF rotation; `name` is the file name or sample id.

`corruption` (CorruptionSettings): the applied corruption, `null` when none was applied.

| Field | Description |
|---|---|
| `type` | `clean`, `salt`, `blur`, `occlusion` |
| `level` | The chosen level, or the severity level custom parameters fall into; `null` for clean |
| `custom` | `true` when custom parameters were used |
| `seed` | Request seed (send it back to reproduce) |
| `params` | Display values: salt `{p}`, blur `{k, sigma}`, occlusion `{n, cover}` (cover achieved), clean `{}` |
| `rects` | Occlusion rectangles `[y, x, height, width]` on the 128x128 image (for overlays), else `null` |
| `spec` | Exact spec passed to `apply_spec` (manifest format; salt's `spec.seed` is its noise seed) |

`metrics` (only when the server applied the corruption): quality of the model input and of the
output against the clean reference, computed on the same [0, 1] images.

| Field | Description |
|---|---|
| `psnr_input_db`, `psnr_output_db` | PSNR in dB, capped at 100 dB (identical images) |
| `ssim_input`, `ssim_output` | SSIM, 1 for identical images. Same definition as training (`src.common.losses.ssim`): 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, data range 1, valid region only (no padding), mean over channels and positions. Implemented in NumPy (`backend/app/services/images.py`); matches the PyTorch version to about 1e-7 |

`timing`: `inference_ms` = ONNX Runtime `session.run` time of all models used; `total_ms` =
server time from reading the image to the finished response (decode, preprocessing, corruption,
inference, PNG encoding). Network transfer is not included.

## GET /api/health

Always 200 while the server runs. `status` is `degraded` when any model is missing or broken;
`workspaces.<name>.ready` tells the UI which workspaces can run (show `missing` otherwise).

```json
{
  "status": "ok",
  "version": "1.0.0",
  "started_at": "2026-10-02T21:05:11.402Z",
  "uptime_s": 84.2,
  "runtime": {"python": "3.11.9", "onnxruntime": "1.30.0",
              "available_providers": ["AzureExecutionProvider", "CPUExecutionProvider"],
              "providers": ["CPUExecutionProvider"]},
  "model_dir": "/models",
  "models": [
    {
      "name": "udae", "file": "udae.onnx", "description": "Task 1: universal denoising autoencoder",
      "present": true, "loaded": true, "size_bytes": 9123456, "load_ms": 41.7,
      "inputs": [{"name": "input", "type": "tensor(float)", "shape": ["batch", 3, 128, 128]}],
      "outputs": [{"name": "output", "type": "tensor(float)", "shape": ["batch", 3, 128, 128]}],
      "metadata": {"task": "...", "test_metrics": {"...": "..."}, "parity": {"max_abs_diff": 3e-07}},
      "error": null,
      "warnings": []
    }
  ],
  "workspaces": {
    "universal": {"ready": true, "missing": []},
    "hard": {"ready": false, "missing": ["specialist_blur.onnx"]},
    "moe": {"ready": true, "missing": []},
    "sketch": {"ready": true, "missing": []}
  },
  "samples": {"pets": 6, "faces": 4}
}
```

`models` always lists the 7 contract files in this order: `udae`, `classifier`,
`specialist_salt`, `specialist_blur`, `specialist_occlusion`, `moe`, `generator`. `metadata`
is the sidecar `<name>.json` (model card) or `null`. At startup each model is checked against
the ONNX contract (names, types, one test inference for shapes and value ranges); `error`
explains why a present model is not usable and `warnings` lists non-fatal findings.

## GET /api/options

```json
{
  "classes": ["clean", "salt", "blur", "occlusion"],
  "levels": ["low", "medium", "high"],
  "corruptions": [
    {"type": "clean", "params": [], "levels": {}},
    {"type": "salt", "params": [{"name": "p", "min": 0.02, "max": 0.15, "choices": null}],
     "levels": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}}},
    {"type": "blur", "params": [{"name": "k", "min": 3, "max": 7, "choices": [3, 5, 7]},
                                {"name": "sigma", "min": 0.5, "max": 2.5, "choices": null}],
     "levels": {"low": {"k": 3, "sigma": 0.7}, "medium": {"k": 5, "sigma": 1.5}, "high": {"k": 7, "sigma": 2.5}}},
    {"type": "occlusion", "params": [{"name": "n", "min": 1, "max": 3, "choices": [1, 2, 3]},
                                     {"name": "cover", "min": 0.1, "max": 0.35, "choices": null}],
     "levels": {"low": {"n": 1, "cover": 0.1}, "medium": {"n": 2, "cover": 0.2}, "high": {"n": 3, "cover": 0.35}}}
  ],
  "styles": [1, 2, 3],
  "image_size": 128,
  "max_upload_mb": 10,
  "max_image_megapixels": 50
}
```

## GET /api/samples

Query `category=pets|faces` (optional). Pets are for the restoration workspaces, faces for the
sketch workspace. Display a sample with `<img src={API_BASE + sample.url}>`.

```json
{"samples": [
  {"id": "pets-abyssinian_12", "category": "pets", "name": "Abyssinian_12.jpg", "url": "/api/samples/pets-abyssinian_12"},
  {"id": "faces-face_01", "category": "faces", "name": "face_01.jpg", "url": "/api/samples/faces-face_01"}
]}
```

## GET /api/samples/{id}

The original image file (`image/jpeg`, `image/png`, ...). 404 for an unknown id.

## POST /api/corrupt

Fields: image + corruption fields (`corruption` required). Returns the corrupted image and the
clean preprocessed image, for a preview before restoring.

```bash
curl -F sample_id=pets-abyssinian_12 -F corruption=occlusion -F level=medium -F seed=42 \
     http://localhost:8000/api/corrupt
```
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "clean": "data:image/png;base64,iVBORw0KGgo...",
  "corruption": {
    "type": "occlusion", "level": "medium", "custom": false, "seed": 42,
    "params": {"n": 2, "cover": 0.1997},
    "rects": [[53, 77, 28, 50], [39, 11, 52, 36]],
    "spec": {"type": "occlusion", "n": 2, "cover": 0.1997, "rects": [[53, 77, 28, 50], [39, 11, 52, 36]], "level": "medium"}
  },
  "source": {"kind": "sample", "name": "pets-abyssinian_12", "format": "JPEG", "width": 500, "height": 375}
}
```

## POST /api/restore/universal

Fields: image + optional corruption fields. Response fields shared by all restore endpoints:

| Field | Description |
|---|---|
| `input` | The 128x128 image the model received ("Model input") |
| `output` | The restored image ("Restored output") |
| `reference` | The clean 128x128 image before the server-side corruption ("Original"); `null` without one |
| `corruption`, `metrics`, `source`, `timing` | See above |

```bash
curl -F file=@photo.jpg -F corruption=blur -F k=5 -F sigma=1.2 -F seed=7 \
     http://localhost:8000/api/restore/universal
```
```json
{
  "input": "data:image/png;base64,iVBORw0KGgo...",
  "output": "data:image/png;base64,iVBORw0KGgo...",
  "reference": "data:image/png;base64,iVBORw0KGgo...",
  "corruption": {"type": "blur", "level": "medium", "custom": true, "seed": 7,
                 "params": {"k": 5, "sigma": 1.2}, "rects": null,
                 "spec": {"type": "blur", "k": 5, "sigma": 1.2, "level": "medium"}},
  "metrics": {"psnr_input_db": 27.41, "psnr_output_db": 30.02, "ssim_input": 0.8113, "ssim_output": 0.8862},
  "source": {"kind": "upload", "name": "photo.jpg", "format": "JPEG", "width": 400, "height": 300},
  "model": "udae.onnx",
  "timing": {"inference_ms": 6.2, "total_ms": 18.3}
}
```

## POST /api/restore/hard

Fields: image + optional corruption fields + `routing_mode`:

- `predicted` (default): the classifier's argmax chooses the expert.
- `oracle`: the true corruption chooses the expert. Only valid when the server applies the
  corruption (`corruption` sent), otherwise **400**. The classifier still runs, so its
  prediction can be compared with the true class.

A `clean` route uses the identity bypass: no expert runs, `output` equals `input`,
`selected_expert` is `identity` and `expert_ms` is 0.

```bash
curl -F sample_id=pets-abyssinian_12 -F corruption=salt -F level=high -F seed=1 \
     -F routing_mode=oracle http://localhost:8000/api/restore/hard
```
```json
{
  "input": "data:image/png;base64,...", "output": "data:image/png;base64,...",
  "reference": "data:image/png;base64,...",
  "corruption": {"type": "salt", "level": "high", "custom": false, "seed": 1, "params": {"p": 0.15},
                 "rects": null, "spec": {"type": "salt", "p": 0.15, "seed": 1016164991, "level": "high"}},
  "metrics": {"psnr_input_db": 12.98, "psnr_output_db": 21.25, "ssim_input": 0.2104, "ssim_output": 0.6537},
  "source": {"kind": "sample", "name": "pets-abyssinian_12", "format": "JPEG", "width": 500, "height": 375},
  "routing_mode": "oracle",
  "probabilities": {"clean": 0.691, "salt": 0.1627, "blur": 0.0551, "occlusion": 0.0912},
  "predicted_class": "clean",
  "true_class": "salt",
  "routed_class": "salt",
  "selected_expert": "specialist_salt",
  "timing": {"inference_ms": 8.4, "total_ms": 15.5, "classifier_ms": 2.2, "expert_ms": 6.2}
}
```

| Field | Description |
|---|---|
| `probabilities` | Classifier softmax, one value per class |
| `predicted_class` | argmax of `probabilities` |
| `true_class` | The server-side corruption type, `null` if unknown |
| `routed_class` | The class that chose the expert (predicted, or true in oracle mode) |
| `selected_expert` | `identity`, `specialist_salt`, `specialist_blur` or `specialist_occlusion` |
| `timing` | `classifier_ms`, `expert_ms`, `inference_ms` (their sum), `total_ms` |

Only the models a request actually uses must exist: with `specialist_blur.onnx` missing, a
request routed to blur returns 503 while clean, salt and occlusion routes still work.

## POST /api/restore/moe

Fields: image + optional corruption fields.

```json
{
  "input": "data:image/png;base64,...", "output": "data:image/png;base64,...",
  "reference": null, "corruption": null, "metrics": null,
  "source": {"kind": "upload", "name": "upload.jpg", "format": "JPEG", "width": 400, "height": 300},
  "model": "moe.onnx",
  "weights": {"clean": 0.3528, "salt": 0.2512, "blur": 0.1837, "occlusion": 0.2123},
  "dominant_branch": "clean",
  "branch_outputs": {"clean": "data:image/png;base64,...", "salt": "data:image/png;base64,...",
                     "blur": "data:image/png;base64,...", "occlusion": "data:image/png;base64,..."},
  "timing": {"inference_ms": 11.9, "total_ms": 22.9}
}
```

`weights` are the gate's softmax weights (sum 1); `clean` is the identity branch ("Identity"),
the others are the salt, blur and occlusion experts. `dominant_branch` is the largest weight
("Main contributor"); `branch_outputs` holds each branch's image for the thumbnails (the
`clean` branch output is the input itself). `output` = sum of weight x branch output.

## POST /api/sketch

| Field | Type | Description |
|---|---|---|
| `file` / `sample_id` | | Face photo (upload, webcam frame or `faces` sample) |
| `style` | int | 1, 2 or 3 (required; sent to the generator as index 0, 1, 2) |

```json
{
  "model": "generator.onnx",
  "style": 2,
  "style_index": 1,
  "photo": "data:image/png;base64,...",
  "sketch": "data:image/png;base64,...",
  "sketch_channels": 1,
  "source": {"kind": "sample", "name": "faces-face_01", "format": "JPEG", "width": 250, "height": 250},
  "timing": {"inference_ms": 35.2, "total_ms": 52.0}
}
```

`photo` is the cropped and resized photo the generator received (show it next to the sketch).
`sketch` is a greyscale PNG when the generator outputs one channel, RGB for three.

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `backend/models` (`/models` in Docker) | The 7 ONNX files and their `<name>.json` sidecars |
| `SAMPLES_DIR` | `backend/app/samples` | `pets/` and `faces/` sample folders |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins, or `*` |
| `MAX_UPLOAD_MB` | `10` | Largest image file |
| `MAX_IMAGE_MEGAPIXELS` | `50` | Largest decoded image |

Models are loaded at startup; restart the backend after adding or replacing model files.

## Running

```bash
# local (repository root; needs backend/requirements.txt)
python backend/scripts/make_dummy_models.py            # placeholder models until the real ones exist (needs torch)
python -m uvicorn app.main:app --app-dir backend --port 8000

# tests
python -m pytest backend/tests -q

# Docker (build context = repository root)
docker build -f backend/Dockerfile -t genai-backend .
docker run -p 8000:8000 -v "$PWD/models:/models:ro" genai-backend
```

The container listens on port 8000, runs as a non-root user and has a `HEALTHCHECK` on
`/api/health`. For Docker Compose: `build: {context: ., dockerfile: backend/Dockerfile}`, mount
the model folder at `/models`, and set `CORS_ORIGINS` to the frontend's origin (not needed when
the frontend proxies `/api` to the backend on the same origin).
