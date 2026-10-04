# Trained models

`docker compose up --build` mounts this folder read-only into the backend container at
`/models`. Put the exported ONNX models here, each with its model card (`<name>.json`, written by
`src/common/onnx_utils.write_model_card` during export). The `.onnx` files are not committed to
Git (`*.onnx` is in `.gitignore`); download them from the GitHub Release
(<https://github.com/fsdev87/generative-ai-A1/releases/latest/download/models.zip>, unzip into this
folder — see the main README) or export them yourself (`python -m src.taskN.export_onnx`, which
writes to `ONNX_DIR`, default `outputs/onnx/`).

| File | Task / workspace | Inputs | Outputs |
|---|---|---|---|
| `udae.onnx` | 1, Universal Restoration | `input` [N,3,128,128] in [0,1] | `output` [N,3,128,128] in [0,1] |
| `classifier.onnx` | 2, Hard-Routed Restoration | `input` | `probs` [N,4] (clean, salt, blur, occlusion) |
| `specialist_salt.onnx` | 2, Hard-Routed Restoration | `input` | `output` |
| `specialist_blur.onnx` | 2, Hard-Routed Restoration | `input` | `output` |
| `specialist_occlusion.onnx` | 2, Hard-Routed Restoration | `input` | `output` |
| `moe.onnx` | 3, Soft Mixture-of-Experts Restoration | `input` | `output`, `weights` [N,4], `branch_outputs` [N,4,3,128,128] |
| `generator.onnx` | 4, Face-to-Sketch Generator | `photo` [N,3,128,128] in [-1,1], `style` int64 [N] | `sketch` [N,1 or 3,128,128] in [-1,1] |

The full contract is in `docs/CONVENTIONS.md` ("ONNX contract"). The expected layout:

```
models/
  README.md
  udae.onnx              udae.json
  classifier.onnx        classifier.json
  specialist_salt.onnx   specialist_salt.json
  specialist_blur.onnx   specialist_blur.json
  specialist_occlusion.onnx  specialist_occlusion.json
  moe.onnx               moe.json
  generator.onnx         generator.json
```

Models are loaded when the backend starts, so restart it after adding or replacing files:

```bash
docker compose restart backend
```

Check the result in the app (status pill and "System information" in the top bar) or at
<http://localhost:8000/api/health>: `status` is `ok` when all seven models are loaded, otherwise
`degraded`, and each model lists why it is missing or failed its contract check. Missing models
never stop the application; only the workspaces that need them answer with an error.

For a quick end-to-end test without trained models, placeholder models that satisfy the contract
can be generated (needs PyTorch; they produce meaningless images, never commit or submit them):

```bash
python backend/scripts/make_dummy_models.py --out models
```
