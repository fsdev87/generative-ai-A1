"""Write small placeholder ONNX models that satisfy the ONNX contract of docs/CONVENTIONS.md.

They let the backend and its tests run before the trained models exist. Each placeholder is a
fixed (untrained) function with exactly the contract's input/output names, dtypes, shapes and
value ranges, a dynamic batch axis and opset 17, plus a sidecar <name>.json marked "dummy".

    udae, specialist_*  fixed filters (box blur or unsharp mask), clamped to [0, 1]
    classifier          softmax of scores from the mean colour: grey -> clean, a red cast -> salt,
                        green -> blur, blue -> occlusion (so tests can choose the predicted class)
    moe                 identity branch + the three specialist filters, gated like the classifier
    generator           edge-based "sketch" whose stroke strength depends on the style

Development only (needs torch). Usage:
    python backend/scripts/make_dummy_models.py                 # writes backend/models/
    python backend/scripts/make_dummy_models.py --out DIR --only generator --sketch-channels 3
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

OPSET = 17
MODEL_NAMES = ("udae", "classifier", "specialist_salt", "specialist_blur", "specialist_occlusion", "moe", "generator")
FILTERS = {"udae": "box3", "specialist_salt": "box3", "specialist_blur": "sharpen", "specialist_occlusion": "box5"}
IMAGE_IO = "float32 [N, 3, 128, 128] in [0, 1]"


class FixedFilter(nn.Module):
    """A fixed per-channel filter with replicated borders, clamped to [0, 1]."""

    def __init__(self, kind: str):
        super().__init__()
        size = 5 if kind == "box5" else 3
        kernel = torch.full((size, size), 1.0 / size**2)
        if kind == "sharpen":  # unsharp mask: 2 * x - box3(x)
            kernel = -kernel
            kernel[1, 1] += 2.0
        self.register_buffer("kernel", kernel.expand(3, 1, size, size).clone())
        self.pad = size // 2

    def forward(self, x):
        padded = F.pad(x, (self.pad,) * 4, mode="replicate")
        return F.conv2d(padded, self.kernel, groups=3).clamp(0.0, 1.0)


def colour_scores(x):
    """Scores in CLASSES order: a neutral image favours clean; a dominant red, green or blue
    channel favours salt, blur or occlusion."""
    rgb = x.mean(dim=(2, 3))  # [N, 3]
    cast = rgb - rgb.mean(dim=1, keepdim=True)
    clean = torch.zeros_like(rgb[:, :1]) + 0.1
    return 20.0 * torch.cat([clean, cast], dim=1)


class Classifier(nn.Module):
    def forward(self, x):
        return torch.softmax(colour_scores(x), dim=1)


class Mixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(
            FixedFilter(FILTERS[f"specialist_{c}"]) for c in ("salt", "blur", "occlusion"))

    def forward(self, x):
        weights = torch.softmax(colour_scores(x) / 4.0, dim=1)  # softer than the classifier
        branches = torch.stack([x] + [expert(x) for expert in self.experts], dim=1)  # [N, 4, 3, H, W]
        output = (weights[:, :, None, None, None] * branches).sum(dim=1)
        return output, weights, branches


class Generator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.stroke = nn.Embedding(3, 1)  # stroke strength per style
        with torch.no_grad():
            self.stroke.weight.copy_(torch.tensor([[2.0], [4.0], [8.0]]))
        sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        self.register_buffer("sobel", torch.stack([sobel, sobel.t()]).unsqueeze(1))  # [2, 1, 3, 3]

    def forward(self, photo, style):
        grey = photo.mean(dim=1, keepdim=True)
        grad = F.conv2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), self.sobel)
        edges = grad.pow(2).sum(dim=1, keepdim=True).sqrt()
        strength = self.stroke(style).view(-1, 1, 1, 1)
        sketch = 1.0 - 2.0 * torch.tanh(strength * edges)  # white paper, dark strokes, in (-1, 1]
        return sketch.repeat(1, 3, 1, 1) if self.channels == 3 else sketch


def build(name: str, sketch_channels: int, batch: int):
    """(module, example inputs, input names, output names, I/O description, what it computes)."""
    image = torch.rand(batch, 3, 128, 128)
    if name in FILTERS:
        return (FixedFilter(FILTERS[name]), (image,), ["input"], ["output"],
                {"input": IMAGE_IO, "output": IMAGE_IO}, f"fixed {FILTERS[name]} filter")
    if name == "classifier":
        return (Classifier(), (image,), ["input"], ["probs"],
                {"input": IMAGE_IO, "probs": "float32 [N, 4] softmax, CLASSES order"}, "mean-colour scores")
    if name == "moe":
        return (Mixture(), (image,), ["input"], ["output", "weights", "branch_outputs"],
                {"input": IMAGE_IO, "output": IMAGE_IO, "weights": "float32 [N, 4] softmax",
                 "branch_outputs": "float32 [N, 4, 3, 128, 128] (identity, salt, blur, occlusion)"},
                "identity + specialist filters, mean-colour gate")
    if name == "generator":
        style = torch.randint(0, 3, (batch,))
        return (Generator(sketch_channels), (image * 2 - 1, style), ["photo", "style"], ["sketch"],
                {"photo": "float32 [N, 3, 128, 128] in [-1, 1]", "style": "int64 [N] in {0, 1, 2}",
                 "sketch": f"float32 [N, {sketch_channels}, 128, 128] in [-1, 1]"}, "Sobel-edge sketch")
    raise ValueError(f"unknown model {name}")


def export(name: str, out_dir: Path, sketch_channels: int = 1) -> Path:
    model, args, input_names, output_names, io, summary = build(name, sketch_channels, batch=2)
    model.eval()
    path = out_dir / f"{name}.onnx"
    with warnings.catch_warnings():
        # torch >= 2.9 defaults to the dynamo exporter, which builds opset 18 and can fail to
        # convert down to the contract's opset 17; the (deprecated) TorchScript exporter
        # selected by dynamo=False writes opset 17 directly.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings("ignore", message="Constant folding")
        torch.onnx.export(model, args, str(path), input_names=input_names, output_names=output_names,
                          dynamic_axes={n: {0: "batch"} for n in input_names + output_names},
                          opset_version=OPSET, dynamo=False)

    # Parity with PyTorch on a different batch size, which also checks the dynamic batch axis.
    _, args, *_ = build(name, sketch_channels, batch=3)
    with torch.no_grad():
        expected = model(*args)
    expected = expected if isinstance(expected, tuple) else (expected,)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(output_names, {n: a.numpy() for n, a in zip(input_names, args)})
    max_diff = max(float(np.abs(e.numpy() - a).max()) for e, a in zip(expected, actual))
    if max_diff > 1e-4:
        raise RuntimeError(f"{name}: ONNX Runtime differs from PyTorch by {max_diff}")

    card = {
        "name": name,
        "dummy": True,
        "description": f"Placeholder from backend/scripts/make_dummy_models.py ({summary}); not a trained model.",
        "onnx": {"file": path.name, "opset": OPSET, "io": io, "dynamic_axes": "batch"},
        "parity": {"max_abs_diff": max_diff, "n_inputs": 3},
        "export_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }
    path.with_suffix(".json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path


def write_dummy_models(out_dir, names=MODEL_NAMES, sketch_channels: int = 1) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    return [export(name, out_dir, sketch_channels) for name in names]


def main():
    parser = argparse.ArgumentParser(description="Write placeholder ONNX models for the backend.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--only", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--sketch-channels", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    for path in write_dummy_models(args.out, args.only, args.sketch_channels):
        print(f"wrote {path} and {path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
