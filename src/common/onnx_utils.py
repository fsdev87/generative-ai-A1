"""ONNX export, ONNX Runtime parity checks and model cards (sidecar JSON read by the app)."""
import datetime
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .paths import ROOT


def export_onnx(model, example_inputs, path, input_names, output_names, opset=17, dynamic_batch=True):
    """Export `model` in eval mode on CPU with a dynamic batch axis, then validate the file."""
    import onnx

    model = model.eval().cpu()
    example_inputs = tuple(t.cpu() for t in example_inputs)
    dynamic_axes = {n: {0: "batch"} for n in [*input_names, *output_names]} if dynamic_batch else None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # TorchScript exporter: stable dynamic_axes support for these CNNs. Newer torch defaults to the
    # dynamo exporter (opset 18); older torch has no `dynamo` argument and always uses TorchScript.
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(
            model, example_inputs, str(path),
            input_names=list(input_names), output_names=list(output_names),
            dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True, **extra,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


@torch.no_grad()
def check_parity(model, onnx_path, inputs, input_names, output_names, atol=1e-4):
    """Compare PyTorch (eval, CPU) with ONNX Runtime on the same inputs."""
    import onnxruntime as ort

    model = model.eval().cpu()
    inputs = tuple(t.cpu() for t in inputs)
    reference = model(*inputs)
    reference = reference if isinstance(reference, (tuple, list)) else (reference,)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(list(output_names), {n: t.numpy() for n, t in zip(input_names, inputs)})
    diffs = {n: float(np.abs(r.numpy() - o).max()) for n, r, o in zip(output_names, reference, outputs)}
    return {
        "max_abs_diff": diffs,
        "atol": atol,
        "n_inputs": int(inputs[0].shape[0]),
        "passed": all(d <= atol for d in diffs.values()),
    }


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray, tuple)):
        return list(obj)
    return str(obj)


def write_model_card(onnx_path, **info):
    """Write <name>.json next to the ONNX file (shown by the app's /api/health).

    Pass task, model_config, inputs/outputs, preprocessing, metrics, hyperparameters,
    parity (from check_parity) and wandb_run as keyword arguments.
    """
    onnx_path = Path(onnx_path)
    card = {
        "file": onnx_path.name,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "size_bytes": onnx_path.stat().st_size,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        **info,
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(card, indent=2, default=_jsonable))
    return card
