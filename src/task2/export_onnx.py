"""Export the Task 2 models to ONNX (opset 17, float32, dynamic batch) with parity checks and model cards.

  ONNX_DIR/classifier.onnx              input [N,3,128,128] in [0,1] -> probs [N,4] (softmax, CLASSES order)
  ONNX_DIR/specialist_{salt,blur,occlusion}.onnx   input -> output [N,3,128,128] in [0,1]

Each file is checked against PyTorch (eval mode, CPU) with ONNX Runtime on real test
inputs (default 64 entries spread evenly over the test manifest; for a specialist, over
the entries of its own corruption). Any max absolute difference above 1e-4 deletes the
file and aborts. The sidecar <name>.json model card records the parity result and the
test metrics written by evaluate_classifier (accuracy, macro-F1) and evaluate_routing
(oracle routing = each specialist's PSNR/SSIM on its own corruption), so run both
evaluations first.

    python -m src.task2.export_onnx --smoke
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from src.common.checkpoint import load_checkpoint
from src.common.onnx_utils import check_parity, export_onnx, write_model_card
from src.common.paths import get_dir
from src.data.corruptions import CLASSES
from src.data.pets import ManifestDataset, load_pets

from .common import SPECIALIST_TYPES, TASK, disable_wandb_for_smoke, plain
from .routing import checkpoint_paths, load_models

INPUT_SPEC = {"name": "input", "shape": ["N", 3, 128, 128], "dtype": "float32", "range": [0, 1]}
PREPROCESSING = ("RGB, resized to 128x128 (bicubic), float32 in [0, 1], NCHW. "
                 "No normalisation; the model sees images exactly as during training.")
ROUTING_RULE = ("route = argmax(probs); clean -> identity bypass (input returned unchanged, no expert), "
                "salt/blur/occlusion -> specialist_<type>.onnx")


def parity_inputs(images, manifest, n, types=None):
    """n real test inputs spread evenly over the (optionally filtered) test manifest."""
    dataset = ManifestDataset(images, manifest, types=types)
    idx = np.unique(np.linspace(0, len(dataset) - 1, min(n, len(dataset))).round().astype(int))
    return torch.stack([dataset[int(i)]["input"] for i in idx])


def export_checked(model, inputs, path, output_name):
    export_onnx(model, (inputs[:2],), path, ("input",), (output_name,))
    parity = check_parity(model, path, (inputs,), ("input",), (output_name,))
    if not parity["passed"]:
        path.unlink()
        raise RuntimeError(f"ONNX parity check failed for {path.name}: {parity}; file deleted")
    print(f"[onnx] {path.name}: parity OK on {parity['n_inputs']} test inputs, "
          f"max |diff| {parity['max_abs_diff'][output_name]:.2e}")
    return parity


def read_json(path, producer):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run {producer} first (its test metrics go into the model cards)")
    return json.loads(path.read_text())


def card_metrics_classifier(metrics):
    t = metrics["test"]
    return {"test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
            "test_macro_precision": t["macro_precision"], "test_macro_recall": t["macro_recall"],
            "test_per_class": t["per_class"], "test_n": t["n"],
            "val_accuracy": metrics["val"]["accuracy"], "val_macro_f1": metrics["val"]["macro_f1"]}


def card_metrics_specialist(summary, ctype):
    own = summary["oracle"][ctype]
    return {"test_corruption": ctype, "test_psnr": own["psnr_finite"], "test_ssim": own["ssim"],
            "test_mse": own["mse"], "test_n": own["n"], "test_by_level": own["by_level"],
            "note": "test entries of its own corruption (oracle routing), all three severity levels"}


def export_all(smoke=False, n_parity=64):
    disable_wandb_for_smoke(smoke)
    tables, eval_dir, onnx_dir = get_dir("OUTPUT_DIR", TASK, "tables"), get_dir("OUTPUT_DIR", TASK, "eval"), get_dir("ONNX_DIR")
    clf_metrics = read_json(tables / "classifier_metrics.json", "src.task2.evaluate_classifier")
    routing = read_json(eval_dir / "summary.json", "src.task2.evaluate_routing")
    classifier, specialists = load_models("cpu")
    images, manifests = load_pets(smoke=smoke)
    paths = checkpoint_paths()
    cards = {}

    probs_model = nn.Sequential(classifier, nn.Softmax(dim=1)).eval()
    inputs = parity_inputs(images["test"], manifests["test"], n_parity)
    path = onnx_dir / "classifier.onnx"
    parity = export_checked(probs_model, inputs, path, "probs")
    ckpt = load_checkpoint(paths["classifier"])
    cards["classifier"] = write_model_card(
        path, task="task2-classifier", model_config=plain(classifier.config),
        inputs=[INPUT_SPEC],
        outputs=[{"name": "probs", "shape": ["N", len(CLASSES)], "dtype": "float32",
                  "meaning": "softmax class probabilities", "classes": list(CLASSES)}],
        preprocessing=PREPROCESSING, routing_rule=ROUTING_RULE, metrics=card_metrics_classifier(clf_metrics),
        hyperparameters=plain(ckpt.get("train_config")), best_epoch=ckpt.get("epoch"), parity=parity,
        wandb_run=ckpt.get("wandb_run_url"), smoke=smoke)

    for ctype in SPECIALIST_TYPES:
        inputs = parity_inputs(images["test"], manifests["test"], n_parity, types=[ctype])
        path = onnx_dir / f"specialist_{ctype}.onnx"
        parity = export_checked(specialists[ctype].eval(), inputs, path, "output")
        ckpt = load_checkpoint(paths[ctype])
        cards[ctype] = write_model_card(
            path, task=f"task2-specialist-{ctype}", corruption=ctype, model_config=plain(specialists[ctype].config),
            inputs=[INPUT_SPEC],
            outputs=[{"name": "output", "shape": ["N", 3, 128, 128], "dtype": "float32", "range": [0, 1]}],
            preprocessing=PREPROCESSING, routing_rule=ROUTING_RULE, metrics=card_metrics_specialist(routing, ctype),
            hyperparameters=plain(ckpt.get("train_config")), best_epoch=ckpt.get("epoch"), parity=parity,
            wandb_run=ckpt.get("wandb_run_url"), smoke=smoke)
    print(f"[onnx] wrote {len(cards)} models and model cards to {onnx_dir}")
    return cards


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data (uses the smoke checkpoints)")
    parser.add_argument("--n-parity", type=int, default=64, help="real test inputs per parity check (>= 32)")
    args = parser.parse_args(argv)
    if args.n_parity < 32:
        parser.error("--n-parity must be at least 32 (docs/CONVENTIONS.md)")
    return export_all(args.smoke, args.n_parity)


if __name__ == "__main__":
    main()
