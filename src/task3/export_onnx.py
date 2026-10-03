"""Task 3: export the complete soft-MoE inference pipeline as ONE ONNX graph (ONNX_DIR/moe.onnx).

The graph contains the gate, the temperature (a constant), the softmax, the identity branch,
the three experts and the weighted sum:

    input [N,3,128,128] in [0,1]  ->  output [N,3,128,128], weights [N,4], branch_outputs [N,4,3,128,128]

(branch order identity, salt, blur, occlusion; docs/CONVENTIONS.md). Parity with PyTorch (eval,
float32, CPU) is checked on 40 real test inputs (4 per corruption type and severity) and must
be below 1e-4 for every output, otherwise the script fails. A dynamic-batch check runs batch
sizes 1 and 3. The sidecar ONNX_DIR/moe.json (model card) records tau, the configs, the test
metrics and routing summary of evaluate.py (when available), the parity result and the W&B run.

    python -m src.task3.export_onnx [--checkpoint PATH] [--smoke]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.onnx_utils import check_parity, export_onnx, write_model_card
from src.data.pets import ManifestDataset, load_pets

from .config import checkpoint_dir, onnx_path, output_dir
from .model import BRANCHES, SoftMoERestorer

INPUT_NAMES = ("input",)
OUTPUT_NAMES = ("output", "weights", "branch_outputs")
PER_GROUP = 4  # test entries per (type, level) group: 10 groups -> 40 parity inputs
ATOL = 1e-4


def parity_inputs(smoke=False, per_group=PER_GROUP):
    """Real test inputs, the first `per_group` entries of every (type, level) group of the test manifest."""
    images, manifests = load_pets(smoke=smoke)
    dataset = ManifestDataset(images["test"], manifests["test"])
    chosen = defaultdict(list)
    for i, e in enumerate(dataset.entries):
        key = (e["type"], e["level"])
        if len(chosen[key]) < per_group:
            chosen[key].append(i)
    indices = [i for group in chosen.values() for i in group]
    return torch.stack([dataset[i]["input"] for i in indices])


def dynamic_batch_check(model, path, inputs):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = {}
    for n in (1, 3):
        outs = session.run(list(OUTPUT_NAMES), {"input": inputs[:n].numpy()})
        with torch.no_grad():
            ref = model(inputs[:n])
        result[n] = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, outs))
        expected = [(n, 3, 128, 128), (n, 4), (n, 4, 3, 128, 128)]
        if [tuple(o.shape) for o in outs] != expected:
            raise SystemExit(f"[task3] ONNX output shapes {[o.shape for o in outs]} != {expected} at batch {n}")
        if not np.allclose(outs[1].sum(axis=1), 1.0, atol=1e-5):
            raise SystemExit("[task3] ONNX routing weights do not sum to 1")
    return result


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def export(checkpoint=None, smoke=False):
    checkpoint = checkpoint or checkpoint_dir(smoke) / "best.pt"
    ckpt = load_checkpoint(checkpoint)
    model = build_model(SoftMoERestorer, ckpt).eval()
    inputs = parity_inputs(smoke)
    path = onnx_path(smoke)
    export_onnx(model, (inputs[:2],), path, INPUT_NAMES, OUTPUT_NAMES)
    parity = check_parity(model, path, (inputs,), INPUT_NAMES, OUTPUT_NAMES, atol=ATOL)
    print(f"[task3] parity on {parity['n_inputs']} real test inputs: {parity['max_abs_diff']} (atol {ATOL})")
    if parity["n_inputs"] < 32 or not parity["passed"]:
        raise SystemExit(f"[task3] ONNX parity FAILED: {parity}")
    parity["dynamic_batch_max_abs_diff"] = dynamic_batch_check(model, path, inputs)

    summary = read_json(output_dir(smoke, "eval") / "summary.json")
    train_config = ckpt.get("config") or {}
    card = write_model_card(
        path,
        task="task3_soft_moe",
        description="Soft mixture-of-experts restoration: w = softmax(G(x)/tau); output = w0*x + w1*A_salt(x) "
                    "+ w2*A_blur(x) + w3*A_occlusion(x). Gate and experts initialised from Task 2, trained jointly.",
        model_config=model.config,
        tau=model.config["tau"],
        branches=list(BRANCHES),
        inputs={"input": {"shape": ["N", 3, 128, 128], "dtype": "float32", "range": [0, 1]}},
        outputs={"output": {"shape": ["N", 3, 128, 128], "range": [0, 1]},
                 "weights": {"shape": ["N", 4], "order": list(BRANCHES), "note": "softmax, rows sum to 1"},
                 "branch_outputs": {"shape": ["N", 4, 3, 128, 128], "order": list(BRANCHES),
                                    "note": "branch 0 is the input itself (identity)"}},
        preprocessing="RGB, resized to 128x128 (bicubic), float32 in [0, 1], NCHW; no normalisation",
        metrics=summary["metrics"] if summary else None,
        routing=({k: summary["routing"][k] for k in ("usage", "off_class", "top1_accuracy", "mean_entropy",
                                                     "collapsed", "collapse_reason", "matrix_true_x_branch")}
                 if summary else None),
        validation={"epoch": ckpt.get("epoch"), "stage": ckpt.get("stage"), "metrics": ckpt.get("metrics")},
        hyperparameters=train_config,
        init_source=ckpt.get("init_source"),
        parity=parity,
        wandb_run=ckpt.get("wandb_url"),
        checkpoint=str(checkpoint),
    )
    print(f"[task3] exported {path} ({card['size_bytes'] / 1e6:.1f} MB) and {path.with_suffix('.json')}")
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="default: CKPT_DIR/task3/moe/best.pt")
    parser.add_argument("--smoke", action="store_true", help="smoke checkpoint, synthetic test inputs, ONNX_DIR/smoke")
    args = parser.parse_args(argv)
    return export(args.checkpoint, smoke=args.smoke)


if __name__ == "__main__":
    main()
