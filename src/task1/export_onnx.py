"""Task 1: export the UDAE best.pt to ONNX_DIR/udae.onnx and write its model card.

Contract (docs/CONVENTIONS.md): opset 17, float32, input `input` [N,3,128,128] in [0,1],
output `output` [N,3,128,128] in [0,1], dynamic batch axis. Parity is checked with ONNX
Runtime against PyTorch (eval mode, CPU) on real test inputs spread over the test
manifest (all conditions and severities), as one batch and as a single image; the
export fails loudly (the file is renamed to *.onnx.failed) if the max absolute
difference exceeds 1e-4. The sidecar udae.json records config, preprocessing, test
metrics (from evaluate.py, if it has run), hyperparameters, parity and the W&B run.

    python -m src.task1.export_onnx
    python -m src.task1.export_onnx --smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.onnx_utils import check_parity, export_onnx, write_model_card
from src.common.paths import get_dir
from src.data.pets import ManifestDataset, load_pets
from src.models.autoencoder import ConvAutoencoder

from .config import MAIN_VARIANT, bottleneck_info, checkpoint_dir, output_dir

INPUT_NAMES, OUTPUT_NAMES = ("input",), ("output",)
PREPROCESSING = "RGB, resized to 128x128, float32 in [0,1], NCHW"
ATOL = 1e-4


def parity_inputs(smoke=False, n=64):
    """n real test inputs at evenly spaced manifest entries (cycles through all 10 conditions)."""
    images, manifests = load_pets(smoke=smoke)
    dataset = ManifestDataset(images["test"], manifests["test"])
    idx = np.unique(np.linspace(0, len(dataset) - 1, min(n, len(dataset))).round().astype(int))
    return torch.stack([dataset[int(i)]["input"] for i in idx])


def test_metrics(variant, smoke):
    path = output_dir(variant, smoke) / "eval" / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    return {
        "split": f"official test manifest ({summary['n_test_entries']} entries)",
        "by_type": {r["type"]: {k: r[k] for k in ("psnr", "ssim", "score", "n")} for r in summary["by_type"]},
        "by_level": {r["level"]: {k: r[k] for k in ("psnr", "ssim", "n")} for r in summary["by_level"]},
        "diagnostics": summary["diagnostics"],
    }


def export(variant=MAIN_VARIANT, checkpoint=None, smoke=False, n_parity=64):
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_dir(variant, smoke) / "best.pt"
    ckpt = load_checkpoint(ckpt_path)
    model = build_model(ConvAutoencoder, ckpt).eval()
    # ONNX files live directly in ONNX_DIR (the app loads udae.onnx); smoke exports never replace them
    onnx_path = get_dir("ONNX_DIR", *(["smoke"] if smoke else [])) / f"{variant}.onnx"
    inputs = parity_inputs(smoke, n_parity)
    if len(inputs) < 32:
        raise RuntimeError(f"parity needs at least 32 real inputs, got {len(inputs)}")

    export_onnx(model, (inputs[:2],), onnx_path, INPUT_NAMES, OUTPUT_NAMES)
    parity = check_parity(model, onnx_path, (inputs,), INPUT_NAMES, OUTPUT_NAMES, atol=ATOL)
    single = check_parity(model, onnx_path, (inputs[:1],), INPUT_NAMES, OUTPUT_NAMES, atol=ATOL)
    parity["single_image"] = single
    parity["passed"] = parity["passed"] and single["passed"]
    if not parity["passed"]:
        failed = onnx_path.with_name(onnx_path.name + ".failed")
        onnx_path.replace(failed)
        raise RuntimeError(f"ONNX parity check FAILED (atol {ATOL}): {parity}; file moved to {failed}")

    info = bottleneck_info(ckpt["model_config"])
    card = write_model_card(
        onnx_path,
        task="task1_udae",
        description="Universal multi-corruption denoising autoencoder (one model for clean, salt-and-pepper, "
                    "Gaussian blur and occlusion inputs; it is not told the corruption).",
        variant=variant,
        model_config=ckpt["model_config"],
        bottleneck={k: info[k] for k in ("latent_shape", "latent_dim", "compression_ratio", "skip_values")},
        inputs=[{"name": "input", "shape": ["N", 3, 128, 128], "dtype": "float32", "range": [0, 1]}],
        outputs=[{"name": "output", "shape": ["N", 3, 128, 128], "dtype": "float32", "range": [0, 1]}],
        preprocessing=PREPROCESSING,
        postprocessing="output is the restored RGB image in [0,1] (sigmoid); multiply by 255 for 8-bit",
        metrics={"test": test_metrics(variant, smoke), "validation_at_best_epoch": ckpt["metrics"]},
        hyperparameters=ckpt.get("config"),
        checkpoint={"path": str(ckpt_path), "epoch": ckpt["epoch"]},
        parity=parity,
        wandb_run=ckpt.get("wandb_url"),
        opset=17,
    )
    diff = parity["max_abs_diff"]["output"]
    print(f"[export] {onnx_path} | parity max |diff| {diff:.2e} on {parity['n_inputs']} inputs "
          f"(single image {single['max_abs_diff']['output']:.2e}) | card {onnx_path.with_suffix('.json')}")
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=MAIN_VARIANT, help="checkpoint CKPT_DIR/task1/<variant>/best.pt; "
                                                                 "the file is ONNX_DIR/<variant>.onnx")
    parser.add_argument("--checkpoint", help="explicit checkpoint path")
    parser.add_argument("--parity-samples", type=int, default=64, help="real test inputs for the parity check")
    parser.add_argument("--smoke", action="store_true", help="export the smoke checkpoint to ONNX_DIR/smoke/")
    args = parser.parse_args(argv)
    return export(args.variant, args.checkpoint, smoke=args.smoke, n_parity=args.parity_samples)


if __name__ == "__main__":
    main()
