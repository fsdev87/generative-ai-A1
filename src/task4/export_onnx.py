"""Task 4: export the generator from best.pt to ONNX_DIR/generator.onnx and write its model card.

Only the generator is exported: the discriminator is a training component and the
application never runs it. Contract (docs/CONVENTIONS.md): opset 17, float32, dynamic
batch, inputs `photo` [N,3,128,128] in [-1,1] and `style` int64 [N] (0/1/2 = Style 1/2/3),
output `sketch` [N,1,128,128] in [-1,1]. Parity against ONNX Runtime is checked on real
test photos covering all three styles, as a batch and as a single image; the export fails
loudly (file renamed to *.onnx.failed) if the max absolute difference exceeds 1e-4.

    python -m src.task4.export_onnx
    python -m src.task4.export_onnx --smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.onnx_utils import check_parity, export_onnx, write_model_card
from src.common.paths import get_dir
from src.data.fs2k import IMAGE_SIZE, NUM_STYLES, SKETCH_CHANNELS, STYLE_NAMES, load_fs2k
from src.task4.models import UNetGenerator

from .config import checkpoint_dir, output_dir

INPUT_NAMES, OUTPUT_NAMES = ("photo", "style"), ("sketch",)
PREPROCESSING = ("EXIF-upright RGB, transparency flattened onto white, centre square crop (side = shorter "
                 "side), bicubic resize to 128x128, float32 scaled to [-1,1], NCHW. Identical to "
                 "src.data.fs2k.preprocess_photo, which the backend mirrors.")
ATOL = 1e-4


def parity_inputs(smoke=False, n=64):
    """n real test pairs spread over the test split, so all three styles are represented."""
    data = load_fs2k(smoke=smoke)["test"]
    idx = np.unique(np.linspace(0, len(data["ids"]) - 1, min(n, len(data["ids"]))).round().astype(int))
    photos = torch.from_numpy(data["photos"][idx]).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    styles = torch.from_numpy(data["styles"][idx]).long()
    return photos, styles


def test_metrics(smoke):
    """Test results from evaluate.py, if it has run (the app shows them in /api/health)."""
    path = output_dir(smoke) / "eval" / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    return {k: summary[k] for k in ("n_test_pairs", "overall", "macro_over_styles", "by_style", "by_source",
                                    "style_sensitivity") if k in summary}


def export(checkpoint=None, smoke=False, n_parity=64):
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_dir(smoke) / "best.pt"
    ckpt = load_checkpoint(ckpt_path)
    generator = build_model(UNetGenerator, ckpt).eval()
    # The app loads ONNX_DIR/generator.onnx; smoke exports go to a subfolder and never replace it
    onnx_path = get_dir("ONNX_DIR", *(["smoke"] if smoke else [])) / "generator.onnx"
    photos, styles = parity_inputs(smoke, n_parity)
    if len(photos) < 32 and not smoke:  # the contract asks for >= 32 real inputs
        raise RuntimeError(f"parity needs at least 32 real test photos, got {len(photos)}")

    export_onnx(generator, (photos[:2], styles[:2]), onnx_path, INPUT_NAMES, OUTPUT_NAMES)
    parity = check_parity(generator, onnx_path, (photos, styles), INPUT_NAMES, OUTPUT_NAMES, atol=ATOL)
    single = check_parity(generator, onnx_path, (photos[:1], styles[:1]), INPUT_NAMES, OUTPUT_NAMES, atol=ATOL)
    parity["single_image"] = single
    parity["passed"] = parity["passed"] and single["passed"]
    if not parity["passed"]:
        failed = onnx_path.with_name(onnx_path.name + ".failed")
        onnx_path.replace(failed)
        raise RuntimeError(f"ONNX parity check FAILED (atol {ATOL}): {parity}; file moved to {failed}")

    card = write_model_card(
        onnx_path,
        task="task4_cgan",
        description="Style-conditioned face-to-sketch generator (pix2pix U-Net with a learned style "
                    "embedding modulating every decoder layer). Only the generator is needed for inference.",
        model_config=ckpt["model_config"],
        inputs=[
            {"name": "photo", "shape": ["N", 3, IMAGE_SIZE, IMAGE_SIZE], "dtype": "float32", "range": [-1, 1]},
            {"name": "style", "shape": ["N"], "dtype": "int64", "range": [0, NUM_STYLES - 1],
             "meaning": dict(enumerate(STYLE_NAMES))},
        ],
        outputs=[{"name": "sketch", "shape": ["N", SKETCH_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
                  "dtype": "float32", "range": [-1, 1]}],
        preprocessing=PREPROCESSING,
        postprocessing="sketch is one greyscale channel in [-1,1]: (sketch + 1) / 2 * 255 for 8-bit display",
        metrics={"test": test_metrics(smoke), "validation_at_best_epoch": ckpt["metrics"]},
        hyperparameters=ckpt.get("config"),
        checkpoint={"path": str(ckpt_path), "epoch": ckpt["epoch"]},
        parity=parity,
        wandb_run=ckpt.get("wandb_url"),
        opset=17,
    )
    print(f"[export] {onnx_path} | parity max |diff| {parity['max_abs_diff']['sketch']:.2e} on "
          f"{parity['n_inputs']} test photos (single image {single['max_abs_diff']['sketch']:.2e}) | "
          f"card {onnx_path.with_suffix('.json')}")
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="explicit checkpoint path (default: CKPT_DIR/task4/cgan/best.pt)")
    parser.add_argument("--parity-samples", type=int, default=64, help="real test photos for the parity check")
    parser.add_argument("--smoke", action="store_true", help="export the smoke checkpoint to ONNX_DIR/smoke/")
    args = parser.parse_args(argv)
    return export(args.checkpoint, smoke=args.smoke, n_parity=args.parity_samples)


if __name__ == "__main__":
    main()
