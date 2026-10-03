"""Extra experiment: soft MoE vs hard routing on two-corruption images (manifests/pets_mixed_manifest.json).

Clearly outside the brief's corruption definitions: every input carries two corruptions
(salt+blur, blur+occlusion, salt+occlusion at medium severity), which no single specialist
was trained for. Systems, all in float32 on the same entries:

    input           the corrupted input itself (reference)
    moe_soft        the trained soft MoE (Task 3)
    moe_hard        the same MoE components with argmax routing (isolates the effect of soft weighting)
    task2_hard      Task 2's classifier + specialists with argmax routing and identity bypass
                    (if the Task 2 checkpoints exist)
    universal       Task 1's universal autoencoder (if CKPT_DIR/task1/udae/best.pt exists)

Writes OUTPUT_DIR/task3/mixed/: mixed_records.csv, mixed_summary.{csv,tex}, mixed_routing.{csv,tex}
(mean MoE weights per combination; share of entries whose two largest weights are exactly the two
corruptions present) and mixed_examples.png.

    python -m src.task3.evaluate_mixed [--smoke]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import build_model
from src.common.evaluation import save_csv, save_latex_table
from src.common.metrics import image_metrics
from src.common.paths import get_dir
from src.common.utils import get_device
from src.data.corruptions import CLASSES
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier

from .config import checkpoint_dir, output_dir, task_dir
from .evaluate import load_moe
from .make_mixed_manifest import MixedManifestDataset, load_mixed
from .model import BRANCHES, EXPERTS, load_task2_checkpoints, missing_task2_checkpoints
from .plots import save_image_rows
from .routing import hard_route

SYSTEM_LABELS = {"input": "Input (no restoration)", "moe_soft": "Soft MoE (T3)", "moe_hard": "MoE, argmax routing",
                 "task2_hard": "Hard routing, predicted (T2)", "universal": "Universal AE (T1)"}


def load_baselines(smoke, device):
    """Optional comparison models: Task 2 hard routing and Task 1's universal AE (None if missing)."""
    task2 = None
    root = get_dir("CKPT_DIR")
    if smoke and missing_task2_checkpoints(root):
        root = task_dir("CKPT_DIR", True, "task2_standin")
    if not missing_task2_checkpoints(root):
        gate_ckpt, expert_ckpts = load_task2_checkpoints(root)
        classifier = build_model(CorruptionClassifier, gate_ckpt).to(device).eval()
        experts = {k: build_model(ConvAutoencoder, expert_ckpts[k]).to(device).eval() for k in EXPERTS}
        task2 = (classifier, experts)
    else:
        print("[mixed] Task 2 checkpoints missing: hard-routing baseline skipped")
    udae_path = get_dir("CKPT_DIR") / "task1" / "udae" / "best.pt"
    universal = build_model(ConvAutoencoder, udae_path).to(device).eval() if udae_path.exists() else None
    if universal is None:
        print(f"[mixed] {udae_path} missing: universal-AE baseline skipped")
    return task2, universal


@torch.no_grad()
def run_systems(model, task2, universal, x):
    output, weights, branches = model(x)
    outputs = {"input": x, "moe_soft": output}
    route = weights.argmax(dim=1)
    outputs["moe_hard"] = branches[torch.arange(len(x), device=x.device), route]
    extra = {"weights": weights, "moe_route": route}
    if task2 is not None:
        classifier, experts = task2
        t2_route = classifier(x).argmax(dim=1)
        outputs["task2_hard"] = hard_route(x, t2_route, experts)
        extra["task2_route"] = t2_route
    if universal is not None:
        outputs["universal"] = universal(x)
    return outputs, extra


def evaluate_mixed(checkpoint=None, smoke=False, batch_size=64, device=None):
    device = device or get_device()
    model, _ = load_moe(checkpoint or checkpoint_dir(smoke) / "best.pt", device)
    task2, universal = load_baselines(smoke, device)
    images, manifest = load_mixed(smoke)
    dataset = MixedManifestDataset(images, manifest)
    combos = dataset.combos
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2 if device.type == "cuda" else 0)
    records = []
    for batch in loader:
        x, y = batch["input"].to(device), batch["target"].to(device)
        outputs, extra = run_systems(model, task2, universal, x)
        metrics = {name: image_metrics(out.clamp(0, 1), y) for name, out in outputs.items()}
        w = extra["weights"].cpu()
        top2 = w.topk(2, dim=1).indices
        for i in range(len(x)):
            present = set(torch.nonzero(batch["components"][i]).squeeze(1).tolist())
            rec = {"entry": len(records), "image_idx": int(batch["image_idx"][i]), "combo": combos[int(batch["combo"][i])],
                   **{f"{n}_{m}": float(v[m][i]) for n, v in metrics.items() for m in ("psnr", "ssim")},
                   **{f"w_{c}": float(w[i, k]) for k, c in enumerate(CLASSES)},
                   "moe_top2_is_components": int(set(top2[i].tolist()) == present),
                   "moe_route": CLASSES[int(extra["moe_route"][i])]}
            if "task2_route" in extra:
                rec["task2_route"] = CLASSES[int(extra["task2_route"][i])]
            records.append(rec)
    df = pd.DataFrame(records)
    systems = [s for s in SYSTEM_LABELS if f"{s}_psnr" in df]
    out = output_dir(smoke, "mixed")
    save_csv(records, out / "mixed_records.csv")

    summary = []
    for combo in [*combos, "all"]:
        sub = df if combo == "all" else df[df["combo"] == combo]
        summary.append({"combo": combo, "n": len(sub), **{f"{s}_{m}": float(sub[f"{s}_{m}"].mean())
                                                          for s in systems for m in ("psnr", "ssim")}})
    save_csv(summary, out / "mixed_summary.csv")
    cols = [("combo", "Input"), ("n", "n")] + [c for s in systems for c in
                                               ((f"{s}_psnr", f"{SYSTEM_LABELS[s]} PSNR"), (f"{s}_ssim", "SSIM"))]
    save_latex_table(summary, cols, out / "mixed_summary.tex",
                     caption="Extra experiment beyond the brief: two corruptions per image (medium severity each).",
                     label="tab:task3_mixed",
                     fmt={c: ("{:.4f}" if c.endswith("ssim") else "{:.2f}") for c, _ in cols[2:]})

    routing = []
    for combo in combos:
        sub = df[df["combo"] == combo]
        row = {"combo": combo, "n": len(sub), **{f"w_{c}": float(sub[f"w_{c}"].mean()) for c in CLASSES},
               "top2_is_components": float(sub["moe_top2_is_components"].mean())}
        if "task2_route" in sub:
            row.update({f"task2_to_{c}": float((sub["task2_route"] == c).mean()) for c in CLASSES})
        routing.append(row)
    save_csv(routing, out / "mixed_routing.csv")
    save_latex_table(routing, [("combo", "Input"), ("n", "n"), *[(f"w_{c}", b) for c, b in zip(CLASSES, BRANCHES)],
                               ("top2_is_components", "Top-2 = parts")], out / "mixed_routing.tex",
                     caption="Mean soft-MoE routing weights on two-corruption inputs; last column: share of "
                             "entries whose two largest weights are exactly the two corruptions present.",
                     label="tab:task3_mixed_routing",
                     fmt={k: "{:.3f}" for k in [f"w_{c}" for c in CLASSES] + ["top2_is_components"]})

    # Two typical examples per combination (closest to the combination's median soft-MoE SSIM)
    rows, labels = [], []
    for combo in combos:
        sub = df[df["combo"] == combo]
        median = sub["moe_soft_ssim"].median()
        for e in sub.iloc[(sub["moe_soft_ssim"] - median).abs().argsort().to_numpy()[:2]]["entry"]:
            item = dataset[int(e)]
            x = item["input"][None].to(device)
            outputs, _ = run_systems(model, task2, universal, x)
            rows.append([outputs[s][0].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() for s in systems]
                        + [item["target"].permute(1, 2, 0).numpy()])
            r = df.iloc[int(e)]
            labels.append(f"{combo}\nw=" + "/".join(f"{r[f'w_{c}']:.2f}" for c in CLASSES))
    save_image_rows(rows, [SYSTEM_LABELS[s] for s in systems] + ["Target"], labels, out / "mixed_examples.png",
                    title="Two-corruption inputs (extra experiment); w = identity/salt/blur/occlusion")
    for row in summary:
        print(f"[mixed] {row['combo']:15s} " + " | ".join(f"{s} {row[f'{s}_psnr']:.2f} dB" for s in systems))
    print(f"[mixed] wrote {out}")
    return {"summary": summary, "routing": routing}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="default: CKPT_DIR/task3/moe/best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return evaluate_mixed(args.checkpoint, smoke=args.smoke, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
