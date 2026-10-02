"""Train the corruption classifier of Task 2 (clean / salt / blur / occlusion).

Training data: RuntimeCorruptionDataset over the training images with
BalancedBatchSampler, so every batch holds exactly batch_size / 4 images of each
class and the labels come from the runtime corruption pipeline. Validation: the
deterministic validation manifest (balanced by construction). Loss: multiclass
cross-entropy. AdamW with linear warm-up and per-iteration cosine decay, AMP on CUDA,
model selection and early stopping on validation macro-F1.

Checkpoints: CKPT_DIR/task2/classifier/{last,best}.pt (last.pt every epoch for resume).

    python -m src.task2.train_classifier --smoke
    python -m src.task2.train_classifier --config outputs/task2/task2_classifier_best_config.yaml --resume
"""
import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common.checkpoint import build_model, load_checkpoint, save_checkpoint
from src.common.paths import get_dir
from src.common.tracking import log_artifact
from src.common.utils import EarlyStopping, count_parameters, seed_everything
from src.data.corruptions import CLASSES
from src.data.pets import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset, load_pets
from src.models.classifier import CorruptionClassifier

from .common import (
    TASK, Amp, build_config, classification_metrics, config_diff, confusion_figure,
    disable_wandb_for_smoke, flat_metrics, guard_smoke_overwrite, log_figure, make_loader, materialize,
    memory_format, model_state, new_run_id, param_groups, plain, resolve_workers, setup_device,
    tracked_run, warmup_cosine,
)

DEFAULTS = {
    "seed": 42,
    "epochs": 60,            # full schedule of the final run (Optuna trials use fewer)
    "warmup_epochs": 2,
    "min_lr_ratio": 0.01,    # cosine decays to 1% of the peak learning rate
    "patience": 15,          # early stopping on validation macro-F1
    "batch_size": 64,        # multiple of 4: equal counts of the four classes per batch
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "channels": [32, 64, 128, 256],
    "convs_per_stage": 2,
    "dropout": 0.3,
    "num_workers": None,     # None: 2 on CUDA, 0 on CPU
    "log_cm_every": 5,       # confusion-matrix image interval (epochs)
}
# Tiny model and data so the full pipeline runs on a CPU in seconds
SMOKE = {"epochs": 2, "warmup_epochs": 1, "batch_size": 8, "channels": [4, 8], "convs_per_stage": 1,
         "num_workers": 0, "log_cm_every": 1, "patience": 5}

WANDB_GROUP = {"final": "task2-classifier-final", "optuna": "task2-classifier-optuna"}


def classifier_dir():
    return get_dir("CKPT_DIR", TASK, "classifier")


def load_data(smoke=False):
    """Clean images, manifests and the materialised validation set (shared by all Optuna trials)."""
    images, manifests = load_pets(smoke=smoke)
    val = materialize(ManifestDataset(images["val"], manifests["val"]))
    return {"images": images, "manifests": manifests, "val": val}


def build_classifier(cfg):
    return CorruptionClassifier(channels=tuple(cfg["channels"]), convs_per_stage=cfg["convs_per_stage"],
                                dropout=cfg["dropout"])


def make_train_loader(images, cfg, device, num_workers):
    dataset = RuntimeCorruptionDataset(images, conditions=CLASSES)
    sampler = BalancedBatchSampler(len(dataset), cfg["batch_size"], conditions=CLASSES, seed=cfg["seed"])
    return make_loader(dataset, device, num_workers, batch_sampler=sampler), sampler


@torch.no_grad()
def predict(model, inputs, device, batch_size=256):
    """Logits for a tensor of inputs (float32, eval mode)."""
    model.eval()
    fmt = memory_format(device)
    out = [model(inputs[i:i + batch_size].to(device, memory_format=fmt)).float().cpu()
           for i in range(0, len(inputs), batch_size)]
    return torch.cat(out)


def validate(model, val, device):
    logits = predict(model, val["input"], device)
    labels = val["label"]
    metrics = classification_metrics(labels.numpy(), logits.argmax(1).numpy())
    metrics["loss"] = float(F.cross_entropy(logits, labels))
    return metrics


def train_epoch(model, loader, optimizer, scheduler, amp, device):
    model.train()
    fmt = memory_format(device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True, memory_format=fmt)
        y = batch["label"].to(device, non_blocking=True)
        with amp.autocast():
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        amp.step(loss, optimizer)
        scheduler.step()
        loss_sum += loss.detach() * len(y)
        correct += (logits.argmax(1) == y).sum()
        seen += len(y)
    return {"loss": float(loss_sum) / seen, "accuracy": float(correct) / seen, "images": seen}


def train_classifier(config, trial=None, smoke=False, resume=False, data=None):
    """Train one classifier. Returns {"best_score", "best_epoch", "best_metrics"}.

    trial=None: final run (checkpoints, resume, W&B group task2-classifier-final,
    best.pt as artifact). With an Optuna trial: no checkpoints, validation macro-F1
    reported every epoch for pruning, W&B group task2-classifier-optuna.
    """
    import optuna

    disable_wandb_for_smoke(smoke)
    final = trial is None
    cfg = dict(config)
    ckpt_dir = classifier_dir() if final else None
    state = None
    if final and resume and (ckpt_dir / "last.pt").exists():
        state = load_checkpoint(ckpt_dir / "last.pt")
        diff = config_diff(plain(state["train_config"]), plain(cfg))
        if diff:
            print(f"[classifier] resuming with the checkpoint's config; ignoring differences {diff}")
        cfg = state["train_config"]
        if state.get("finished"):
            print(f"[classifier] {ckpt_dir / 'last.pt'} is a finished run; nothing to do")
            return state["result"]
    if final:
        guard_smoke_overwrite(ckpt_dir, smoke)

    device = setup_device()
    seed_everything(cfg["seed"])
    data = data or load_data(smoke)
    num_workers = resolve_workers(cfg["num_workers"], device)
    loader, sampler = make_train_loader(data["images"]["train"], cfg, device, num_workers)
    steps = len(sampler)
    if steps == 0:
        raise ValueError(f"batch_size {cfg['batch_size']} larger than the training set")

    model = build_classifier(cfg).to(device, memory_format=memory_format(device))
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    scheduler = warmup_cosine(optimizer, cfg["epochs"] * steps, cfg["warmup_epochs"] * steps, cfg["min_lr_ratio"])
    amp = Amp(device)
    stopper = EarlyStopping(patience=cfg["patience"])
    start_epoch, best = 0, {"score": -math.inf, "epoch": -1, "metrics": None}
    run_id = new_run_id()
    if state is not None:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        amp.scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["early_stopping"])
        start_epoch, best, run_id = state["epoch"] + 1, state["best"], state["wandb_run_id"]
        print(f"[classifier] resumed at epoch {start_epoch}")

    name = "classifier-final" if final else f"trial-{trial.number:03d}"
    group = WANDB_GROUP["final" if final else "optuna"]
    print(f"[classifier] {name}: {count_parameters(model) / 1e6:.2f}M parameters, {steps} balanced batches/epoch, "
          f"device {device}, workers {num_workers}")
    with tracked_run(name, group, cfg, run_id=run_id, tags=["smoke"] if smoke else None) as run:
        run.summary["parameters"] = count_parameters(model)
        for epoch in range(start_epoch, cfg["epochs"]):
            sampler.rng = np.random.default_rng([cfg["seed"], epoch])  # reproducible order, also after resume
            t0 = time.time()
            with warnings.catch_warnings():
                # GradScaler skips the very first steps while calibrating its scale
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")
                train = train_epoch(model, loader, optimizer, scheduler, amp, device)
            if not math.isfinite(train["loss"]):
                if trial is not None:
                    raise optuna.TrialPruned(f"non-finite training loss at epoch {epoch}")
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            val = validate(model, data["val"], device)
            score = val["macro_f1"]
            is_best = stopper.step(score)
            seconds = time.time() - t0
            if is_best:
                best = {"score": score, "epoch": epoch, "metrics": val}
            run.log({"epoch": epoch, "train/loss": train["loss"], "train/accuracy": train["accuracy"],
                     "val/loss": val["loss"], **flat_metrics(val, "val"), "lr": optimizer.param_groups[0]["lr"],
                     "epoch_seconds": seconds, "train_images_per_s": train["images"] / seconds,
                     "best/macro_f1": best["score"]})
            print(f"epoch {epoch + 1:3d}/{cfg['epochs']} | train loss {train['loss']:.4f} acc {train['accuracy']:.4f}"
                  f" | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} macro-F1 {score:.4f}"
                  f"{' *' if is_best else ''} | {seconds:.1f}s", flush=True)

            if final:
                if cfg["log_cm_every"] and (epoch + 1) % cfg["log_cm_every"] == 0:
                    log_figure(run, "val/confusion_matrix", confusion_figure(val, f"Validation, epoch {epoch + 1}"))
                if is_best:
                    save_checkpoint(ckpt_dir / "best.pt", model_config=model.config, model_state=model_state(model),
                                    epoch=epoch, metrics=val, train_config=cfg, task="task2-classifier",
                                    smoke=smoke, wandb_run_url=run.url)
                save_checkpoint(ckpt_dir / "last.pt", model_config=model.config, model_state=model_state(model),
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                scaler=amp.scaler.state_dict(), early_stopping=stopper.state_dict(),
                                epoch=epoch, best=best, train_config=cfg, wandb_run_id=run_id, smoke=smoke,
                                finished=False)
            else:
                trial.report(score, epoch)
                if trial.should_prune():
                    run.summary["pruned_at_epoch"] = epoch
                    raise optuna.TrialPruned(f"pruned at epoch {epoch} (macro-F1 {score:.4f})")
            if stopper.should_stop:
                print(f"[classifier] early stopping: no macro-F1 improvement for {cfg['patience']} epochs")
                break

        result = {"best_score": best["score"], "best_epoch": best["epoch"], "best_metrics": best["metrics"]}
        run.summary.update({"best_macro_f1": best["score"], "best_epoch": best["epoch"],
                            "best_accuracy": best["metrics"]["accuracy"]})
        if final:
            best_model = build_model(CorruptionClassifier, ckpt_dir / "best.pt").to(device)
            log_figure(run, "val/confusion_matrix_best",
                       confusion_figure(validate(best_model, data["val"], device),
                                        f"Validation, best epoch {best['epoch'] + 1}"))
            log_artifact(run, ckpt_dir / "best.pt", "task2-classifier", aliases=["best"],
                         metadata={"epoch": best["epoch"], "val_macro_f1": best["score"],
                                   "val_accuracy": best["metrics"]["accuracy"]})
            last = load_checkpoint(ckpt_dir / "last.pt")
            save_checkpoint(ckpt_dir / "last.pt", **{**last, "finished": True, "result": result})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML with config overrides (e.g. the Optuna best config)")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data and model, W&B disabled")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(DEFAULTS, args.config)
    if args.smoke:
        cfg.update(SMOKE)
    cfg = build_config(cfg, overrides={"epochs": args.epochs, "num_workers": args.num_workers})
    result = train_classifier(cfg, smoke=args.smoke, resume=args.resume)
    print(f"[classifier] best validation macro-F1 {result['best_score']:.4f} at epoch {result['best_epoch'] + 1}")
    return result


if __name__ == "__main__":
    main()
