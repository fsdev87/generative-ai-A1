"""Task 1: train the universal multi-corruption denoising autoencoder (UDAE).

One ConvAutoencoder learns to restore clean, salt-and-pepper, blurred and occluded
inputs without being told which corruption was applied. Training inputs are
corrupted at load time by RuntimeCorruptionDataset (a fresh condition, uniform over
the four, every time an image is loaded); validation uses the fixed val manifest.
`train_model` is shared by this CLI and the Optuna search (src/task1/optuna_search.py).

    python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --resume
    python -m src.task1.train --config $OUTPUT_DIR/task1/best_config.yaml --skip-resolutions 16 --resume
    python -m src.task1.train --smoke

Checkpoints: CKPT_DIR/task1/<variant>/last.pt (every epoch, full state for --resume)
and best.pt (best validation restoration_score). See docs/task1_notes.md.
"""
import argparse
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.losses import RestorationLoss, ssim
from src.common.metrics import psnr, restoration_score
from src.common.tracking import image_grid, init_run, log_artifact
from src.common.utils import EarlyStopping, count_parameters, get_device, seed_everything
from src.data.corruptions import CLASSES
from src.data.pets import ManifestDataset, RuntimeCorruptionDataset, load_pets
from src.models.autoencoder import ConvAutoencoder

from .config import (
    DEFAULTS, bottleneck_info, checkpoint_dir, model_config, resolve_config, resolve_num_workers, variant_name,
)

FINAL_GROUP = "task1-final"
SAMPLES_PER_CLASS = 2  # fixed validation entries per condition shown in the W&B image grid
VAL_BATCH = 128
# Changing these on --resume does not affect the trained model, so no warning is printed
RUN_CONTROL_KEYS = {"num_workers", "sample_every"}


class NonFiniteLossError(RuntimeError):
    """The training loss became NaN/inf (diverged run)."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def tensorize(dataset):
    """Materialise a (small, deterministic) ManifestDataset once as tensors."""
    items = [dataset[i] for i in range(len(dataset))]
    return {
        "input": torch.stack([it["input"] for it in items]),
        "target": torch.stack([it["target"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def load_data(smoke=False, device=None):
    """Clean training images and the validation set as tensors (computed once, reused every epoch).

    The 736 validation inputs are deterministic, so corrupting them once instead of every
    epoch saves CPU time; the Optuna search shares one copy across all trials.
    """
    images, manifests = load_pets(smoke=smoke)
    val = tensorize(ManifestDataset(images["val"], manifests["val"]))
    if device is not None and device.type == "cuda":
        val = {k: v.pin_memory() for k, v in val.items()}
    return {"train_images": images["train"], "val": val}


def fixed_sample_indices(labels, per_class=SAMPLES_PER_CLASS):
    """The first `per_class` validation entries of every condition (same entries in every run)."""
    labels = labels.tolist()
    return [i for c in range(len(CLASSES)) for i in [j for j, lab in enumerate(labels) if lab == c][:per_class]]


def make_train_loader(images, config, device, seed):
    workers = resolve_num_workers(config["num_workers"], device)
    if config["batch_size"] > len(images):
        raise ValueError(f"batch_size {config['batch_size']} exceeds the {len(images)} training images")
    generator = torch.Generator().manual_seed(seed)  # shuffling order and worker seeds
    return DataLoader(
        RuntimeCorruptionDataset(images, conditions=CLASSES),
        batch_size=config["batch_size"], shuffle=True, drop_last=True,  # constant batch size for BatchNorm
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=generator,
    )


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def warmup_cosine(warmup_steps, total_steps, min_ratio):
    """LR multiplier per optimizer step: linear warm-up, then cosine decay to min_ratio."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, use_amp, grad_clip):
    """One pass over the training set. Running sums stay on the device: no per-step host sync."""
    model.train()
    sums = torch.zeros(3, device=device)  # loss, l1, ssim (weighted by batch size)
    grad_sum = torch.zeros((), device=device)
    grad_steps = torch.zeros((), device=device)
    n_images = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            output = model(x)
        loss, parts = criterion(output, y)  # L1 in fp32; SSIM always runs in fp32
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # true gradient norm for clipping and logging
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float("inf"))
        scaler.step(optimizer)  # skipped by the scaler if fp16 gradients overflowed
        scaler.update()
        scheduler.step()

        n = x.shape[0]
        sums += torch.stack([loss.detach(), parts["l1"], parts["ssim"]]) * n
        finite = torch.isfinite(norm)
        grad_sum += torch.where(finite, norm, torch.zeros_like(norm))
        grad_steps += finite
        n_images += n
    loss, l1, s = (sums / n_images).tolist()
    if not math.isfinite(loss):
        raise NonFiniteLossError(f"training loss is {loss}")
    return {"loss": loss, "l1": l1, "ssim": s, "grad_norm": (grad_sum / grad_steps.clamp_min(1)).item(),
            "images": n_images}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, inputs, device, use_amp, batch_size=VAL_BATCH):
    """Outputs (float32, on the device) for a CPU tensor of inputs, in batches."""
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            outputs.append(model(x).float().clamp(0, 1))
    return torch.cat(outputs)


@torch.no_grad()
def validate(model, val, device, alpha, use_amp, batch_size=VAL_BATCH):
    """Mean loss, PSNR, SSIM and restoration_score on the validation set, overall and per condition.

    Runs under the same autocast as training (fp16 on CUDA): the val pass happens every
    epoch of every trial. Final test metrics (evaluate.py) are computed in float32.
    """
    model.eval()
    p, s, l1 = [], [], []
    for start in range(0, len(val["label"]), batch_size):
        x = val["input"][start:start + batch_size].to(device, non_blocking=True)
        y = val["target"][start:start + batch_size].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            out = model(x).float().clamp(0, 1)
        p.append(psnr(out, y))
        s.append(ssim(out, y, reduction="none"))
        l1.append((out - y).abs().flatten(1).mean(1))
    p, s, l1 = (torch.cat(v).cpu().numpy() for v in (p, s, l1))
    labels = val["label"].numpy()

    def summary(mask):
        mp, ms = float(p[mask].mean()), float(s[mask].mean())
        return {"psnr": mp, "ssim": ms, "score": restoration_score(mp, ms),
                "loss": alpha * float(l1[mask].mean()) + (1 - alpha) * (1 - ms)}

    metrics = summary(np.ones_like(labels, dtype=bool))
    for c, name in enumerate(CLASSES):
        if (labels == c).any():
            metrics.update({f"{name}/{k}": v for k, v in summary(labels == c).items() if k != "loss"})
    return metrics


def sample_grid(model, val, indices, device, use_amp):
    """Rows of input | output | target for fixed validation entries, as one uint8 image."""
    out = predict(model, val["input"][indices], device, use_amp).cpu()

    def hwc(t):
        return t.permute(1, 2, 0).numpy()

    return image_grid([[hwc(val["input"][i]), hwc(out[k]), hwc(val["target"][i])] for k, i in enumerate(indices)])


# --------------------------------------------------------------------------- #
# Training loop (shared with the Optuna search)
# --------------------------------------------------------------------------- #
def _log_epoch(run, record, grid, sample_caption):
    import wandb

    payload = {"epoch": record["epoch"], "lr": record["lr"]}
    payload.update({f"train/{k}": record[f"train_{k}"] for k in ("loss", "l1", "ssim", "grad_norm")})
    payload.update({f"val/{k[4:]}": v for k, v in record.items() if k.startswith("val_")})
    payload.update({"time/epoch_s": record["epoch_time_s"], "time/train_images_per_s": record["train_images_per_s"]})
    if grid is not None:
        payload["val/samples"] = wandb.Image(grid, caption=sample_caption)
    run.log(payload)


def train_model(config, data, run=None, trial=None, ckpt_dir=None, resume_state=None, device=None, verbose=True):
    """Train one UDAE configuration; returns a summary dict.

    run: W&B run or None. trial: Optuna trial or None; when given, the validation score
    is reported every epoch and optuna.TrialPruned is raised if the pruner says so.
    ckpt_dir: where last.pt / best.pt go (None: no checkpoints, as in Optuna trials).
    resume_state: a loaded last.pt to continue from.
    """
    device = device or get_device()
    use_amp = bool(config["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128x128 inputs: pick the fastest conv kernels once
    start_epoch = resume_state["epoch"] + 1 if resume_state else 1
    seed_everything(config["seed"] + start_epoch - 1)  # a resumed run does not replay epoch 1's corruptions

    mcfg = model_config(config)
    model = ConvAutoencoder(**mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loader = make_train_loader(data["train_images"], config, device, seed=config["seed"] + start_epoch - 1)
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(
        config["warmup_epochs"] * steps_per_epoch, config["epochs"] * steps_per_epoch, config["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = RestorationLoss(alpha=config["alpha"])
    stopper = EarlyStopping(patience=config["patience"])
    history, best_epoch, best_metrics = [], None, None

    if resume_state:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        if use_amp and resume_state.get("scaler_state"):
            scaler.load_state_dict(resume_state["scaler_state"])
        stopper.load_state_dict(resume_state["early_stopping"])
        history = list(resume_state["history"])
        best_epoch, best_metrics = resume_state["best_epoch"], resume_state["best_metrics"]

    info = bottleneck_info(mcfg)
    n_params = count_parameters(model)
    sample_idx = fixed_sample_indices(data["val"]["label"])
    caption = "input | output | target; rows: " + ", ".join(
        f"{CLASSES[int(data['val']['label'][i])]}" for i in sample_idx)
    wandb_id = getattr(run, "id", None) if run is not None else None
    wandb_url = getattr(run, "url", None) if run is not None else None
    if run is not None:
        for prefix in ("train/*", "val/*", "time/*", "lr"):
            run.define_metric(prefix, step_metric="epoch")
        run.summary.update({"n_params": n_params, **{k: v for k, v in info.items() if k != "latent_shape"}})
    if verbose:
        print(f"[task1] device={device} amp={use_amp} params={n_params / 1e6:.2f}M "
              f"latent={'x'.join(map(str, info['latent_shape']))}={info['latent_dim']} "
              f"(compression {info['compression_ratio']:.1f}x) skips={config['skip_resolutions']} "
              f"steps/epoch={steps_per_epoch} epochs {start_epoch}..{config['epochs']}", flush=True)

    stop_reason = None
    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.perf_counter()
        train_stats = train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device,
                                      use_amp, config["grad_clip"])
        train_time = time.perf_counter() - t0
        val_metrics = validate(model, data["val"], device, config["alpha"], use_amp)
        if not math.isfinite(val_metrics["score"]):
            raise NonFiniteLossError(f"validation score is {val_metrics['score']}")
        is_best = stopper.step(val_metrics["score"])
        if is_best:
            best_epoch, best_metrics = epoch, dict(val_metrics)
        record = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_stats.items() if k != "images"},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_time_s": time.perf_counter() - t0,
            "train_images_per_s": train_stats["images"] / train_time,
        }
        history.append(record)
        finished = epoch == config["epochs"] or stopper.should_stop
        if finished:
            stop_reason = "early_stopping" if epoch < config["epochs"] else "max_epochs"

        if ckpt_dir is not None:
            common = {"model_config": mcfg, "config": config, "wandb_run_id": wandb_id, "wandb_url": wandb_url}
            if is_best:  # best.pt before last.pt: a crash in between never leaves best.pt behind last.pt
                save_checkpoint(ckpt_dir / "best.pt", **common, model_state=model.state_dict(), epoch=epoch,
                                metrics=val_metrics)
            save_checkpoint(
                ckpt_dir / "last.pt", **common, model_state=model.state_dict(), epoch=epoch, metrics=val_metrics,
                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict(), early_stopping=stopper.state_dict(), history=history,
                best_epoch=best_epoch, best_metrics=best_metrics, finished=finished, stop_reason=stop_reason,
            )
        if run is not None:
            show = epoch % config["sample_every"] == 0 or finished or epoch == 1
            grid = sample_grid(model, data["val"], sample_idx, device, use_amp) if show else None
            _log_epoch(run, record, grid, caption)
        if verbose:
            print(f"epoch {epoch:3d}/{config['epochs']} | loss {train_stats['loss']:.4f} | val psnr "
                  f"{val_metrics['psnr']:.2f} ssim {val_metrics['ssim']:.4f} score {val_metrics['score']:.4f}"
                  f"{' *' if is_best else ''} | lr {record['lr']:.2e} | {record['epoch_time_s']:.1f}s", flush=True)
        if trial is not None:
            import optuna

            trial.report(val_metrics["score"], step=epoch)
            trial.set_user_attr("epochs_run", epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val", {k: round(v, 5) for k, v in best_metrics.items()})
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (score {val_metrics['score']:.4f})")
        if stopper.should_stop:
            if verbose:
                print(f"[task1] early stopping: no improvement for {config['patience']} epochs", flush=True)
            break

    result = {
        "best_score": stopper.best, "best_epoch": best_epoch, "best_metrics": best_metrics,
        "epochs_run": len(history), "stop_reason": stop_reason, "history": history,
        "n_params": n_params, **info,
    }
    if run is not None:
        run.summary.update({"best/score": stopper.best, "best/epoch": best_epoch,
                            **{f"best/{k}": v for k, v in (best_metrics or {}).items()}})
    if run is not None and ckpt_dir is not None and (ckpt_dir / "best.pt").exists():
        try:
            log_artifact(run, ckpt_dir / "best.pt", name=f"task1-{ckpt_dir.name}", aliases=["best"],
                         metadata={"best_epoch": best_epoch, **(best_metrics or {}), **info})
        except Exception as exc:  # the checkpoint is safe on disk; a failed upload must not lose the run
            print(f"[task1] WARNING: could not upload best.pt to W&B: {exc}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
HYPERPARAMETER_FLAGS = {
    "base_channels": (int, "encoder channels at full resolution (doubled per level, capped at 8x)"),
    "latent_channels": (int, "bottleneck channels; the latent is latent_channels x 8 x 8"),
    "depth": (int, "number of stride-2 encoder stages (4: 128 -> 8)"),
    "dropout": (float, "Dropout2d rate after every conv block"),
    "alpha": (float, "loss weight: alpha * L1 + (1 - alpha) * (1 - SSIM)"),
    "lr": (float, "peak learning rate (AdamW)"),
    "weight_decay": (float, "AdamW decoupled weight decay"),
    "batch_size": (int, "training batch size"),
    "epochs": (int, "number of epochs (length of the cosine schedule)"),
    "warmup_epochs": (float, "linear LR warm-up length in epochs"),
    "min_lr_ratio": (float, "final LR as a fraction of the peak LR"),
    "grad_clip": (float, "max gradient norm (0 disables clipping)"),
    "patience": (int, "early-stopping patience in epochs"),
    "seed": (int, "random seed"),
    "sample_every": (int, "log the validation image grid every N epochs"),
    "num_workers": (int, "DataLoader workers on CUDA machines (always 0 on Windows/CPU)"),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with any of the hyperparameters below")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data, 2 epochs, W&B disabled")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    parser.add_argument("--output-subdir", help="checkpoint folder CKPT_DIR/task1/<name> (default: udae, or "
                                                "udae_skip<r> when skip resolutions are given)")
    parser.add_argument("--run-name", help="W&B run name (default: the output subdir)")
    parser.add_argument("--wandb-group", default=FINAL_GROUP)
    hp = parser.add_argument_group("hyperparameters (override --config and the defaults)")
    for key, (typ, text) in HYPERPARAMETER_FLAGS.items():
        hp.add_argument("--" + key.replace("_", "-"), type=typ, default=argparse.SUPPRESS,
                        help=f"{text} (default {DEFAULTS[key]})")
    hp.add_argument("--skip-resolutions", type=int, nargs="*", default=argparse.SUPPRESS,
                    help="limited skip connections at these encoder resolutions, e.g. 16 or 32 or 128 "
                         "(ablation only; default none)")
    hp.add_argument("--no-amp", dest="amp", action="store_false", default=argparse.SUPPRESS,
                    help="disable fp16 mixed precision on CUDA")
    return parser


@contextmanager
def _wandb_resume_env(run_id):
    """Make wandb.init continue run `run_id` (init_run has no id argument; wandb reads these variables)."""
    keys = ("WANDB_RUN_ID", "WANDB_RESUME")
    old = {k: os.environ.get(k) for k in keys}
    if run_id:
        os.environ.update(WANDB_RUN_ID=run_id, WANDB_RESUME="allow")
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        os.environ["WANDB_MODE"] = "disabled"
    overrides = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}
    config = resolve_config(args.config, overrides, smoke=args.smoke)
    variant = args.output_subdir or variant_name(config["skip_resolutions"])
    ckpt_dir = checkpoint_dir(variant, smoke=args.smoke)
    last = ckpt_dir / "last.pt"

    state = None
    if args.resume and last.exists():
        state = load_checkpoint(last)
        changed = {k for k in config if k not in RUN_CONTROL_KEYS and state["config"].get(k) != config[k]}
        if changed:
            print(f"[task1] --resume: using the checkpoint's settings; ignoring changed {sorted(changed)}")
        config = {**state["config"], **{k: config[k] for k in RUN_CONTROL_KEYS}}
        if state.get("finished"):
            print(f"[task1] {last} is already finished ({state.get('stop_reason')}, best epoch "
                  f"{state['best_epoch']}, val score {state['early_stopping']['best']:.4f}); nothing to do.")
            return {"best_score": state["early_stopping"]["best"], "best_epoch": state["best_epoch"],
                    "epochs_run": state["epoch"], "already_finished": True}
        print(f"[task1] resuming {variant} from epoch {state['epoch'] + 1}")
    elif last.exists():
        print(f"[task1] WARNING: {last} exists and --resume was not given: starting a new run that overwrites it")

    device = get_device()
    data = load_data(smoke=args.smoke, device=device)
    run_config = {**config, "variant": variant, "model_config": model_config(config),
                  **bottleneck_info(model_config(config)), "device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    tags = ["task1", "ablation" if config["skip_resolutions"] else "main"] + (["smoke"] if args.smoke else [])
    with _wandb_resume_env(state.get("wandb_run_id") if state else None):
        run = init_run(args.run_name or variant, args.wandb_group, config=run_config, job_type="train", tags=tags)
    try:
        result = train_model(config, data, run=run, ckpt_dir=ckpt_dir, resume_state=state, device=device)
    finally:
        run.finish()
    print(f"[task1] done: best val score {result['best_score']:.4f} at epoch {result['best_epoch']} "
          f"-> {ckpt_dir / 'best.pt'}")
    return result


if __name__ == "__main__":
    main()
