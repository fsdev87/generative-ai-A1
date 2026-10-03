"""Task 3: train the soft mixture-of-experts restorer in two stages.

    stage 1  warm-up   (warmup_epochs): experts frozen (no gradients, eval-mode BatchNorm),
                       only the gate (initialised from the Task 2 classifier) trains.
    stage 2  joint     (joint_epochs): experts unfrozen; gate at lr, experts at
                       lr * expert_lr_scale, cosine decay; experts keep their BatchNorm statistics.

Loss on the final reconstruction, with exactly class-balanced batches (BalancedBatchSampler):
    L = lambda_l1 * L1 + lambda_ssim * (1 - SSIM) + lambda_ce * CE(G(x), y) + lambda_balance * L_balance

`train_moe` is shared by this CLI and the Optuna search (src/task3/optuna_search.py).

    python -m src.task3.train --config $OUTPUT_DIR/task3/best_config.yaml --resume
    python -m src.task3.train --smoke

Checkpoints: CKPT_DIR/task3/moe/last.pt (every epoch, full state incl. the stage, for --resume)
and best.pt (best validation score). See docs/task3_notes.md.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.losses import ssim
from src.common.metrics import psnr, restoration_score
from src.common.paths import get_dir
from src.common.tracking import image_grid, init_run, log_artifact
from src.common.utils import EarlyStopping, count_parameters, get_device, seed_everything
from src.data.corruptions import CLASSES
from src.data.pets import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset, load_pets

from .config import (
    DEFAULTS, FINAL_GROUP, RUN_CONTROL_KEYS, checkpoint_dir, resolve_config, resolve_num_workers, task_dir,
)
from .model import EXPERTS, SoftMoERestorer, load_task2_checkpoints, missing_task2_checkpoints, write_standin_task2
from .routing import balance_loss, detect_collapse, flat_routing_stats, moe_loss, routing_stats

# Per-image PSNR is capped at this value inside the validation objective only. restoration_score
# divides PSNR by 40 dB, i.e. treats 40 dB like a perfect SSIM of 1; without the cap, clean inputs
# routed (almost) exactly to the identity reach 60-100 dB and would dominate the mean PSNR.
OBJECTIVE_PSNR_CAP_DB = 40.0
SAMPLES_PER_CLASS = 2  # fixed validation entries per condition in the W&B image grid
VAL_BATCH = 64


class NonFiniteLossError(RuntimeError):
    """The training loss became NaN/inf (diverged run)."""


# --------------------------------------------------------------------------- #
# Data and initialisation
# --------------------------------------------------------------------------- #
def tensorize(dataset):
    """Materialise the deterministic validation set once as tensors (reused every epoch and trial)."""
    items = [dataset[i] for i in range(len(dataset))]
    return {
        "input": torch.stack([it["input"] for it in items]),
        "target": torch.stack([it["target"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
        "level": [it["level"] for it in items],
    }


def load_data(smoke=False):
    images, manifests = load_pets(smoke=smoke)
    return {"train_images": images["train"], "val": tensorize(ManifestDataset(images["val"], manifests["val"]))}


def load_init(smoke=False):
    """The Task 2 checkpoints the MoE starts from: (gate_ckpt, expert_ckpts, source description).

    Smoke runs fall back to randomly initialised stand-ins when Task 2 is not trained yet.
    """
    root = get_dir("CKPT_DIR")
    if smoke and missing_task2_checkpoints(root):
        root = task_dir("CKPT_DIR", True, "task2_standin")
        write_standin_task2(root)
        print(f"[task3] smoke: Task 2 checkpoints missing, using random stand-ins in {root}")
    gate_ckpt, expert_ckpts = load_task2_checkpoints(root)
    source = {"ckpt_dir": str(root), "standin": bool(gate_ckpt.get("standin")),
              "epochs": {"classifier": gate_ckpt.get("epoch"), **{k: expert_ckpts[k].get("epoch") for k in EXPERTS}}}
    return gate_ckpt, expert_ckpts, source


def make_train_loader(images, config, device, epoch):
    """Exactly class-balanced batches; order and corruptions depend only on (seed, epoch), so a resumed
    run sees the same data as an uninterrupted one."""
    if config["batch_size"] > len(images):
        raise ValueError(f"batch_size {config['batch_size']} exceeds the {len(images)} training images")
    workers = resolve_num_workers(config["num_workers"], device)
    return DataLoader(
        RuntimeCorruptionDataset(images, conditions=CLASSES),
        batch_sampler=BalancedBatchSampler(len(images), config["batch_size"], seed=config["seed"] + epoch),
        num_workers=workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"] + epoch),  # worker seeds
    )


def fixed_sample_indices(labels, per_class=SAMPLES_PER_CLASS):
    """The first `per_class` validation entries of every condition (same entries in every run)."""
    labels = labels.tolist()
    return [i for c in range(len(CLASSES)) for i in [j for j, lab in enumerate(labels) if lab == c][:per_class]]


# --------------------------------------------------------------------------- #
# Stages and optimisation
# --------------------------------------------------------------------------- #
def stage_of(epoch, config):
    """Epochs are 1-based: 1..warmup_epochs are the warm-up, the rest joint fine-tuning."""
    return "warmup" if epoch <= config["warmup_epochs"] else "joint"


def apply_stage(model, stage, config):
    model.set_expert_mode("frozen" if stage == "warmup" else
                          "frozen_bn" if config["freeze_expert_bn"] else "train")


def cosine_factor(total_steps, min_ratio):
    def factor(step):
        progress = min(1.0, step / max(1, total_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def build_optimizer(model, stage, config, steps_per_epoch):
    """Warm-up: AdamW on the gate only, constant lr_warmup. Joint: two parameter groups
    (gate lr, experts lr * expert_lr_scale), cosine decay per step to min_lr_ratio."""
    wd = config["weight_decay"]
    if stage == "warmup":
        optimizer = torch.optim.AdamW(model.gate.parameters(), lr=config["lr_warmup"], weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
        optimizer = torch.optim.AdamW([
            {"params": list(model.gate.parameters()), "lr": config["lr"]},
            {"params": list(model.experts.parameters()), "lr": config["lr"] * config["expert_lr_scale"]},
        ], weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, cosine_factor(config["joint_epochs"] * steps_per_epoch, config["min_lr_ratio"]))
    return optimizer, scheduler


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _to_device(t, device, channels_last):
    t = t.to(device, non_blocking=True)
    return t.contiguous(memory_format=torch.channels_last) if channels_last else t


def train_one_epoch(model, loader, optimizer, scheduler, scaler, config, device, use_amp, channels_last):
    """One pass over the training set. Running sums stay on the device: no per-step host sync."""
    model.train()
    params = [p for g in optimizer.param_groups for p in g["params"]]
    sums = torch.zeros(5, device=device)  # loss, l1, ssim, ce, balance (weighted by batch size)
    grad_sum = torch.zeros((), device=device)
    grad_steps = torch.zeros((), device=device)
    n_images = 0
    for batch in loader:
        x = _to_device(batch["input"], device, channels_last)
        y = _to_device(batch["target"], device, channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            output, weights, _, logits = model.forward_with_logits(x)
        loss, parts = moe_loss(output, y, weights, logits, labels, config)  # computed in float32
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # true gradient norm for clipping and logging
        norm = torch.nn.utils.clip_grad_norm_(params, config["grad_clip"] if config["grad_clip"] > 0 else float("inf"))
        scaler.step(optimizer)  # skipped by the scaler if fp16 gradients overflowed
        scaler.update()
        scheduler.step()

        n = x.shape[0]
        sums += torch.stack([loss.detach(), parts["l1"], parts["ssim"], parts["ce"], parts["balance"]]) * n
        finite = torch.isfinite(norm)
        grad_sum += torch.where(finite, norm, torch.zeros_like(norm))
        grad_steps += finite
        n_images += n
    loss, l1, s, ce, bal = (sums / n_images).tolist()
    if not math.isfinite(loss):
        raise NonFiniteLossError(f"training loss is {loss}")
    return {"loss": loss, "l1": l1, "ssim": s, "ce": ce, "balance": bal,
            "grad_norm": (grad_sum / grad_steps.clamp_min(1)).item(), "images": n_images}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def objective_score(psnr_values, ssim_values, cap=OBJECTIVE_PSNR_CAP_DB):
    """restoration_score with per-image PSNR capped at 40 dB (the validation objective of Task 3)."""
    return restoration_score(float(np.minimum(psnr_values, cap).mean()), float(np.mean(ssim_values)))


@torch.no_grad()
def predict(model, inputs, device, use_amp, channels_last=False, batch_size=VAL_BATCH):
    """(output, weights, logits) for a CPU tensor of inputs; outputs on CPU, float32."""
    model.eval()
    outs, ws, ls = [], [], []
    for start in range(0, len(inputs), batch_size):
        x = _to_device(inputs[start:start + batch_size], device, channels_last)
        with _autocast(device, use_amp):
            output, weights, _, logits = model.forward_with_logits(x)
        outs.append(output.float().clamp(0, 1).cpu())
        ws.append(weights.float().cpu())
        ls.append(logits.float().cpu())
    return torch.cat(outs), torch.cat(ws), torch.cat(ls)


@torch.no_grad()
def validate(model, val, device, config, use_amp, channels_last=False):
    """Loss terms, PSNR/SSIM, the objective and routing statistics on the validation manifest.

    Runs under the training autocast (fp16 on CUDA); final test metrics (evaluate.py) are float32.
    """
    output, weights, logits = predict(model, val["input"], device, use_amp, channels_last)
    target, labels = val["target"], val["label"]
    p = psnr(output, target).numpy()
    s = ssim(output, target, reduction="none").numpy()
    l1 = (output - target).abs().flatten(1).mean(1).numpy()
    ce_logits = logits if config["ce_input"] == "logits" else logits / config["tau"]
    ce = torch.nn.functional.cross_entropy(ce_logits, labels, reduction="none").numpy()
    lab = labels.numpy()

    metrics = {
        "score": objective_score(p, s),
        "restoration_score": restoration_score(float(p.mean()), float(s.mean())),  # uncapped, for reference
        "psnr": float(p.mean()), "psnr_capped40": float(np.minimum(p, OBJECTIVE_PSNR_CAP_DB).mean()),
        "ssim": float(s.mean()), "l1": float(l1.mean()), "ce": float(ce.mean()),
        "balance": float(balance_loss(weights)),
    }
    metrics["loss"] = (config["lambda_l1"] * metrics["l1"] + config["lambda_ssim"] * (1 - metrics["ssim"])
                       + config["lambda_ce"] * metrics["ce"] + config["lambda_balance"] * metrics["balance"])
    corrupted = lab != 0
    if corrupted.any():
        metrics["score_corrupted"] = objective_score(p[corrupted], s[corrupted])
    for c, name in enumerate(CLASSES):
        if (lab == c).any():
            metrics[f"{name}/psnr"] = float(p[lab == c].mean())
            metrics[f"{name}/ssim"] = float(s[lab == c].mean())
    stats = routing_stats(weights.numpy(), lab)
    return metrics, stats


def sample_grid(model, val, indices, device, use_amp, channels_last=False):
    """Rows: input | output | target | |error| | salt | blur | occlusion expert outputs; caption has the weights."""
    model.eval()
    x = _to_device(val["input"][indices], device, channels_last)
    with torch.no_grad(), _autocast(device, use_amp):
        output, weights, branches = model(x)
    output, weights, branches = output.float().clamp(0, 1).cpu(), weights.float().cpu(), branches.float().cpu()

    def hwc(t):
        return t.permute(1, 2, 0).numpy()

    rows, lines = [], []
    for k, i in enumerate(indices):
        target = val["target"][i]
        error = (output[k] - target).abs().mean(0).mul(2).clamp(0, 1).numpy()  # x2: errors up to 0.5 visible
        rows.append([hwc(val["input"][i]), hwc(output[k]), hwc(target), error,
                     *[hwc(branches[k, b]) for b in range(1, len(CLASSES))]])
        lines.append(f"{CLASSES[int(val['label'][i])]}: w=" + "/".join(f"{w:.2f}" for w in weights[k].tolist()))
    caption = ("input | output | target | |error| x2 | salt | blur | occlusion expert; "
               "weights identity/salt/blur/occlusion: " + "; ".join(lines))
    return image_grid(rows), caption


# --------------------------------------------------------------------------- #
# Training loop (shared with the Optuna search)
# --------------------------------------------------------------------------- #
def _log_epoch(run, record, stats, grid=None):
    import wandb

    from .plots import routing_heatmap_array

    payload = {"epoch": record["epoch"], "stage": int(record["stage"] == "joint"),
               "lr/gate": record["lr_gate"], "lr/experts": record["lr_experts"],
               "time/epoch_s": record["epoch_time_s"], "time/train_images_per_s": record["train_images_per_s"]}
    payload.update({f"train/{k[6:]}": v for k, v in record.items() if k.startswith("train_")})
    payload.update({f"val/{k[4:]}": v for k, v in record.items() if k.startswith("val_")})
    payload.update({k: v for k, v in flat_routing_stats(stats).items()})
    payload["routing/collapsed"] = int(record["collapsed"])
    payload["routing/heatmap"] = wandb.Image(
        routing_heatmap_array(stats["matrix"], title=f"epoch {record['epoch']} ({record['stage']})"),
        caption="mean routing weight: rows = true input class, columns = branch")
    if grid is not None:
        payload["val/samples"] = wandb.Image(grid[0], caption=grid[1])
    run.log(payload)


def train_moe(config, trial=None, data=None, init=None, run=None, ckpt_dir=None, resume_state=None,
              device=None, smoke=False, verbose=True):
    """Train one MoE configuration through both stages; returns a summary dict.

    trial: Optuna trial or None. When given, the validation score is reported every epoch;
      the trial is pruned (optuna.TrialPruned, reason in the user attribute "prune_reason")
      on routing collapse, on a diverged loss, or when the median pruner says so.
    data / init: preloaded data (load_data) and Task 2 checkpoints (load_init); loaded here if None.
    run: W&B run or None. ckpt_dir: where last.pt / best.pt go (None: no checkpoints, as in trials).
    resume_state: a loaded last.pt to continue from (its stage, optimizer and schedule included).
    """
    device = device or get_device()
    use_amp = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128x128 inputs: pick the fastest conv kernels once
    data = data or load_data(smoke)
    total_epochs = config["warmup_epochs"] + config["joint_epochs"]

    if resume_state:
        model = SoftMoERestorer(**resume_state["model_config"])
        model.load_state_dict(resume_state["model_state"])
        source = resume_state.get("init_source")
    else:
        gate_ckpt, expert_ckpts, source = init or load_init(smoke)
        model = SoftMoERestorer.from_checkpoints(gate_ckpt, expert_ckpts, tau=config["tau"])
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    start_epoch = resume_state["epoch"] + 1 if resume_state else 1
    stopper = EarlyStopping(patience=config["patience"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_epoch, best_metrics = [], None, None
    optimizer = scheduler = None
    current_stage = None
    steps_per_epoch = len(data["train_images"]) // config["batch_size"]
    if resume_state:
        stopper.load_state_dict(resume_state["early_stopping"])
        history = list(resume_state["history"])
        best_epoch, best_metrics = resume_state["best_epoch"], resume_state["best_metrics"]
        if use_amp and resume_state.get("scaler_state"):
            scaler.load_state_dict(resume_state["scaler_state"])
        if start_epoch <= total_epochs and stage_of(start_epoch, config) == resume_state["stage"]:
            current_stage = resume_state["stage"]
            apply_stage(model, current_stage, config)
            optimizer, scheduler = build_optimizer(model, current_stage, config, steps_per_epoch)
            optimizer.load_state_dict(resume_state["optimizer_state"])
            scheduler.load_state_dict(resume_state["scheduler_state"])

    sample_idx = fixed_sample_indices(data["val"]["label"])
    wandb_id = getattr(run, "id", None) if run is not None else None
    wandb_url = getattr(run, "url", None) if run is not None else None
    n_params = {"gate": count_parameters(model.gate), "experts": count_parameters(model.experts)}
    if run is not None:
        for prefix in ("train/*", "val/*", "time/*", "lr/*", "routing/*", "stage"):
            run.define_metric(prefix, step_metric="epoch")
        run.summary.update({"n_params_gate": n_params["gate"], "n_params_experts": n_params["experts"]})
    if verbose:
        print(f"[task3] device={device} amp={use_amp} tau={config['tau']} gate={n_params['gate'] / 1e6:.2f}M "
              f"experts={n_params['experts'] / 1e6:.2f}M steps/epoch={steps_per_epoch} epochs {start_epoch}.."
              f"{total_epochs} (warm-up 1..{config['warmup_epochs']}) init={source}", flush=True)

    stop_reason = None
    for epoch in range(start_epoch, total_epochs + 1):
        stage = stage_of(epoch, config)
        if stage != current_stage:
            apply_stage(model, stage, config)
            optimizer, scheduler = build_optimizer(model, stage, config, steps_per_epoch)
            current_stage = stage
        if epoch == config["warmup_epochs"] + 1:
            stopper.bad_epochs = 0  # the joint stage gets its full patience
        seed_everything(config["seed"] + epoch)
        loader = make_train_loader(data["train_images"], config, device, epoch)

        t0 = time.perf_counter()
        try:
            train_stats = train_one_epoch(model, loader, optimizer, scheduler, scaler, config, device,
                                          use_amp, channels_last)
        except NonFiniteLossError as exc:
            if trial is not None:
                import optuna

                trial.set_user_attr("prune_reason", f"diverged at epoch {epoch}: {exc}")
                raise optuna.TrialPruned(f"diverged at epoch {epoch}") from exc
            raise
        train_time = time.perf_counter() - t0
        val_metrics, stats = validate(model, data["val"], device, config, use_amp, channels_last)
        collapsed, collapse_reason = detect_collapse(stats, config["collapse_min_usage"],
                                                     config["collapse_max_off_class"])
        is_best = stopper.step(val_metrics["score"])
        if is_best:
            best_epoch, best_metrics = epoch, dict(val_metrics)
        record = {
            "epoch": epoch, "stage": stage,
            "lr_gate": optimizer.param_groups[0]["lr"], "lr_experts": optimizer.param_groups[-1]["lr"] if stage == "joint" else 0.0,
            **{f"train_{k}": v for k, v in train_stats.items() if k != "images"},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "routing_matrix": stats["matrix"].tolist(), "collapsed": collapsed, "collapse_reason": collapse_reason,
            "epoch_time_s": time.perf_counter() - t0, "train_images_per_s": train_stats["images"] / train_time,
        }
        history.append(record)
        finished = epoch == total_epochs or (stage == "joint" and stopper.should_stop)
        if finished:
            stop_reason = "max_epochs" if epoch == total_epochs else "early_stopping"

        if ckpt_dir is not None:
            common = {"model_config": model.config, "config": config, "init_source": source,
                      "wandb_run_id": wandb_id, "wandb_url": wandb_url}
            if is_best:  # best.pt before last.pt: a crash in between never leaves best.pt behind last.pt
                save_checkpoint(ckpt_dir / "best.pt", **common, model_state=model.state_dict(), epoch=epoch,
                                stage=stage, metrics=val_metrics, routing_matrix=stats["matrix"].tolist())
            save_checkpoint(
                ckpt_dir / "last.pt", **common, model_state=model.state_dict(), epoch=epoch, stage=stage,
                metrics=val_metrics, optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict(), early_stopping=stopper.state_dict(), history=history,
                best_epoch=best_epoch, best_metrics=best_metrics, finished=finished, stop_reason=stop_reason,
            )
        if run is not None:
            show = trial is None and (epoch % config["sample_every"] == 0 or finished or epoch == 1)
            grid = sample_grid(model, data["val"], sample_idx, device, use_amp, channels_last) if show else None
            _log_epoch(run, record, stats, grid)
        if verbose:
            usage = "/".join(f"{u:.2f}" for u in stats["usage"])
            print(f"epoch {epoch:3d}/{total_epochs} {stage:6s} | loss {train_stats['loss']:.4f} "
                  f"(ce {train_stats['ce']:.3f} bal {train_stats['balance']:.4f}) | val psnr {val_metrics['psnr']:.2f} "
                  f"ssim {val_metrics['ssim']:.4f} score {val_metrics['score']:.4f}{' *' if is_best else ''} | "
                  f"route acc {stats['accuracy']:.3f} usage {usage} | {record['epoch_time_s']:.1f}s", flush=True)
        if collapsed:
            print(f"[task3] WARNING: routing collapse at epoch {epoch}: {collapse_reason}", flush=True)
        if trial is not None:
            import optuna

            trial.report(val_metrics["score"], step=epoch)
            trial.set_user_attr("epochs_run", epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val", {k: round(v, 5) for k, v in best_metrics.items()})
            trial.set_user_attr("routing_usage", [round(float(u), 4) for u in stats["usage"]])
            trial.set_user_attr("routing_off_class", [round(float(u), 4) for u in stats["off_class"]])
            trial.set_user_attr("routing_accuracy", round(stats["accuracy"], 4))
            if collapsed:
                trial.set_user_attr("prune_reason", f"collapse at epoch {epoch}: {collapse_reason}")
                raise optuna.TrialPruned(f"routing collapse at epoch {epoch}: {collapse_reason}")
            if trial.should_prune():
                trial.set_user_attr("prune_reason", f"median pruner at epoch {epoch} "
                                                    f"(score {val_metrics['score']:.4f})")
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (score {val_metrics['score']:.4f})")
        if finished:
            break
    if stop_reason == "early_stopping" and verbose:
        print(f"[task3] early stopping: no improvement for {config['patience']} joint epochs", flush=True)

    result = {"best_score": stopper.best, "best_epoch": best_epoch, "best_metrics": best_metrics,
              "epochs_run": len(history), "stop_reason": stop_reason, "history": history, "n_params": n_params}
    if run is not None:
        run.summary.update({"best/score": stopper.best, "best/epoch": best_epoch,
                            **{f"best/{k}": v for k, v in (best_metrics or {}).items()}})
    if run is not None and ckpt_dir is not None and (ckpt_dir / "best.pt").exists():
        try:
            log_artifact(run, ckpt_dir / "best.pt", name="task3-moe", aliases=["best"],
                         metadata={"best_epoch": best_epoch, **(best_metrics or {})})
        except Exception as exc:  # the checkpoint is safe on disk; a failed upload must not lose the run
            print(f"[task3] WARNING: could not upload best.pt to W&B: {exc}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
HYPERPARAMETER_FLAGS = {
    "tau": (float, "routing temperature: w = softmax(G(x) / tau)"),
    "lambda_l1": (float, "weight of the L1 term"),
    "lambda_ssim": (float, "weight of the (1 - SSIM) term"),
    "lambda_ce": (float, "weight of the gate cross-entropy"),
    "lambda_balance": (float, "weight of the routing-balance term"),
    "ce_input": (str, "'logits' (CE on G(x)) or 'routing' (CE on G(x) / tau)"),
    "warmup_epochs": (int, "stage 1 epochs (experts frozen, gate only)"),
    "lr_warmup": (float, "gate learning rate in the warm-up"),
    "joint_epochs": (int, "stage 2 epochs (joint fine-tuning; length of the cosine schedule)"),
    "lr": (float, "gate learning rate in the joint stage (experts: lr * expert_lr_scale)"),
    "expert_lr_scale": (float, "expert learning rate as a fraction of the joint gate learning rate"),
    "min_lr_ratio": (float, "final joint learning rate as a fraction of the peak"),
    "weight_decay": (float, "AdamW decoupled weight decay"),
    "grad_clip": (float, "max gradient norm (0 disables clipping)"),
    "batch_size": (int, "training batch size (multiple of 4)"),
    "patience": (int, "early-stopping patience in joint epochs"),
    "seed": (int, "random seed"),
    "sample_every": (int, "log the W&B validation image grid every N epochs"),
    "num_workers": (int, "DataLoader workers on CUDA machines"),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with any of the hyperparameters below")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data, 1+1 epochs, W&B disabled")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    parser.add_argument("--run-name", default="moe-final", help="W&B run name")
    parser.add_argument("--wandb-group", default=FINAL_GROUP)
    hp = parser.add_argument_group("hyperparameters (override --config and the defaults)")
    for key, (typ, text) in HYPERPARAMETER_FLAGS.items():
        hp.add_argument("--" + key.replace("_", "-"), type=typ, default=argparse.SUPPRESS,
                        help=f"{text} (default {DEFAULTS[key]})")
    hp.add_argument("--expert-bn", choices=("frozen", "train"), default=argparse.SUPPRESS,
                    help="joint stage: keep the experts' BatchNorm statistics (default) or re-estimate them")
    hp.add_argument("--no-amp", dest="amp", action="store_false", default=argparse.SUPPRESS,
                    help="disable fp16 mixed precision on CUDA")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        os.environ["WANDB_MODE"] = "disabled"
    overrides = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}
    if hasattr(args, "expert_bn"):
        overrides["freeze_expert_bn"] = args.expert_bn == "frozen"
    config = resolve_config(args.config, overrides, smoke=args.smoke)
    ckpt_dir = checkpoint_dir(args.smoke)
    last = ckpt_dir / "last.pt"

    state = None
    if args.resume and last.exists():
        state = load_checkpoint(last)
        changed = {k for k in config if k not in RUN_CONTROL_KEYS and state["config"].get(k) != config[k]}
        if changed:
            print(f"[task3] --resume: using the checkpoint's settings; ignoring changed {sorted(changed)}")
        config = {**state["config"], **{k: config[k] for k in RUN_CONTROL_KEYS}}
        if state.get("finished"):
            print(f"[task3] {last} is already finished ({state.get('stop_reason')}, best epoch "
                  f"{state['best_epoch']}, val score {state['early_stopping']['best']:.4f}); nothing to do.")
            return {"best_score": state["early_stopping"]["best"], "best_epoch": state["best_epoch"],
                    "epochs_run": state["epoch"], "already_finished": True}
        print(f"[task3] resuming from epoch {state['epoch'] + 1} ({stage_of(state['epoch'] + 1, config)} stage)")
    elif last.exists():
        print(f"[task3] WARNING: {last} exists and --resume was not given: starting a new run that overwrites it")

    device = get_device()
    data = load_data(args.smoke)
    init = None if state else load_init(args.smoke)
    run_config = {**config, "device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "init_source": state.get("init_source") if state else init[2]}
    tags = ["task3", "moe"] + (["smoke"] if args.smoke else [])
    # Continue the interrupted run's W&B history: the id goes to wandb.init, never through
    # WANDB_RUN_ID (wandb reads that variable once per process).
    resume = {"id": state["wandb_run_id"], "resume": "allow"} if state and state.get("wandb_run_id") else {}
    run = init_run(args.run_name, args.wandb_group, config=run_config, job_type="train", tags=tags, **resume)
    try:
        result = train_moe(config, data=data, init=init, run=run, ckpt_dir=ckpt_dir, resume_state=state,
                           device=device, smoke=args.smoke)
    finally:
        run.finish()
    print(f"[task3] done: best val score {result['best_score']:.4f} at epoch {result['best_epoch']} "
          f"-> {ckpt_dir / 'best.pt'}")
    return result


if __name__ == "__main__":
    main()
