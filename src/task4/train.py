"""Task 4: train the style-conditioned face-to-sketch conditional GAN (pix2pix recipe).

The discriminator sees (photo, real sketch, style) as real and (photo, G(photo, style), style)
as fake; the generator minimises its adversarial loss plus `lambda_l1` times the L1 distance
to the paired sketch. The four loss components (D real, D fake, G adversarial, G L1) are
logged separately per step and per epoch, as the brief requires. `train_model` is shared by
this CLI and the Optuna search (src/task4/optuna_search.py).

    python -m src.task4.train --config $OUTPUT_DIR/task4/best_config.yaml --resume
    python -m src.task4.train --smoke

Checkpoints: CKPT_DIR/task4/cgan/last.pt (every epoch, full G+D state for --resume) and
best.pt (best validation sketch_score; only the generator is needed afterwards).
See docs/fs2k_notes.md.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.tracking import image_grid, init_run, log_artifact
from src.common.utils import RunningMean, count_parameters, get_device, seed_everything
from src.data.fs2k import FS2KPairs, NUM_STYLES, STYLE_NAMES, load_fs2k
from src.task4.models import PatchDiscriminator, UNetGenerator

from .config import (
    DEFAULTS, RUN_CONTROL_KEYS, checkpoint_dir, discriminator_config, generator_config, resolve_config,
    resolve_num_workers,
)
from .metrics import sketch_metrics, sketch_score, to_unit

FINAL_GROUP = "task4-final"
SAMPLES_PER_STYLE = 2  # fixed validation pairs per style shown in the W&B grid
VAL_BATCH = 32
LOSS_KEYS = ("d_real", "d_fake", "g_adv", "g_l1")


class NonFiniteLossError(RuntimeError):
    """A training loss became NaN/inf (diverged run; fp16 overflow is handled by the GradScaler)."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def tensorize(data):
    """Materialise a split as tensors once (validation is small and never augmented)."""
    dataset = FS2KPairs(data)
    items = [dataset[i] for i in range(len(dataset))]
    return {
        "photo": torch.stack([it["photo"] for it in items]),
        "sketch": torch.stack([it["sketch"] for it in items]),
        "style": torch.tensor([it["style"] for it in items], dtype=torch.long),
        "id": [it["id"] for it in items],
    }


def load_data(smoke=False):
    """Training arrays (augmented fresh every epoch) and the validation split as tensors."""
    data = load_fs2k(smoke=smoke)
    return {"train": data["train"], "val": tensorize(data["val"])}


def fixed_sample_indices(val, per_style=SAMPLES_PER_STYLE):
    """The same validation pairs in every run and every epoch, so development is comparable.

    Per style the first entries of distinct photo sources are preferred, so the grid also
    shows the photo2/photo3 domains and not only the dominant photo1 (see docs/fs2k_notes.md).
    """
    styles = val["style"].tolist()
    chosen = []
    for s in range(NUM_STYLES):
        members = [i for i, st in enumerate(styles) if st == s]
        seen, picks = set(), []
        for i in members:  # one per source first, then fill up in order
            source = val["id"][i].split("/")[0]
            if source not in seen:
                seen.add(source)
                picks.append(i)
        picks += [i for i in members if i not in picks]
        chosen += picks[:per_style]
    return chosen


def make_train_loader(data, config, device, seed):
    workers = resolve_num_workers(config["num_workers"], device)
    dataset = FS2KPairs(data, augment=True, hflip=config["hflip"], max_zoom=config["max_zoom"])
    if config["batch_size"] > len(dataset):
        raise ValueError(f"batch_size {config['batch_size']} exceeds the {len(dataset)} training pairs")
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def linear_decay(epochs, decay_fraction):
    """pix2pix LR schedule: constant, then linear decay towards 0 over the last fraction."""
    keep = max(1, int(round(epochs * (1 - decay_fraction))))
    decay = max(1, epochs - keep)

    def factor(epoch):  # epoch is 0-based, as LambdaLR counts
        return 1.0 if epoch < keep else max(0.0, (epochs - epoch) / decay)

    return factor


def build_optimizers(generator, discriminator, config):
    """Adam with beta1 = 0.5 (DCGAN: beta1 = 0.9 oscillates; Radford et al., 2016)."""
    betas = (config["beta1"], 0.999)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=betas)
    factor = linear_decay(config["epochs"], config["decay_fraction"])
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, factor)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, factor)
    return opt_g, opt_d, sch_g, sch_d


def _autocast(device, enabled):
    # fp16 on CUDA only. BCE-with-logits is on PyTorch's fp32 autocast list, and the L1 term
    # is cast explicitly, so both generator losses are accumulated in fp32.
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def gan_step(batch, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, device, use_amp, lambda_l1):
    """One discriminator update and one generator update; returns the four loss components."""
    photo = batch["photo"].to(device, non_blocking=True)
    sketch = batch["sketch"].to(device, non_blocking=True)
    style = batch["style"].to(device, non_blocking=True)

    with _autocast(device, use_amp):
        fake = generator(photo, style)

    # Discriminator: real pairs -> 1, generated pairs -> 0 (fake detached: no G gradients)
    with _autocast(device, use_amp):
        pred_real = discriminator(photo, sketch, style)
        pred_fake = discriminator(photo, fake.detach(), style)
        d_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        d_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        d_loss = 0.5 * (d_real + d_fake)  # pix2pix halves D's objective to slow it down
    opt_d.zero_grad(set_to_none=True)
    scaler_d.scale(d_loss).backward()
    scaler_d.step(opt_d)
    scaler_d.update()

    # Generator: fool D on the same pairs, and stay close to the paired sketch
    with _autocast(device, use_amp):
        pred = discriminator(photo, fake, style)
        g_adv = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
    g_l1 = F.l1_loss(fake.float(), sketch.float())
    g_loss = g_adv + lambda_l1 * g_l1
    opt_g.zero_grad(set_to_none=True)
    scaler_g.scale(g_loss).backward()
    scaler_g.step(opt_g)
    scaler_g.update()

    return {"d_real": d_real.detach(), "d_fake": d_fake.detach(), "g_adv": g_adv.detach(), "g_l1": g_l1.detach()}


def train_one_epoch(loader, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, device, config,
                    use_amp, run=None, global_step=0):
    """One pass over the training set; logs the four losses every `log_every` steps."""
    generator.train()
    discriminator.train()
    means, n_images = RunningMean(), 0
    for batch in loader:
        losses = gan_step(batch, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, device,
                          use_amp, config["lambda_l1"])
        n = batch["photo"].shape[0]
        values = {k: float(v) for k, v in losses.items()}
        means.update(values, n)
        n_images += n
        global_step += 1
        if run is not None and global_step % config["log_every"] == 0:
            run.log({"global_step": global_step, **{f"step/{k}": v for k, v in values.items()},
                     "step/g_total": values["g_adv"] + config["lambda_l1"] * values["g_l1"]})
    epoch_means = means.means()
    if not all(math.isfinite(v) for v in epoch_means.values()):
        raise NonFiniteLossError(f"non-finite training losses: {epoch_means}")
    return {**epoch_means, "images": n_images}, global_step


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(generator, photos, styles, device, batch_size=VAL_BATCH):
    """Generated sketches (float32, on the device) for CPU tensors, in eval mode.

    Validation and all reported metrics run in float32, never under autocast: the pass is
    cheap (159 pairs) and fp16 rounding would add noise to the numbers Optuna compares.
    """
    generator.eval()
    outputs = []
    for start in range(0, len(photos), batch_size):
        x = photos[start:start + batch_size].to(device, non_blocking=True)
        s = styles[start:start + batch_size].to(device, non_blocking=True)
        outputs.append(generator(x, s).float())
    return torch.cat(outputs)


@torch.no_grad()
def validate(generator, val, device):
    """Validation metrics: overall and per style, plus the style-sensitivity gap.

    style_gap = mean L1 when the generator is given a wrong style minus the L1 with the
    true style (averaged over both wrong styles). A model that ignores the style scores 0.
    """
    styles = val["style"]
    output = generate(generator, val["photo"], styles, device)
    target = val["sketch"].to(output.device)
    per_image = sketch_metrics(output, target)

    metrics = {k: float(v.mean()) for k, v in per_image.items()}
    metrics["score"] = sketch_score(metrics["ssim"], metrics["l1"])
    for s in range(NUM_STYLES):
        mask = (styles == s).numpy()
        if mask.any():
            for k, v in per_image.items():
                metrics[f"style{s}/{k}"] = float(v.numpy()[mask].mean())

    shifted = []
    for shift in range(1, NUM_STYLES):
        wrong = generate(generator, val["photo"], (styles + shift) % NUM_STYLES, device)
        shifted.append(float((to_unit(wrong) - to_unit(target)).abs().flatten(1).mean(1).mean()))
    metrics["l1_wrong_style"] = float(np.mean(shifted))
    metrics["style_gap"] = metrics["l1_wrong_style"] - metrics["l1"]
    return metrics


def sample_grid(generator, val, indices, device):
    """Rows of photo | ground truth | sketch generated for each of the three styles.

    The same validation pairs every time, so the grids of different epochs (and of the
    Optuna trials) can be compared directly; the three style columns also show whether the
    style condition is doing anything.
    """
    photos = val["photo"][indices]
    rows = []
    per_style = [generate(generator, photos, torch.full((len(indices),), s, dtype=torch.long), device).cpu()
                 for s in range(NUM_STYLES)]
    for k, i in enumerate(indices):
        row = [to_unit(val["photo"][i]).permute(1, 2, 0).numpy(), to_unit(val["sketch"][i])[0].numpy()]
        row += [to_unit(per_style[s][k])[0].numpy() for s in range(NUM_STYLES)]
        rows.append(row)
    return image_grid(rows)


def sample_caption(val, indices):
    pairs = ", ".join(f"{val['id'][i]} ({STYLE_NAMES[int(val['style'][i])]})" for i in indices)
    return f"photo | ground truth | {' | '.join(STYLE_NAMES)}; rows: {pairs}"


# --------------------------------------------------------------------------- #
# Training loop (shared with the Optuna search)
# --------------------------------------------------------------------------- #
def _log_epoch(run, record, grid, caption):
    import wandb

    payload = {"epoch": record["epoch"], "lr_g": record["lr_g"], "lr_d": record["lr_d"]}
    payload.update({f"train/{k}": record[f"train_{k}"] for k in LOSS_KEYS})
    payload.update({f"val/{k[4:]}": v for k, v in record.items() if k.startswith("val_")})
    payload["time/epoch_s"] = record["epoch_time_s"]
    payload["time/train_images_per_s"] = record["train_images_per_s"]
    if grid is not None:
        payload["val/samples"] = wandb.Image(grid, caption=caption)
    run.log(payload)


def train_model(config, data, run=None, trial=None, ckpt_dir=None, resume_state=None, device=None, verbose=True):
    """Train one cGAN configuration; returns a summary dict.

    run: W&B run or None. trial: Optuna trial or None; when given, the validation score is
    reported every epoch and optuna.TrialPruned is raised if the pruner says so.
    ckpt_dir: where last.pt / best.pt go (None: no checkpoints, as in Optuna trials).
    resume_state: a loaded last.pt to continue from.
    """
    device = device or get_device()
    use_amp = bool(config["amp"]) and device.type == "cuda"  # the config keeps what was requested
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128x128 inputs
    start_epoch = resume_state["epoch"] + 1 if resume_state else 1
    seed_everything(config["seed"] + start_epoch - 1)  # a resumed run does not replay epoch 1

    gcfg, dcfg = generator_config(config), discriminator_config(config)
    generator = UNetGenerator(**gcfg).to(device)
    discriminator = PatchDiscriminator(**dcfg).to(device)
    opt_g, opt_d, sch_g, sch_d = build_optimizers(generator, discriminator, config)
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = make_train_loader(data["train"], config, device, seed=config["seed"] + start_epoch - 1)
    history, best_epoch, best_metrics, best_score, global_step = [], None, None, -float("inf"), 0

    if resume_state:
        generator.load_state_dict(resume_state["model_state"])
        discriminator.load_state_dict(resume_state["d_state"])
        opt_g.load_state_dict(resume_state["opt_g_state"])
        opt_d.load_state_dict(resume_state["opt_d_state"])
        sch_g.load_state_dict(resume_state["sch_g_state"])
        sch_d.load_state_dict(resume_state["sch_d_state"])
        if use_amp:
            scaler_g.load_state_dict(resume_state["scaler_g_state"])
            scaler_d.load_state_dict(resume_state["scaler_d_state"])
        history = list(resume_state["history"])
        best_epoch, best_metrics = resume_state["best_epoch"], resume_state["best_metrics"]
        best_score, global_step = resume_state["best_score"], resume_state["global_step"]

    sample_idx = fixed_sample_indices(data["val"])
    caption = sample_caption(data["val"], sample_idx)
    n_params = {"generator": count_parameters(generator), "discriminator": count_parameters(discriminator)}
    wandb_id = getattr(run, "id", None) if run is not None else None
    wandb_url = getattr(run, "url", None) if run is not None else None
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("global_step")
        for prefix in ("train/*", "val/*", "time/*", "lr_g", "lr_d"):
            run.define_metric(prefix, step_metric="epoch")
        run.define_metric("step/*", step_metric="global_step")
        run.summary.update({f"n_params_{k}": v for k, v in n_params.items()})
    if verbose:
        print(f"[task4] device={device} amp={use_amp} G={n_params['generator'] / 1e6:.2f}M "
              f"D={n_params['discriminator'] / 1e6:.2f}M base={config['base_channels']} "
              f"style_dim={config['style_dim']} lambda_l1={config['lambda_l1']} "
              f"steps/epoch={len(loader)} epochs {start_epoch}..{config['epochs']}", flush=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.perf_counter()
        train_stats, global_step = train_one_epoch(loader, generator, discriminator, opt_g, opt_d, scaler_g,
                                                   scaler_d, device, config, use_amp, run, global_step)
        train_time = time.perf_counter() - t0
        val_metrics = validate(generator, data["val"], device)
        if not math.isfinite(val_metrics["score"]):
            raise NonFiniteLossError(f"validation score is {val_metrics['score']}")
        record = {
            "epoch": epoch, "lr_g": opt_g.param_groups[0]["lr"], "lr_d": opt_d.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_stats.items() if k != "images"},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_time_s": time.perf_counter() - t0,
            "train_images_per_s": train_stats["images"] / train_time,
        }
        history.append(record)
        sch_g.step()
        sch_d.step()

        is_best = val_metrics["score"] > best_score
        if is_best:
            best_score, best_epoch, best_metrics = val_metrics["score"], epoch, dict(val_metrics)
        finished = epoch == config["epochs"]

        if ckpt_dir is not None:
            common = {"model_config": gcfg, "d_config": dcfg, "config": config,
                      "wandb_run_id": wandb_id, "wandb_url": wandb_url}
            if is_best:  # best.pt before last.pt: a crash in between never leaves best.pt behind
                save_checkpoint(ckpt_dir / "best.pt", **common, model_state=generator.state_dict(),
                                epoch=epoch, metrics=val_metrics)
            save_checkpoint(
                ckpt_dir / "last.pt", **common, model_state=generator.state_dict(),
                d_state=discriminator.state_dict(), epoch=epoch, metrics=val_metrics,
                opt_g_state=opt_g.state_dict(), opt_d_state=opt_d.state_dict(),
                sch_g_state=sch_g.state_dict(), sch_d_state=sch_d.state_dict(),
                scaler_g_state=scaler_g.state_dict(), scaler_d_state=scaler_d.state_dict(),
                history=history, best_epoch=best_epoch, best_metrics=best_metrics, best_score=best_score,
                global_step=global_step, finished=finished,
            )
        if run is not None:
            show = epoch % config["sample_every"] == 0 or finished or epoch == 1
            _log_epoch(run, record, sample_grid(generator, data["val"], sample_idx, device) if show else None,
                       caption)
        if verbose:
            print(f"epoch {epoch:3d}/{config['epochs']} | D real {train_stats['d_real']:.3f} fake "
                  f"{train_stats['d_fake']:.3f} | G adv {train_stats['g_adv']:.3f} L1 {train_stats['g_l1']:.4f}"
                  f" | val L1 {val_metrics['l1']:.4f} ssim {val_metrics['ssim']:.4f} score "
                  f"{val_metrics['score']:.4f}{' *' if is_best else ''} | gap {val_metrics['style_gap']:+.4f}"
                  f" edge {val_metrics['edge_ratio']:.2f} | {record['epoch_time_s']:.1f}s", flush=True)
        if trial is not None:
            import optuna

            trial.report(val_metrics["score"], step=epoch)
            trial.set_user_attr("epochs_run", epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val", {k: round(v, 5) for k, v in best_metrics.items()})
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (score {val_metrics['score']:.4f})")

    result = {"best_score": best_score, "best_epoch": best_epoch, "best_metrics": best_metrics,
              "epochs_run": len(history), "history": history, "n_params": n_params,
              "global_step": global_step}
    if run is not None:
        run.summary.update({"best/score": best_score, "best/epoch": best_epoch,
                            **{f"best/{k}": v for k, v in (best_metrics or {}).items()}})
        if ckpt_dir is not None and (ckpt_dir / "best.pt").exists():
            try:
                log_artifact(run, ckpt_dir / "best.pt", name="task4-cgan", aliases=["best"],
                             metadata={"best_epoch": best_epoch, **(best_metrics or {})})
            except Exception as exc:  # the checkpoint is safe on disk; a failed upload must not lose the run
                print(f"[task4] WARNING: could not upload best.pt to W&B: {exc}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
HYPERPARAMETER_FLAGS = {
    "base_channels": (int, "width of G and D (G: c, 2c, ... 8c; D: the same base)"),
    "dropout": (float, "dropout rate in the three innermost decoder blocks of G"),
    "style_dim": (int, "dimension of the learned style embedding (used by G and D)"),
    "d_layers": (int, "stride-2 blocks in D: 3 gives a 70x70 patch, 2 gives 34x34"),
    "lambda_l1": (float, "weight of the L1 reconstruction term in the generator loss"),
    "lr_g": (float, "generator learning rate (Adam)"),
    "lr_d": (float, "discriminator learning rate (Adam)"),
    "beta1": (float, "Adam beta1 for G and D (0.5 as in DCGAN/pix2pix)"),
    "batch_size": (int, "training batch size"),
    "epochs": (int, "number of epochs (length of the LR schedule)"),
    "decay_fraction": (float, "fraction of the epochs over which the LR decays linearly to 0"),
    "max_zoom": (float, "pix2pix jitter: maximum upscale factor before the random 128 crop"),
    "seed": (int, "random seed"),
    "log_every": (int, "log the four losses every N optimiser steps"),
    "sample_every": (int, "log the validation image grid every N epochs"),
    "num_workers": (int, "DataLoader workers on CUDA machines (always 0 on Windows/CPU)"),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with any of the hyperparameters below")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic data, 2 epochs, W&B disabled")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    parser.add_argument("--run-name", default="cgan", help="W&B run name")
    parser.add_argument("--wandb-group", default=FINAL_GROUP)
    hp = parser.add_argument_group("hyperparameters (override --config and the defaults)")
    for key, (typ, text) in HYPERPARAMETER_FLAGS.items():
        hp.add_argument("--" + key.replace("_", "-"), type=typ, default=argparse.SUPPRESS,
                        help=f"{text} (default {DEFAULTS[key]})")
    hp.add_argument("--no-hflip", dest="hflip", action="store_false", default=argparse.SUPPRESS,
                    help="disable the paired horizontal flip (ablation)")
    hp.add_argument("--no-amp", dest="amp", action="store_false", default=argparse.SUPPRESS,
                    help="disable fp16 mixed precision on CUDA")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        os.environ["WANDB_MODE"] = "disabled"
    overrides = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}
    config = resolve_config(args.config, overrides, smoke=args.smoke)
    ckpt_dir = checkpoint_dir(smoke=args.smoke)
    last = ckpt_dir / "last.pt"

    state = None
    if args.resume and last.exists():
        state = load_checkpoint(last)
        changed = {k for k in config if k not in RUN_CONTROL_KEYS and state["config"].get(k) != config[k]}
        if changed:
            print(f"[task4] --resume: using the checkpoint's settings; ignoring changed {sorted(changed)}")
        config = {**state["config"], **{k: config[k] for k in RUN_CONTROL_KEYS}}
        if state.get("finished"):
            print(f"[task4] {last} is already finished (best epoch {state['best_epoch']}, val score "
                  f"{state['best_score']:.4f}); nothing to do.")
            return {"best_score": state["best_score"], "best_epoch": state["best_epoch"],
                    "epochs_run": state["epoch"], "already_finished": True}
        print(f"[task4] resuming from epoch {state['epoch'] + 1}")
    elif last.exists():
        print(f"[task4] WARNING: {last} exists and --resume was not given: starting a new run that overwrites it")

    device = get_device()
    data = load_data(smoke=args.smoke)
    run_config = {**config, "generator_config": generator_config(config),
                  "discriminator_config": discriminator_config(config), "device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "n_train_pairs": len(data["train"]["ids"]), "n_val_pairs": len(data["val"]["id"])}
    tags = ["task4"] + (["smoke"] if args.smoke else [])
    resume_id = state.get("wandb_run_id") if state else None
    run = init_run(args.run_name, args.wandb_group, config=run_config, job_type="train", tags=tags,
                   **({"id": resume_id, "resume": "allow"} if resume_id else {}))
    try:
        result = train_model(config, data, run=run, ckpt_dir=ckpt_dir, resume_state=state, device=device)
    finally:
        run.finish()
    print(f"[task4] done: best val score {result['best_score']:.4f} at epoch {result['best_epoch']} "
          f"-> {ckpt_dir / 'best.pt'}")
    return result


if __name__ == "__main__":
    main()
