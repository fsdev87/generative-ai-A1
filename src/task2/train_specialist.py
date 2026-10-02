"""Train one Task 2 specialist autoencoder (salt, blur or occlusion).

A specialist is a ConvAutoencoder trained ONLY on its own corruption:
RuntimeCorruptionDataset(train images, conditions=(type,)) draws a fresh severity on
every load, and the clean image is the target. Loss: RestorationLoss(alpha) =
alpha * L1 + (1 - alpha) * (1 - SSIM). Validation: the validation manifest filtered to
the same type; model selection and early stopping on restoration_score (0.5 SSIM +
0.5 PSNR / 40), which does not depend on alpha. AdamW, warm-up + cosine, AMP on CUDA.

Checkpoints: CKPT_DIR/task2/specialist_{type}/{last,best}.pt; W&B group task2-specialist-{type}.

    python -m src.task2.train_specialist --type salt --smoke
    python -m src.task2.train_specialist --type blur --config outputs/task2/task2_specialists_best_config.yaml --resume
"""
import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.losses import RestorationLoss
from src.common.metrics import image_metrics, restoration_score
from src.common.paths import get_dir
from src.common.tracking import log_artifact, log_image_grid
from src.common.utils import EarlyStopping, count_parameters, seed_everything
from src.data.corruptions import CLASS_TO_IDX, LEVELS
from src.data.pets import ManifestDataset, RuntimeCorruptionDataset, load_pets
from src.models.autoencoder import ConvAutoencoder

from .common import (
    SPECIALIST_TYPES, TASK, Amp, batches, build_config, config_diff, disable_wandb_for_smoke,
    guard_smoke_overwrite, make_loader, materialize, memory_format, model_state, new_run_id, param_groups,
    plain, resolve_workers, setup_device, tracked_run, warmup_cosine,
)

DEFAULTS = {
    "seed": 42,
    "epochs": 40,
    "warmup_epochs": 1,
    "min_lr_ratio": 0.01,
    "patience": 10,           # early stopping on validation restoration_score
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,     # light: runtime corruption already acts as strong augmentation
    "alpha": 0.8,             # L1 weight of RestorationLoss (brief's starting value)
    "base_channels": 32,      # channels per stage: base x (1, 2, 4, 8, 8)
    "depth": 4,               # 128 -> 8x8 latent
    "latent_channels": 64,
    "dropout": 0.0,
    "skip_resolutions": [],   # pure bottleneck: the latent is the only path from input to output
    "num_workers": None,
    "log_images_every": 5,
}
SMOKE = {"epochs": 2, "warmup_epochs": 1, "batch_size": 8, "base_channels": 4, "latent_channels": 4,
         "num_workers": 0, "log_images_every": 1, "patience": 5}


def specialist_dir(ctype):
    return get_dir("CKPT_DIR", TASK, f"specialist_{ctype}")


def load_data(smoke=False, types=SPECIALIST_TYPES):
    """Clean images, manifests, and per type the materialised validation subset and its identity baseline."""
    images, manifests = load_pets(smoke=smoke)
    val, baseline = {}, {}
    for ctype in types:
        val[ctype] = materialize(ManifestDataset(images["val"], manifests["val"], types=[ctype]))
        m = image_metrics(val[ctype]["input"], val[ctype]["target"])
        baseline[ctype] = {"psnr": float(m["psnr"].mean()), "ssim": float(m["ssim"].mean())}
    return {"images": images, "manifests": manifests, "val": val, "baseline": baseline}


def build_specialist(cfg):
    return ConvAutoencoder(base_channels=cfg["base_channels"], depth=cfg["depth"],
                           latent_channels=cfg["latent_channels"], dropout=cfg["dropout"],
                           skip_resolutions=tuple(cfg["skip_resolutions"]))


def fixed_sample_indices(levels, n=4):
    """Fixed validation entries for the W&B image panels: one per severity level, then the next ones."""
    chosen = [levels.index(lv) for lv in LEVELS if lv in levels]
    chosen += [i for i in range(len(levels)) if i not in chosen]
    return chosen[:n]


class SpecialistTrainer:
    """Model, optimiser, schedule and data of one specialist; one call per epoch.

    Used directly by the final run below and, three at a time, by the shared Optuna study.
    """

    def __init__(self, ctype, cfg, data, device):
        self.ctype, self.cfg, self.device = ctype, cfg, device
        seed_everything(cfg["seed"] + CLASS_TO_IDX[ctype])  # different initialisation per specialist
        self.num_workers = resolve_workers(cfg["num_workers"], device)
        dataset = RuntimeCorruptionDataset(data["images"]["train"], conditions=(ctype,))
        self.loader = make_loader(dataset, device, self.num_workers, batch_size=cfg["batch_size"],
                                  shuffle=True, drop_last=True)
        self.steps = len(self.loader)
        if self.steps == 0:
            raise ValueError(f"batch_size {cfg['batch_size']} larger than the training set")
        self.val = data["val"][ctype]
        self.baseline = data["baseline"][ctype]
        self.fmt = memory_format(device)
        self.model = build_specialist(cfg).to(device, memory_format=self.fmt)
        self.loss_fn = RestorationLoss(cfg["alpha"])
        self.optimizer = torch.optim.AdamW(param_groups(self.model, cfg["weight_decay"]), lr=cfg["lr"])
        self.scheduler = warmup_cosine(self.optimizer, cfg["epochs"] * self.steps, cfg["warmup_epochs"] * self.steps,
                                       cfg["min_lr_ratio"])
        self.amp = Amp(device)

    def train_epoch(self):
        self.model.train()
        sums = {k: torch.zeros((), device=self.device) for k in ("loss", "l1", "ssim")}
        seen = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")
            for batch in self.loader:
                x = batch["input"].to(self.device, non_blocking=True, memory_format=self.fmt)
                y = batch["target"].to(self.device, non_blocking=True)
                with self.amp.autocast():
                    out = self.model(x)
                loss, parts = self.loss_fn(out, y)  # float32 L1 and SSIM (autocast disabled inside ssim)
                self.amp.step(loss, self.optimizer)
                self.scheduler.step()
                n = len(y)
                sums["loss"] += loss.detach() * n
                sums["l1"] += parts["l1"] * n
                sums["ssim"] += parts["ssim"] * n
                seen += n
        out = {k: float(v) / seen for k, v in sums.items()}
        out["images"] = seen
        return out

    @torch.no_grad()
    def predict(self, x, batch_size=64):
        self.model.eval()
        return torch.cat([self.model(x[i:i + batch_size].to(self.device, memory_format=self.fmt)).float().cpu()
                          for i in range(0, len(x), batch_size)])

    @torch.no_grad()
    def validate(self, batch_size=64):
        """Float32 validation: loss, mean PSNR/SSIM, restoration_score, and per-level PSNR/SSIM."""
        self.model.eval()
        psnr, ssim, loss_sum = [], [], 0.0
        for b in batches(self.val, batch_size):
            out = self.model(b["input"].to(self.device, memory_format=self.fmt)).float()
            target = b["target"].to(self.device)
            loss, _ = self.loss_fn(out, target)
            loss_sum += float(loss) * len(target)
            m = image_metrics(out, target)
            psnr.append(m["psnr"])
            ssim.append(m["ssim"])
        psnr, ssim = torch.cat(psnr), torch.cat(ssim)
        metrics = {"loss": loss_sum / len(psnr), "psnr": float(psnr.mean()), "ssim": float(ssim.mean())}
        metrics["score"] = restoration_score(metrics["psnr"], metrics["ssim"])
        levels = np.array(self.val["level"])
        for lv in LEVELS:
            mask = torch.from_numpy(levels == lv)
            if mask.any():
                metrics[f"psnr_{lv}"] = float(psnr[mask].mean())
                metrics[f"ssim_{lv}"] = float(ssim[mask].mean())
        return metrics

    def sample_rows(self, indices):
        """Rows of Target | Input | Output | |Error| x4 for log_image_grid."""
        x, y = self.val["input"][indices], self.val["target"][indices]
        out = self.predict(x)
        hwc = lambda t: t.permute(1, 2, 0).numpy()  # noqa: E731
        return [[hwc(y[i]), hwc(x[i]), hwc(out[i]), np.clip(4 * (out[i] - y[i]).abs().mean(0).numpy(), 0, 1)]
                for i in range(len(indices))]

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.amp.scaler.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.amp.scaler.load_state_dict(state["scaler"])


def log_dict(prefix, metrics):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def train_specialist(config, ctype, smoke=False, resume=False, data=None):
    """Final training run of one specialist with checkpoints, resume and W&B.

    Returns {"best_score", "best_epoch", "best_metrics"}.
    """
    if ctype not in SPECIALIST_TYPES:
        raise ValueError(f"unknown specialist type {ctype!r}; expected one of {SPECIALIST_TYPES}")
    disable_wandb_for_smoke(smoke)
    cfg = dict(config)
    ckpt_dir = specialist_dir(ctype)
    state = None
    if resume and (ckpt_dir / "last.pt").exists():
        state = load_checkpoint(ckpt_dir / "last.pt")
        diff = config_diff(plain(state["train_config"]), plain(cfg))
        if diff:
            print(f"[{ctype}] resuming with the checkpoint's config; ignoring differences {diff}")
        cfg = state["train_config"]
        if state.get("finished"):
            print(f"[{ctype}] {ckpt_dir / 'last.pt'} is a finished run; nothing to do")
            return state["result"]
    guard_smoke_overwrite(ckpt_dir, smoke)

    device = setup_device()
    data = data or load_data(smoke, types=(ctype,))
    trainer = SpecialistTrainer(ctype, cfg, data, device)
    stopper = EarlyStopping(patience=cfg["patience"])
    start_epoch, best, run_id = 0, {"score": -math.inf, "epoch": -1, "metrics": None}, new_run_id()
    if state is not None:
        trainer.load_state_dict(state)
        stopper.load_state_dict(state["early_stopping"])
        start_epoch, best, run_id = state["epoch"] + 1, state["best"], state["wandb_run_id"]
        print(f"[{ctype}] resumed at epoch {start_epoch}")

    samples = fixed_sample_indices(trainer.val["level"])
    model = trainer.model
    print(f"[{ctype}] specialist: {count_parameters(model) / 1e6:.2f}M parameters, latent "
          f"{cfg['latent_channels']}x{model.resolutions[-1]}x{model.resolutions[-1]} ({model.latent_dim} values), "
          f"{trainer.steps} batches/epoch, device {device}, workers {trainer.num_workers}")
    print(f"[{ctype}] identity baseline on validation: PSNR {trainer.baseline['psnr']:.2f} dB, "
          f"SSIM {trainer.baseline['ssim']:.4f}")
    with tracked_run(f"specialist-{ctype}", f"task2-specialist-{ctype}", {**cfg, "type": ctype}, run_id=run_id,
                     tags=["smoke"] if smoke else None) as run:
        run.summary.update({"parameters": count_parameters(model), "latent_dim": model.latent_dim,
                            "baseline_psnr": trainer.baseline["psnr"], "baseline_ssim": trainer.baseline["ssim"]})
        for epoch in range(start_epoch, cfg["epochs"]):
            t0 = time.time()
            train = trainer.train_epoch()
            if not math.isfinite(train["loss"]):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            val = trainer.validate()
            is_best = stopper.step(val["score"])
            seconds = time.time() - t0
            if is_best:
                best = {"score": val["score"], "epoch": epoch, "metrics": val}
            run.log({"epoch": epoch, **log_dict("train", {k: v for k, v in train.items() if k != "images"}),
                     **log_dict("val", val), "lr": trainer.optimizer.param_groups[0]["lr"], "epoch_seconds": seconds,
                     "train_images_per_s": train["images"] / seconds, "best/score": best["score"]})
            if cfg["log_images_every"] and (epoch + 1) % cfg["log_images_every"] == 0:
                log_image_grid(run, "val/samples", trainer.sample_rows(samples),
                               caption=f"epoch {epoch + 1}: target | input | output | |error| x4")
            print(f"epoch {epoch + 1:3d}/{cfg['epochs']} | train loss {train['loss']:.4f} | val PSNR {val['psnr']:.2f}"
                  f" SSIM {val['ssim']:.4f} score {val['score']:.4f}{' *' if is_best else ''} | {seconds:.1f}s",
                  flush=True)
            common = dict(model_config=model.config, model_state=model_state(model), train_config=cfg,
                          task=f"task2-specialist-{ctype}", corruption=ctype, smoke=smoke)
            if is_best:
                save_checkpoint(ckpt_dir / "best.pt", **common, epoch=epoch, metrics=val, baseline=trainer.baseline,
                                wandb_run_url=run.url)
            save_checkpoint(ckpt_dir / "last.pt", **common, **trainer.state_dict(),
                            early_stopping=stopper.state_dict(), epoch=epoch, best=best, wandb_run_id=run_id,
                            finished=False)
            if stopper.should_stop:
                print(f"[{ctype}] early stopping: no improvement for {cfg['patience']} epochs")
                break

        result = {"best_score": best["score"], "best_epoch": best["epoch"], "best_metrics": best["metrics"]}
        run.summary.update({"best_score": best["score"], "best_epoch": best["epoch"],
                            "best_psnr": best["metrics"]["psnr"], "best_ssim": best["metrics"]["ssim"]})
        best_state = load_checkpoint(ckpt_dir / "best.pt")
        model.load_state_dict(best_state["model_state"])
        log_image_grid(run, "val/samples_best", trainer.sample_rows(samples),
                       caption=f"best epoch {best['epoch'] + 1}: target | input | output | |error| x4")
        log_artifact(run, ckpt_dir / "best.pt", f"task2-specialist-{ctype}", aliases=["best"],
                     metadata={"epoch": best["epoch"], **best["metrics"]})
        last = load_checkpoint(ckpt_dir / "last.pt")
        save_checkpoint(ckpt_dir / "last.pt", **{**last, "finished": True, "result": result})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=SPECIALIST_TYPES, help="corruption this specialist restores")
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
    result = train_specialist(cfg, args.type, smoke=args.smoke, resume=args.resume)
    print(f"[{args.type}] best validation score {result['best_score']:.4f} at epoch {result['best_epoch'] + 1}")
    return result


if __name__ == "__main__":
    main()
