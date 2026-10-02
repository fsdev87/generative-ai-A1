"""Shared Optuna study "task2_specialists": one common architecture and training setup for all specialists.

The brief allows one shared search for the three specialists. Every trial trains the
salt, blur and occlusion specialists (separate models and optimisers, each only on its
own corruption) with the trial's hyperparameters, interleaved epoch by epoch: after
epoch e of all three, the mean of their three validation restoration_scores is
reported to Optuna, so MedianPruner judges the running mean over all corruption types
rather than one type at a time. The objective is the mean of the three best
validation scores; the per-type scores are stored as trial user attributes and logged
to W&B (group task2-specialists-optuna, one run per trial).

Afterwards each specialist is trained independently with the selected configuration
and the full schedule (src.task2.train_specialist --type ... --config <best config>).

Outputs: OUTPUT_DIR/task2/optuna/task2_specialists/ and
OUTPUT_DIR/task2/task2_specialists_best_config.yaml.

    python -m src.task2.optuna_specialists --smoke
    python -m src.task2.optuna_specialists --n-trials 20 --epochs 6
"""
import argparse
import json
import time

import numpy as np
import optuna

from src.common.optuna_utils import create_study, run_study, save_study_report
from src.common.paths import get_dir

from .common import (
    SPECIALIST_STUDY, SPECIALIST_TYPES, TASK, best_config_path, disable_wandb_for_smoke, save_config, setup_device,
    study_name, tracked_run,
)
from .train_specialist import DEFAULTS, SMOKE, SpecialistTrainer, load_data

# Bottleneck = latent tensor (resolution x resolution x channels); the input has 128*128*3 = 49,152 values.
LATENT_PRESETS = {
    "8x8x32": {"depth": 4, "latent_channels": 32},     # 2,048 values, compression 24x
    "8x8x64": {"depth": 4, "latent_channels": 64},     # 4,096 values, 12x
    "8x8x128": {"depth": 4, "latent_channels": 128},   # 8,192 values, 6x
    "16x16x16": {"depth": 3, "latent_channels": 16},   # 4,096 values, 12x
    "16x16x32": {"depth": 3, "latent_channels": 32},   # 8,192 values, 6x
    "16x16x64": {"depth": 3, "latent_channels": 64},   # 16,384 values, 3x
}

SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "base_channels": {"type": "categorical", "choices": [16, 32, 48]},
    "latent": {"type": "categorical", "choices": list(LATENT_PRESETS)},
    "alpha": {"type": "float", "low": 0.4, "high": 0.95},
}

SMOKE_FORCED = {k: SMOKE[k] for k in ("base_channels", "latent_channels", "batch_size")}


def suggest(trial):
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def apply_params(base, params):
    cfg = {**base, **{k: v for k, v in params.items() if k != "latent"}}
    cfg.update(LATENT_PRESETS[params["latent"]])
    return cfg


def run_trial(trial, cfg, data, device, smoke=False):
    """Train the three specialists interleaved; report the running mean score each epoch."""
    trainers = {t: SpecialistTrainer(t, cfg, data, device) for t in SPECIALIST_TYPES}
    best = {t: {"score": -np.inf} for t in SPECIALIST_TYPES}
    name = f"trial-{trial.number:03d}"
    with tracked_run(name, "task2-specialists-optuna", {**cfg, **trial.params}, tags=["smoke"] if smoke else None) as run:
        try:
            for epoch in range(cfg["epochs"]):
                t0, log, current = time.time(), {"epoch": epoch}, []
                for ctype, trainer in trainers.items():
                    train = trainer.train_epoch()
                    if not np.isfinite(train["loss"]):
                        raise optuna.TrialPruned(f"non-finite {ctype} training loss at epoch {epoch}")
                    val = trainer.validate()
                    if val["score"] > best[ctype]["score"]:
                        best[ctype] = val
                    current.append(val["score"])
                    log.update({f"{ctype}/train_loss": train["loss"], **{f"{ctype}/val_{k}": v for k, v in val.items()}})
                mean_score = float(np.mean(current))
                log.update({"val/mean_score": mean_score, "epoch_seconds": time.time() - t0})
                run.log(log)
                print(f"[{name}] epoch {epoch + 1}/{cfg['epochs']} | " + " | ".join(
                    f"{t} {s:.4f}" for t, s in zip(trainers, current)) + f" | mean {mean_score:.4f}", flush=True)
                trial.report(mean_score, epoch)
                if trial.should_prune():
                    run.summary["pruned_at_epoch"] = epoch
                    raise optuna.TrialPruned(f"pruned at epoch {epoch} (running mean {mean_score:.4f})")
            objective = float(np.mean([best[t]["score"] for t in SPECIALIST_TYPES]))
            run.summary["objective"] = objective
        finally:
            # per-type results are kept for pruned trials too
            for ctype, m in best.items():
                if np.isfinite(m["score"]):
                    for key in ("score", "psnr", "ssim"):
                        trial.set_user_attr(f"{key}_{ctype}", m[key])
                        run.summary[f"best/{key}_{ctype}"] = m[key]
            for trainer in trainers.values():
                del trainer.loader  # shut down persistent DataLoader workers
    return objective


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, help="finished trials wanted in total, all sessions (default 20; smoke 2)")
    parser.add_argument("--epochs", type=int, help="epochs per specialist per trial (default 6; smoke 2)")
    parser.add_argument("--timeout-min", type=float, help="stop starting new trials after this many minutes")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="2 trials x 2 epochs on tiny synthetic data")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    epochs = args.epochs or (2 if args.smoke else 6)
    n_trials = args.n_trials or (2 if args.smoke else 20)
    disable_wandb_for_smoke(args.smoke)
    name = study_name(SPECIALIST_STUDY, args.smoke)
    trial_base = {**DEFAULTS, "epochs": epochs, "num_workers": args.num_workers}
    final_base = dict(DEFAULTS)
    if args.smoke:
        trial_base.update({**SMOKE, "epochs": epochs, "patience": epochs})
        final_base.update(SMOKE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study, sync = create_study(name, direction="maximize", pruner=pruner, n_startup_trials=8, fresh=args.smoke)
    device = setup_device()
    data = load_data(args.smoke)
    for ctype, b in data["baseline"].items():
        print(f"[optuna] {ctype}: identity baseline PSNR {b['psnr']:.2f} dB, SSIM {b['ssim']:.4f}")

    def objective(trial):
        cfg = apply_params(trial_base, suggest(trial))
        if args.smoke:  # keep the smoke models and batches tiny whatever was sampled
            cfg.update(SMOKE_FORCED)
        return run_trial(trial, cfg, data, device, smoke=args.smoke)

    timeout = args.timeout_min * 60 if args.timeout_min else None
    run_study(study, objective, n_trials, callbacks=[sync], timeout=timeout)

    report_dir = get_dir("OUTPUT_DIR", TASK, "optuna", name)
    save_study_report(study, report_dir)
    (report_dir / "search_space.json").write_text(json.dumps(
        {"search_space": SEARCH_SPACE, "latent_presets": LATENT_PRESETS, "epochs_per_trial": trial_base["epochs"],
         "fixed": {k: trial_base[k] for k in ("weight_decay", "dropout", "skip_resolutions", "warmup_epochs",
                                              "min_lr_ratio")},
         "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=2) on the running mean score",
         "sampler": "TPE(seed=42, n_startup_trials=8)",
         "objective": "mean over salt/blur/occlusion of the best validation restoration_score"}, indent=2))
    best = study.best_trial  # raises if no trial completed
    cfg = apply_params(final_base, best.params)
    if args.smoke:
        cfg.update(SMOKE_FORCED)
    path = save_config(cfg, best_config_path(name))
    print(f"[optuna] best trial {best.number}: mean score {best.value:.4f} params {best.params}")
    print(f"[optuna] per-type scores {({t: round(best.user_attrs.get(f'score_{t}', float('nan')), 4) for t in SPECIALIST_TYPES})}")
    print(f"[optuna] final-run config (shared by the three specialists) written to {path}")
    return study


if __name__ == "__main__":
    main()
