"""Task 1: Optuna search over the UDAE hyperparameters (study "task1_udae").

Each trial trains a configuration for a short budget (default 15 epochs, a quarter of
the final run) with the same code as the final training (train.train_model), reports
the validation restoration_score every epoch and can be stopped early by a
MedianPruner. Objective: the best validation restoration_score of the trial
(0.5 * SSIM + 0.5 * PSNR / 40), which does not depend on alpha, so trials with
different loss weights are comparable.

The study is stored with src.common.optuna_utils (local SQLite, copied to OPTUNA_DIR
after every trial), so re-running this command after a Colab disconnect continues
where it stopped. Afterwards the report (plots, trials.csv, summary.json, search
space) goes to OUTPUT_DIR/task1/optuna/task1_udae/ and the selected configuration to
OUTPUT_DIR/task1/best_config.yaml (input of the final training run).

    python -m src.task1.optuna_search --n-trials 40 --epochs 15
    python -m src.task1.optuna_search --smoke
"""
import argparse
import gc
import json
import os

import optuna
import torch
from optuna.trial import TrialState

from src.common.optuna_utils import create_study, run_study, save_study_report
from src.common.tracking import wandb_run
from src.common.utils import get_device

from .config import (
    DEFAULTS, IMAGE_SIZE, SMOKE_OVERRIDES, best_config_path, bottleneck_info, model_config, resolve_config,
    save_yaml_config, study_name, task_dir,
)
from .train import NonFiniteLossError, load_data, train_model

GROUP = "task1-optuna"

# Search space (see docs/task1_notes.md for the reasoning behind every range)
LR_RANGE = (1e-4, 3e-3)            # log-uniform around Adam's usual 1e-3
BATCH_SIZES = (16, 32, 64)          # 184 / 92 / 46 updates per epoch on 2,944 images
LATENT_CHANNELS = (8, 16, 32, 64)   # latent 8x8xC = 512..4096 values = 96x..12x compression
BASE_CHANNELS = (16, 32, 48, 64)    # encoder width; compute grows ~quadratically
DROPOUT_RANGE = (0.0, 0.3)          # spatial dropout after every conv block
ALPHA_RANGE = (0.1, 0.95)           # weight of L1 vs (1 - SSIM)

# Trial 0: the brief's starting point (alpha = 0.8) with conventional defaults, as a reference
REFERENCE_PARAMS = {"lr": 1e-3, "batch_size": 32, "latent_channels": 32, "base_channels": 32,
                    "dropout": 0.0, "alpha": 0.8}

SEARCHED = tuple(REFERENCE_PARAMS)


def suggest(trial):
    return {
        "lr": trial.suggest_float("lr", *LR_RANGE, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "latent_channels": trial.suggest_categorical("latent_channels", LATENT_CHANNELS),
        "base_channels": trial.suggest_categorical("base_channels", BASE_CHANNELS),
        "dropout": trial.suggest_float("dropout", *DROPOUT_RANGE),
        "alpha": trial.suggest_float("alpha", *ALPHA_RANGE),
    }


def search_space_description(epochs, n_trials, pruner):
    latent = {c: bottleneck_info(model_config({**DEFAULTS, "latent_channels": c})) for c in LATENT_CHANNELS}
    return {
        "objective": "max over epochs of validation restoration_score = 0.5*SSIM + 0.5*PSNR/40 (maximize)",
        "params": {
            "lr": {"type": "float", "range": list(LR_RANGE), "log": True},
            "batch_size": {"type": "categorical", "choices": list(BATCH_SIZES)},
            "latent_channels": {"type": "categorical", "choices": list(LATENT_CHANNELS),
                                "latent_dim": {c: v["latent_dim"] for c, v in latent.items()},
                                "compression_ratio": {c: v["compression_ratio"] for c, v in latent.items()}},
            "base_channels": {"type": "categorical", "choices": list(BASE_CHANNELS)},
            "dropout": {"type": "float", "range": list(DROPOUT_RANGE)},
            "alpha": {"type": "float", "range": list(ALPHA_RANGE)},
        },
        "fixed": {k: v for k, v in DEFAULTS.items() if k not in SEARCHED and k not in ("epochs", "patience")},
        "epochs_per_trial": epochs,
        "n_trials": n_trials,
        "reference_trial": REFERENCE_PARAMS,
        "sampler": "TPESampler(seed=42, n_startup_trials=10)",
        "pruner": repr(pruner),
    }


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def make_objective(base_config, data, epochs, device, forced=None):
    """forced: values that override the suggestions (smoke runs keep the model tiny)."""

    def objective(trial):
        params = suggest(trial)
        config = {**base_config, **params, **(forced or {}), "epochs": epochs,
                  "patience": epochs}  # no early stopping inside a trial: the pruner decides
        info = bottleneck_info(model_config(config))
        trial.set_user_attr("latent_dim", info["latent_dim"])
        trial.set_user_attr("compression_ratio", info["compression_ratio"])
        run_config = {**config, **info, "trial": trial.number, "study": trial.study.study_name}
        failure = None
        with wandb_run(f"task1-trial-{trial.number:03d}", GROUP, config=run_config, job_type="optuna",
                       tags=["task1", "optuna"]) as run:
            try:
                result = train_model(config, data, run=run, trial=trial, device=device)
            except optuna.TrialPruned:
                trial.set_user_attr("pruned_reason", "pruner")
                run.summary["trial_state"] = "pruned (pruner)"
                raise
            except NonFiniteLossError as exc:
                failure = ("non_finite_loss", str(exc))
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                failure = ("cuda_oom", str(exc).splitlines()[0])
            if failure is not None:
                # Outside the except block the traceback (and the tensors its frames hold) is released
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr("pruned_reason", failure[0])
                run.summary["trial_state"] = f"pruned ({failure[0]})"
                print(f"[optuna] trial {trial.number} pruned: {failure[0]}: {failure[1]}", flush=True)
                raise optuna.TrialPruned(failure[0])
            trial.set_user_attr("n_params", result["n_params"])
            run.summary["trial_state"] = "complete"
        return result["best_score"]

    return objective


def _enqueue_reference(study):
    """Enqueue the reference trial unless it already ran (a FAILed attempt from a disconnect is retried)."""
    for t in study.get_trials(deepcopy=False):
        if t.system_attrs.get("fixed_params", t.params) == REFERENCE_PARAMS and t.state != TrialState.FAIL:
            return
    study.enqueue_trial(REFERENCE_PARAMS)


def write_best_config(study, epochs_per_trial, smoke=False):
    best = study.best_trial
    config = {k: v for k, v in DEFAULTS.items() if k not in ("epochs",)}
    config.update(best.params)
    config.pop("num_workers")
    config.pop("sample_every")
    info = bottleneck_info(model_config({**DEFAULTS, **config}))
    header = (
        f"Task 1 UDAE configuration selected by Optuna study {study.study_name}\n"
        f"best trial {best.number}: validation restoration_score {best.value:.4f} "
        f"after a {epochs_per_trial}-epoch trial (epochs here is left to train.py's default)\n"
        f"latent {info['latent_shape'][0]}x{info['latent_shape'][1]}x{info['latent_shape'][2]} = "
        f"{info['latent_dim']} values, compression {info['compression_ratio']:.1f}x of the "
        f"3x{IMAGE_SIZE}x{IMAGE_SIZE} input\n"
        "Use: python -m src.task1.train --config <this file> --resume"
    )
    return save_yaml_config(config, best_config_path(smoke), header)


def run_search(n_trials, epochs, smoke=False, timeout_min=None, pruner=None, fresh=False):
    if smoke:
        os.environ["WANDB_MODE"] = "disabled"
    device = get_device()
    base_config = resolve_config(smoke=smoke)
    pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study, sync = create_study(study_name(smoke), direction="maximize", pruner=pruner, fresh=fresh or smoke)
    _enqueue_reference(study)
    data = load_data(smoke=smoke, device=device)
    forced = {k: SMOKE_OVERRIDES[k] for k in ("base_channels", "latent_channels", "batch_size")} if smoke else None
    objective = make_objective(base_config, data, epochs, device, forced)
    run_study(study, objective, n_trials, callbacks=[sync], timeout=timeout_min * 60 if timeout_min else None)

    report_dir = task_dir("OUTPUT_DIR", smoke, "optuna", study.study_name)
    save_study_report(study, report_dir)
    (report_dir / "search_space.json").write_text(
        json.dumps(search_space_description(epochs, n_trials, pruner), indent=2, default=str))
    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not completed:
        print("[optuna] no completed trials yet: best_config.yaml not written")
        return study, None
    path = write_best_config(study, epochs, smoke)
    best = study.best_trial
    print(f"[optuna] best trial {best.number}: score {best.value:.4f} params {best.params}")
    print(f"[optuna] report -> {report_dir}\n[optuna] best config -> {path}")
    return study, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=40,
                        help="total finished (complete + pruned) trials, counting earlier sessions")
    parser.add_argument("--epochs", type=int, default=15, help="training epochs per trial")
    parser.add_argument("--timeout-min", type=float, default=90,
                        help="stop starting new trials after this many minutes in this session (0: no limit)")
    parser.add_argument("--smoke", action="store_true", help="2 trials x 2 epochs on synthetic data")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n_trials, args.epochs = min(args.n_trials, 2), min(args.epochs, 2)
    return run_search(args.n_trials, args.epochs, smoke=args.smoke, timeout_min=args.timeout_min or None)


if __name__ == "__main__":
    main()
