"""Task 4: Optuna search over the cGAN hyperparameters (study "task4_cgan").

Each trial trains a configuration for a short budget (default 10 epochs instead of the
final 100) with the same code as the final run (train.train_model), reports the validation
sketch_score every epoch and can be stopped early by a MedianPruner. GAN training is
expensive, which is why the brief allows shorter trials; the selected configuration is
afterwards retrained for the complete schedule with src.task4.train.

Objective: the best validation sketch_score of the trial (0.5 * SSIM + 0.5 * (1 - L1) on
the validation split only, never the official test set). It involves neither lambda_l1 nor
the discriminator, so trials with different loss weights are comparable; `edge_ratio` is
recorded as a user attribute so the blur/sharpness trade-off of large lambda_l1 stays visible.

The study is stored with src.common.optuna_utils (local SQLite, copied to OPTUNA_DIR after
every trial), so re-running this command after a Colab disconnect continues where it
stopped. The report goes to OUTPUT_DIR/task4/optuna/task4_cgan/ and the selected
configuration to OUTPUT_DIR/task4/best_config.yaml (input of the final training run).

    python -m src.task4.optuna_search --n-trials 20 --epochs 10 --timeout-min 80
    python -m src.task4.optuna_search --smoke
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
    DEFAULTS, RUN_CONTROL_KEYS, SMOKE_OVERRIDES, best_config_path, discriminator_config, generator_config,
    resolve_config, save_yaml_config, study_name, task_dir,
)
from .train import NonFiniteLossError, load_data, train_model

GROUP = "task4-optuna"

# Search space (docs/fs2k_notes.md explains every range)
LR_G_RANGE = (5e-5, 5e-4)       # log-uniform around pix2pix's 2e-4
LR_D_RANGE = (5e-5, 5e-4)       # searched separately: the G/D learning-rate ratio sets the balance
BATCH_SIZES = (4, 8, 16)        # 224 / 112 / 56 updates per epoch on 899 training pairs
BASE_CHANNELS = (32, 48, 64)    # 10.5M / 23.6M / 41.9M generator parameters (64 -> a 168 MB ONNX file)
DROPOUT_RANGE = (0.0, 0.5)      # pix2pix uses 0.5 as its only noise source
STYLE_DIMS = (8, 16, 32)        # dimension of the style embedding shared by G and D
LAMBDA_L1_RANGE = (25.0, 200.0)  # log-uniform around the brief's starting value of 100

# Trial 0: the brief's / pix2pix's starting point, as a reference every other trial is compared to
REFERENCE_PARAMS = {"lr_g": 2e-4, "lr_d": 2e-4, "batch_size": 8, "base_channels": 64,
                    "dropout": 0.5, "style_dim": 16, "lambda_l1": 100.0}

SEARCHED = tuple(REFERENCE_PARAMS)


def suggest(trial):
    return {
        "lr_g": trial.suggest_float("lr_g", *LR_G_RANGE, log=True),
        "lr_d": trial.suggest_float("lr_d", *LR_D_RANGE, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "base_channels": trial.suggest_categorical("base_channels", BASE_CHANNELS),
        "dropout": trial.suggest_float("dropout", *DROPOUT_RANGE),
        "style_dim": trial.suggest_categorical("style_dim", STYLE_DIMS),
        "lambda_l1": trial.suggest_float("lambda_l1", *LAMBDA_L1_RANGE, log=True),
    }


def search_space_description(epochs, n_trials, pruner):
    return {
        "objective": "max over epochs of validation sketch_score = 0.5*SSIM + 0.5*(1 - L1) (maximize)",
        "validation_split": "FS2K official training portion, 15% held out, stratified by style "
                            "(the official test set is never used here)",
        "params": {
            "lr_g": {"type": "float", "range": list(LR_G_RANGE), "log": True},
            "lr_d": {"type": "float", "range": list(LR_D_RANGE), "log": True},
            "batch_size": {"type": "categorical", "choices": list(BATCH_SIZES)},
            "base_channels": {"type": "categorical", "choices": list(BASE_CHANNELS)},
            "dropout": {"type": "float", "range": list(DROPOUT_RANGE)},
            "style_dim": {"type": "categorical", "choices": list(STYLE_DIMS)},
            "lambda_l1": {"type": "float", "range": list(LAMBDA_L1_RANGE), "log": True},
        },
        "fixed": {k: v for k, v in DEFAULTS.items()
                  if k not in SEARCHED and k not in ("epochs", *RUN_CONTROL_KEYS)},
        "epochs_per_trial": epochs,
        "n_trials": n_trials,
        "reference_trial": REFERENCE_PARAMS,
        "sampler": "TPESampler(seed=42, n_startup_trials=10)",
        "pruner": repr(pruner),
    }


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def make_objective(base_config, data, epochs, device, forced=None):
    """forced: values that override the suggestions (smoke runs keep the models tiny)."""

    def objective(trial):
        params = suggest(trial)
        config = {**base_config, **params, **(forced or {}), "epochs": epochs}
        run_config = {**config, "generator_config": generator_config(config),
                      "discriminator_config": discriminator_config(config),
                      "trial": trial.number, "study": trial.study.study_name}
        failure = None
        with wandb_run(f"task4-trial-{trial.number:03d}", GROUP, config=run_config, job_type="optuna",
                       tags=["task4", "optuna"]) as run:
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
            best = result["best_metrics"]
            trial.set_user_attr("n_params_generator", result["n_params"]["generator"])
            # Recorded, not optimised: a large lambda_l1 can win on L1/SSIM with blurry strokes
            trial.set_user_attr("edge_ratio", round(best["edge_ratio"], 4))
            trial.set_user_attr("style_gap", round(best["style_gap"], 5))
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
    config = {k: v for k, v in DEFAULTS.items() if k not in ("epochs", *RUN_CONTROL_KEYS)}
    config.update(best.params)
    header = (
        f"Task 4 cGAN configuration selected by Optuna study {study.study_name}\n"
        f"best trial {best.number}: validation sketch_score {best.value:.4f} after a "
        f"{epochs_per_trial}-epoch trial (epochs here is left to train.py's default)\n"
        f"edge_ratio {best.user_attrs.get('edge_ratio')}, style_gap {best.user_attrs.get('style_gap')}\n"
        "Use: python -m src.task4.train --config <this file> --resume"
    )
    return save_yaml_config(config, best_config_path(smoke), header)


def run_search(n_trials, epochs, smoke=False, timeout_min=None, pruner=None, fresh=False):
    if smoke:
        os.environ["WANDB_MODE"] = "disabled"
    device = get_device()
    base_config = resolve_config(smoke=smoke)
    pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3)
    study, sync = create_study(study_name(smoke), direction="maximize", pruner=pruner, fresh=fresh or smoke)
    _enqueue_reference(study)
    data = load_data(smoke=smoke)
    forced = {k: SMOKE_OVERRIDES[k] for k in ("base_channels", "style_dim", "batch_size")} if smoke else None
    objective = make_objective(base_config, data, epochs, device, forced)
    run_study(study, objective, n_trials, callbacks=[sync], timeout=timeout_min * 60 if timeout_min else None)

    report_dir = task_dir("OUTPUT_DIR", smoke, "optuna", study.study_name)
    save_study_report(study, report_dir)
    (report_dir / "search_space.json").write_text(
        json.dumps(search_space_description(epochs, n_trials, pruner), indent=2, default=str))
    if not study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        print("[optuna] no completed trials yet: best_config.yaml not written")
        return study, None
    path = write_best_config(study, epochs, smoke)
    best = study.best_trial
    print(f"[optuna] best trial {best.number}: score {best.value:.4f} params {best.params}")
    print(f"[optuna] report -> {report_dir}\n[optuna] best config -> {path}")
    return study, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=20,
                        help="total finished (complete + pruned) trials, counting earlier sessions")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs per trial")
    parser.add_argument("--timeout-min", type=float, default=80,
                        help="stop starting new trials after this many minutes in this session (0: no limit)")
    parser.add_argument("--smoke", action="store_true", help="2 trials x 1 epoch on synthetic data")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n_trials, args.epochs = min(args.n_trials, 2), min(args.epochs, 1)
    return run_search(args.n_trials, args.epochs, smoke=args.smoke, timeout_min=args.timeout_min or None)


if __name__ == "__main__":
    main()
