"""Optuna study "task2_classifier": hyperparameters of the corruption classifier.

Each trial trains a classifier for a short schedule (default 12 epochs, warm-up +
cosine over the trial's own budget) and returns its best validation macro-F1.
MedianPruner stops trials whose validation macro-F1 falls below the median of earlier
trials at the same epoch (never during the first 3 epochs, when the warm-up makes
the scores noisy). The study resumes after a disconnect (src.common.optuna_utils).

Outputs: OUTPUT_DIR/task2/optuna/task2_classifier/ (trials.csv, summary.json, plots,
search_space.json) and OUTPUT_DIR/task2/task2_classifier_best_config.yaml, the full
configuration for the final run (best hyperparameters + the final schedule).

    python -m src.task2.optuna_classifier --smoke
    python -m src.task2.optuna_classifier --n-trials 30 --epochs 12
"""
import argparse
import json

import optuna

from src.common.optuna_utils import create_study, run_study, save_study_report
from src.common.paths import get_dir

from .common import CLASSIFIER_STUDY, TASK, best_config_path, disable_wandb_for_smoke, save_config, study_name
from .train_classifier import DEFAULTS, SMOKE, load_data, train_classifier

# Per-stage widths of CorruptionClassifier; Optuna categoricals must be strings
CHANNEL_PRESETS = {
    "16-32-64-128": [16, 32, 64, 128],          # small: 0.21 GMAC per image with 2 convs/stage
    "32-64-128-256": [32, 64, 128, 256],        # default: 0.84 GMAC
    "48-96-192-384": [48, 96, 192, 384],        # wide: 1.89 GMAC
    "32-64-128-256-256": [32, 64, 128, 256, 256],  # one more stage: 4x4 final map, larger receptive field
}

SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    "channels": {"type": "categorical", "choices": list(CHANNEL_PRESETS)},
    "convs_per_stage": {"type": "categorical", "choices": [1, 2]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
}

SMOKE_FORCED = {k: SMOKE[k] for k in ("channels", "batch_size")}


def suggest(trial):
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def apply_params(base, params):
    cfg = {**base, **params}
    cfg["channels"] = list(CHANNEL_PRESETS[params["channels"]])
    return cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, help="finished trials wanted in total, all sessions (default 30; smoke 2)")
    parser.add_argument("--epochs", type=int, help="epochs per trial (default 12; smoke 2)")
    parser.add_argument("--timeout-min", type=float, help="stop starting new trials after this many minutes")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="2 trials x 2 epochs on tiny synthetic data")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    epochs = args.epochs or (2 if args.smoke else 12)
    n_trials = args.n_trials or (2 if args.smoke else 30)
    disable_wandb_for_smoke(args.smoke)
    name = study_name(CLASSIFIER_STUDY, args.smoke)
    trial_base = {**DEFAULTS, "epochs": epochs, "patience": epochs, "warmup_epochs": 1,
                  "num_workers": args.num_workers}
    final_base = {**DEFAULTS, "num_workers": None}
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    if args.smoke:
        trial_base.update({**SMOKE, "epochs": epochs, "patience": epochs})
        final_base.update(SMOKE)

    study, sync = create_study(name, direction="maximize", pruner=pruner, n_startup_trials=10, fresh=args.smoke)
    data = load_data(args.smoke)

    def objective(trial):
        cfg = apply_params(trial_base, suggest(trial))
        if args.smoke:  # keep the smoke model and batches tiny whatever was sampled
            cfg.update(SMOKE_FORCED)
        result = train_classifier(cfg, trial=trial, smoke=args.smoke, data=data)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("val_accuracy", result["best_metrics"]["accuracy"])
        return result["best_score"]

    timeout = args.timeout_min * 60 if args.timeout_min else None
    run_study(study, objective, n_trials, callbacks=[sync], timeout=timeout)

    report_dir = get_dir("OUTPUT_DIR", TASK, "optuna", name)
    save_study_report(study, report_dir)
    (report_dir / "search_space.json").write_text(json.dumps(
        {"search_space": SEARCH_SPACE, "channel_presets": CHANNEL_PRESETS, "epochs_per_trial": trial_base["epochs"],
         "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=3)", "sampler": "TPE(seed=42, n_startup_trials=10)",
         "objective": "best validation macro-F1 within the trial"}, indent=2))
    best = study.best_trial  # raises if no trial completed
    cfg = apply_params(final_base, best.params)
    if args.smoke:
        cfg.update(SMOKE_FORCED)
    path = save_config(cfg, best_config_path(name))
    print(f"[optuna] best trial {best.number}: macro-F1 {best.value:.4f} params {best.params}")
    print(f"[optuna] final-run config written to {path}")
    return study


if __name__ == "__main__":
    main()
