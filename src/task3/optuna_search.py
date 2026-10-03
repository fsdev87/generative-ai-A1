"""Task 3: Optuna search over the soft-MoE training hyperparameters (study "task3_moe").

Searched (the brief's list): joint fine-tuning learning rate, temperature tau, CE weight,
balance weight and the reconstruction weighting (lambda_l1 = alpha, lambda_ssim = 1 - alpha).
Trial 0 is the brief's starting configuration. Every trial runs a short schedule
(--trial-warmup-epochs + --trial-joint-epochs) and is its own W&B run (group task3-optuna).

Pruning: the median pruner (poor configurations) and routing collapse on the validation set
(routing.detect_collapse); the reason is stored in the trial attribute "prune_reason".
Objective: the validation score of train.py (restoration_score, per-image PSNR capped at 40 dB).

    python -m src.task3.optuna_search --n-trials 20 --timeout-min 90
    python -m src.task3.optuna_search --smoke

The study resumes after a disconnect (finished trials are counted, interrupted ones re-run).
Writes OUTPUT_DIR/task3/optuna/task3_moe/ (report) and OUTPUT_DIR/task3/best_config.yaml.
"""
import argparse
import os

import optuna
from optuna.trial import TrialState

from src.common.optuna_utils import create_study, run_study, save_study_report, study_summary
from src.common.tracking import wandb_run
from src.common.utils import get_device

from .config import DEFAULTS, OPTUNA_GROUP, best_config_path, output_dir, resolve_config, save_yaml_config, study_name
from .train import load_data, load_init, train_moe

SEARCH_SPACE = {
    # name: (low, high, log scale)
    "lr": (1e-5, 3e-4, True),            # joint gate LR (experts x0.1); upper end = the warm-up LR
    "tau": (0.3, 3.0, True),             # sharp (near hard routing) .. soft; 1 = classifier probabilities
    "lambda_ce": (0.01, 1.0, True),      # one decade either side of the brief's 0.1
    "lambda_balance": (1e-3, 1e-1, True),  # one decade either side of the brief's 0.01
    "alpha": (0.5, 0.95, False),         # lambda_l1 = alpha, lambda_ssim = 1 - alpha (brief: 0.8)
}
BASELINE = {"lr": 1e-4, "tau": 1.0, "lambda_ce": 0.1, "lambda_balance": 0.01, "alpha": 0.8}


def suggest(trial):
    return {name: trial.suggest_float(name, low, high, log=log) for name, (low, high, log) in SEARCH_SPACE.items()}


def params_to_config(params, base):
    config = dict(base)
    config.update({k: v for k, v in params.items() if k != "alpha"})
    config["lambda_l1"] = params["alpha"]
    config["lambda_ssim"] = round(1.0 - params["alpha"], 12)
    return config


def enqueue_baseline(study):
    """Run the brief's starting values as a trial unless one with these parameters already exists."""
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.WAITING)
    if not any(t.params == BASELINE for t in study.get_trials(deepcopy=False, states=states)):
        study.enqueue_trial(BASELINE, user_attrs={"source": "brief starting values"})


def make_objective(base, data, init, device, smoke):
    def objective(trial):
        config = params_to_config(suggest(trial), base)
        with wandb_run(f"trial-{trial.number}", OPTUNA_GROUP, config={**config, "trial": trial.number},
                       job_type="optuna", tags=["task3", "optuna"] + (["smoke"] if smoke else [])) as run:
            trial.set_user_attr("wandb_url", getattr(run, "url", None))
            try:
                result = train_moe(config, trial=trial, data=data, init=init, run=run, device=device,
                                   smoke=smoke, verbose=True)
            except optuna.TrialPruned:
                run.summary["pruned"] = True
                run.summary["prune_reason"] = trial.user_attrs.get("prune_reason")
                raise
        trial.set_user_attr("prune_reason", None)
        return result["best_score"]

    return objective


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=20, help="total finished trials, counting earlier sessions")
    parser.add_argument("--timeout-min", type=float, default=None, help="stop starting new trials after this many "
                                                                        "minutes (per session)")
    parser.add_argument("--trial-warmup-epochs", type=int, default=1)
    parser.add_argument("--trial-joint-epochs", type=int, default=4)
    parser.add_argument("--final-warmup-epochs", type=int, default=DEFAULTS["warmup_epochs"],
                        help="warm-up epochs written to best_config.yaml for the final run")
    parser.add_argument("--final-joint-epochs", type=int, default=DEFAULTS["joint_epochs"],
                        help="joint epochs written to best_config.yaml for the final run")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--report-only", action="store_true", help="only rewrite the report and best config")
    parser.add_argument("--smoke", action="store_true", help="2 tiny trials on synthetic data, W&B disabled")
    return parser


def write_best_config(study, args, smoke):
    best = study.best_trial
    config = params_to_config(best.params, resolve_config(smoke=smoke, cli={"batch_size": args.batch_size}))
    config["warmup_epochs"], config["joint_epochs"] = args.final_warmup_epochs, args.final_joint_epochs
    if smoke:
        config["warmup_epochs"], config["joint_epochs"] = 1, 1
    header = (f"Task 3 final-run config from Optuna study {study.study_name}: best trial {best.number} "
              f"(val score {best.value:.4f} after {best.user_attrs.get('epochs_run')} short-schedule epochs).\n"
              f"Searched: {', '.join(SEARCH_SPACE)} (alpha -> lambda_l1 = alpha, lambda_ssim = 1 - alpha).\n"
              "Usage: python -m src.task3.train --config <this file> --resume")
    return save_yaml_config(config, best_config_path(smoke), header)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        os.environ["WANDB_MODE"] = "disabled"
        args.n_trials = min(args.n_trials, 2)
        args.trial_warmup_epochs, args.trial_joint_epochs = 1, 1
    name = study_name(args.smoke)
    # Smoke studies start fresh; the real study always resumes from its Drive snapshot
    study, sync = create_study(name, direction="maximize", fresh=args.smoke)

    if not args.report_only:
        base = resolve_config(smoke=args.smoke, cli={
            "warmup_epochs": args.trial_warmup_epochs, "joint_epochs": args.trial_joint_epochs,
            "batch_size": args.batch_size})
        device = get_device()
        data, init = load_data(args.smoke), load_init(args.smoke)
        enqueue_baseline(study)
        run_study(study, make_objective(base, data, init, device, args.smoke), args.n_trials, callbacks=[sync],
                  timeout=args.timeout_min * 60 if args.timeout_min else None)

    report_dir = output_dir(args.smoke, "optuna", name)
    save_study_report(study, report_dir)
    summary = study_summary(study)
    reasons = [t.user_attrs.get("prune_reason", "") for t in study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))]
    n_collapse = sum(1 for r in reasons if r and r.startswith("collapse"))
    print(f"[task3] study {name}: {summary['n_complete']} complete, {summary['n_pruned']} pruned "
          f"({n_collapse} for routing collapse), {summary['n_failed']} failed; report in {report_dir}")
    if summary["n_complete"]:
        path = write_best_config(study, args, args.smoke)
        print(f"[task3] best trial {summary['best_trial']}: score {summary['best_value']:.4f} "
              f"params {summary['best_params']} -> {path}")
    else:
        print("[task3] no completed trial yet: best config not written")
    return study


if __name__ == "__main__":
    main()
