"""Optuna helpers: studies that survive Colab disconnects, and the report artefacts of a study.

SQLite over Google Drive's FUSE mount is slow and its file locking is unreliable, so
the study database lives on local disk and a consistent snapshot is copied to
OPTUNA_DIR (Drive on Colab) after every finished trial. On restart the snapshot is
restored first, so a study continues where it stopped.
"""
import json
import os
import shutil
import sqlite3
import tempfile
import warnings
from contextlib import closing
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState

from .paths import get_dir

FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)


def _local_dir():
    path = Path(os.environ.get("OPTUNA_LOCAL_DIR") or Path(tempfile.gettempdir()) / "optuna_local")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snapshot(src, dst):
    """Copy an SQLite database consistently (backup API), then move it into place."""
    local_tmp = src.with_name(src.name + ".snapshot")
    # closing(): a sqlite3 connection used as a context manager commits but stays open
    with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(local_tmp)) as target:
        source.backup(target)
    remote_tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(local_tmp, remote_tmp)
    os.replace(remote_tmp, dst)
    local_tmp.unlink()


def create_study(name, direction="maximize", seed=42, pruner=None, n_startup_trials=10, fresh=False):
    """Create or resume a study. Returns (study, sync) where sync is an Optuna callback.

    fresh=True deletes any previous copy first (used by smoke tests).
    """
    remote = get_dir("OPTUNA_DIR") / f"{name}.db"
    local = _local_dir() / f"{name}.db"
    same_file = remote.resolve() == local.resolve()
    if fresh:
        for path in {local, remote}:
            path.unlink(missing_ok=True)
    elif remote.exists() and not same_file:
        shutil.copyfile(remote, local)

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{local.as_posix()}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    # Only one process runs a study, so a RUNNING trial at start-up was interrupted
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study._storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)

    def sync(study, trial):
        if not same_file:
            _snapshot(local, remote)

    return study, sync


def run_study(study, objective, n_trials, callbacks=(), timeout=None):
    """Run trials until `n_trials` have finished (completed or pruned), counting earlier sessions."""
    done = len(study.get_trials(deepcopy=False, states=FINISHED))
    remaining = max(0, n_trials - done)
    print(f"[optuna] {study.study_name}: {done} finished trials, running {remaining} more")
    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=list(callbacks), timeout=timeout, gc_after_trial=True)
    return study


def study_summary(study):
    states = [t.state for t in study.get_trials(deepcopy=False)]
    summary = {
        "study": study.study_name,
        "direction": study.direction.name.lower(),
        "n_complete": states.count(TrialState.COMPLETE),
        "n_pruned": states.count(TrialState.PRUNED),
        "n_failed": states.count(TrialState.FAIL),
    }
    if summary["n_complete"]:
        best = study.best_trial
        summary.update(best_trial=best.number, best_value=best.value, best_params=best.params,
                       best_user_attrs=best.user_attrs)
    return summary


def save_study_report(study, out_dir):
    """Write trials.csv, summary.json and the standard Optuna plots to out_dir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(study_summary(study), indent=2, default=str))

    plots = {
        "optimization_history": ovm.plot_optimization_history,
        "param_importances": ovm.plot_param_importances,
        "parallel_coordinate": ovm.plot_parallel_coordinate,
        "slice": ovm.plot_slice,
        "intermediate_values": ovm.plot_intermediate_values,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        for name, plot in plots.items():
            try:
                ax = plot(study)
                fig = np.ravel(ax)[0].figure
                fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:  # e.g. importances need >= 2 completed trials
                print(f"[optuna] skipped {name} plot: {exc}")
