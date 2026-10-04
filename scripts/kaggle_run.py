"""Run all remaining training on a Kaggle GPU session (Colab's free GPU quota ran out).

    python scripts/kaggle_run.py            # set up data, then run both pipelines
    python scripts/kaggle_run.py --pack     # zip the ONNX models and outputs for download
    python scripts/kaggle_run.py --smoke    # quick CPU check of the whole orchestration

Task 1 was finished on Colab. Task 2 is retrained from the configurations its Colab Optuna
studies selected (configs/task2_*.yaml); Tasks 3 and 4 run their own (shorter) Optuna studies.

With two GPUs (Kaggle "GPU T4 x2") the work is split so both GPUs stay busy:
  GPU 0: Task 2 classifier, salt and blur specialists -> (waits for occlusion) -> routing
         evaluation and ONNX export -> Task 3 (needs all Task 2 models)
  GPU 1: Task 2 occlusion specialist -> Task 4
With one GPU everything runs in that order on it.

Every step is one command whose output goes to LOG_DIR/<pipeline>.log. Finished steps leave a
marker file and are skipped when the script is run again, so after an interruption just run it
again: training and Optuna resume from their checkpoints and study databases.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working/genai"))
SCRATCH = Path(os.environ.get("KAGGLE_SCRATCH", "/tmp/genai"))
PY = sys.executable

ENV = {
    "DATA_DIR": SCRATCH / "data",
    "CACHE_DIR": SCRATCH / "cache",
    "OPTUNA_LOCAL_DIR": SCRATCH / "optuna_local",
    "CKPT_DIR": WORK / "checkpoints",
    "OPTUNA_DIR": WORK / "optuna",
    "ONNX_DIR": WORK / "onnx",
    "OUTPUT_DIR": WORK / "outputs",
}
LOG_DIR = WORK / "logs"
MARKERS = WORK / "markers"


def module(name, *args):
    return [PY, "-m", name, *map(str, args)]


def retrain_pipelines(smoke):
    """Second Kaggle run: Task 1 skip-connection ablation and a larger Task 4 search.

    Task 1 is retrained three times from the configuration its Colab search selected
    (configs/task1.yaml): without skips and with one limited skip at 16x16 or 32x32; every
    variant is evaluated and exported, and the comparison table is written. Task 4 runs a larger
    Optuna study (narrowed discriminator learning rate) and a 200-epoch final run.
    """
    out = ENV["OUTPUT_DIR"]
    s = ["--smoke"] if smoke else []
    cfg = [] if smoke else ["--config", REPO / "configs/task1.yaml", "--resume"]

    def t1(variant, skips):
        skip_args = ["--skip-resolutions", *skips] if skips else []
        name = "udae" if not skips else f"udae_skip{skips[0]}"
        return [
            (f"t1-{name}", module("src.task1.train", *cfg, *skip_args, *s)),
            (f"t1-{name}-eval", module("src.task1.evaluate", "--variant", name, *s)),
        ]

    if smoke:
        t4_search, t4_train = module("src.task4.optuna_search", "--smoke"), module("src.task4.train", "--smoke")
    else:
        t4_search = module("src.task4.optuna_search", "--n-trials", 18, "--epochs", 12, "--timeout-min", 35)
        t4_train = module("src.task4.train", "--config", out / "task4" / "best_config.yaml", "--resume",
                          "--epochs", 200)
    # GPU 0: the three Task 1 variants (~25 min each); GPU 1: Task 4 (~70 min) -- balanced
    gpu0 = [
        *t1("udae", []),
        *t1("udae_skip16", [16]),
        *t1("udae_skip32", [32]),
        ("t1-compare", module("src.task1.evaluate", "--compare", "udae", "udae_skip16", "udae_skip32", *s)),
        ("t1-export-udae", module("src.task1.export_onnx", *s)),
        ("t1-export-skip16", module("src.task1.export_onnx", "--variant", "udae_skip16", *s)),
        ("t1-export-skip32", module("src.task1.export_onnx", "--variant", "udae_skip32", *s)),
    ]
    gpu1 = [
        ("t4-optuna", t4_search),
        ("t4-train", t4_train),
        ("t4-eval", module("src.task4.evaluate", *s)),
        ("t4-export", module("src.task4.export_onnx", *s)),
    ]
    # Third job, sharing GPU 1 with Task 4: Task 2 retrained with skip specialists, then Task 3
    if smoke:
        cls = module("src.task2.train_classifier", "--smoke")
        spec = {t: module("src.task2.train_specialist", "--type", t, "--smoke") for t in ("salt", "blur", "occlusion")}
        t3_search, t3_train = module("src.task3.optuna_search", "--smoke"), module("src.task3.train", "--smoke")
    else:
        cls = module("src.task2.train_classifier", "--config", REPO / "configs/task2_classifier.yaml", "--resume")
        spec = {t: module("src.task2.train_specialist", "--type", t, "--config",
                          REPO / "configs/task2_specialists_skip.yaml", "--resume") for t in ("salt", "blur", "occlusion")}
        t3_search = module("src.task3.optuna_search", "--n-trials", 6, "--trial-warmup-epochs", 1,
                           "--trial-joint-epochs", 3, "--final-warmup-epochs", 2, "--final-joint-epochs", 10,
                           "--timeout-min", 25)
        t3_train = module("src.task3.train", "--config", out / "task3" / "best_config.yaml", "--resume")
    gpu1b = [
        ("t2-classifier", cls),
        ("t2-classifier-eval", module("src.task2.evaluate_classifier", *s)),
        ("t2-salt", spec["salt"]), ("t2-blur", spec["blur"]), ("t2-occlusion", spec["occlusion"]),
        ("t2-routing-eval", module("src.task2.evaluate_routing", *s)),
        ("t2-export", module("src.task2.export_onnx", *s)),
        ("t3-optuna", t3_search),
        ("t3-train", t3_train),
        ("t3-eval", module("src.task3.evaluate", *s)),
        ("t3-compare", module("src.task3.compare", *s)),
        ("t3-mixed", module("src.task3.evaluate_mixed", *s)),
        ("t3-export", module("src.task3.export_onnx", *s)),
    ]
    return gpu0, gpu1, gpu1b


def pipelines(smoke):
    """First Kaggle run (2026-10-04): Task 2 final models, Task 3 and Task 4.

    (GPU-0 steps, GPU-1 steps). A step is (name, command) or ("wait", step_name).
    """
    out = ENV["OUTPUT_DIR"]
    if smoke:
        t2_cls = [module("src.task2.train_classifier", "--smoke")]
        t2_spec = {t: module("src.task2.train_specialist", "--type", t, "--smoke") for t in ("salt", "blur", "occlusion")}
        t3_search = module("src.task3.optuna_search", "--smoke")
        t3_train = module("src.task3.train", "--smoke")
        t4_search = module("src.task4.optuna_search", "--smoke")
        t4_train = module("src.task4.train", "--smoke")
        flag = ["--smoke"]
    else:
        t2_cls = [module("src.task2.train_classifier", "--config", REPO / "configs/task2_classifier.yaml", "--resume")]
        t2_spec = {t: module("src.task2.train_specialist", "--type", t, "--config",
                             REPO / "configs/task2_specialists.yaml", "--resume") for t in ("salt", "blur", "occlusion")}
        # Shorter studies than the Colab notebooks plan (deadline): 6 trials of 1 warm-up + 3 joint epochs,
        # final run 2 warm-up + 10 joint epochs
        t3_search = module("src.task3.optuna_search", "--n-trials", 6, "--trial-warmup-epochs", 1,
                           "--trial-joint-epochs", 3, "--final-warmup-epochs", 2, "--final-joint-epochs", 10,
                           "--timeout-min", 25)
        t3_train = module("src.task3.train", "--config", out / "task3" / "best_config.yaml", "--resume")
        t4_search = module("src.task4.optuna_search", "--n-trials", 10, "--epochs", 8, "--timeout-min", 30)
        t4_train = module("src.task4.train", "--config", out / "task4" / "best_config.yaml", "--resume")
        flag = []

    gpu0 = [
        ("t2-classifier", t2_cls[0]),
        ("t2-classifier-eval", module("src.task2.evaluate_classifier", *flag)),
        ("t2-salt", t2_spec["salt"]),
        ("t2-blur", t2_spec["blur"]),
        ("wait", "t2-occlusion"),
        ("t2-routing-eval", module("src.task2.evaluate_routing", *flag)),
        ("t2-export", module("src.task2.export_onnx", *flag)),
        ("t3-optuna", t3_search),
        ("t3-train", t3_train),
        ("t3-eval", module("src.task3.evaluate", *flag)),
        ("t3-compare", module("src.task3.compare", *flag)),
        ("t3-mixed", module("src.task3.evaluate_mixed", *flag)),
        ("t3-export", module("src.task3.export_onnx", *flag)),
    ]
    gpu1 = [
        ("t2-occlusion", t2_spec["occlusion"]),
        ("t4-optuna", t4_search),
        ("t4-train", t4_train),
        ("t4-eval", module("src.task4.evaluate", *flag)),
        ("t4-export", module("src.task4.export_onnx", *flag)),
    ]
    return gpu0, gpu1


class Pipeline(threading.Thread):
    def __init__(self, name, steps, gpu, others=()):
        super().__init__(name=name, daemon=True)
        self.steps, self.gpu, self.others = steps, gpu, list(others)
        self.current, self.failed, self.log = "starting", None, LOG_DIR / f"{name}.log"

    def run(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        for name, cmd in self.steps:
            if name == "wait":
                self.current = f"waiting for {cmd}"
                while not (MARKERS / f"{cmd}.done").exists():
                    if self.others and not any(p.is_alive() for p in self.others):
                        self.failed = f"{cmd} never finished (see the other pipeline's log)"
                        return
                    time.sleep(15)
                continue
            if (MARKERS / f"{name}.done").exists():
                say(f"[{self.name}] {name}: already done, skipping")
                continue
            self.current = name
            say(f"[{self.name}] {name}: started")
            start = time.time()
            with open(self.log, "a") as log:
                log.write(f"\n===== {name}: {' '.join(map(str, cmd))}\n")
                log.flush()
                code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            minutes = (time.time() - start) / 60
            if code != 0:
                self.failed = f"{name} failed (exit {code}) after {minutes:.1f} min; see {self.log}"
                say(f"[{self.name}] FAILED: {self.failed}")
                return
            (MARKERS / f"{name}.done").touch()
            say(f"[{self.name}] {name}: done in {minutes:.1f} min")
        self.current = "finished"


def say(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def last_line(path):
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip() and "wandb:" not in l]
        return lines[-1][:150] if lines else ""
    except FileNotFoundError:
        return ""


def run(cmd, **kwargs):
    say("$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=REPO, check=True, **kwargs)


def setup(smoke):
    """Packages, both datasets and their caches (skipped when already present)."""
    if not smoke:
        run([PY, "-m", "pip", "install", "-q", "-U", "optuna", "onnx", "onnxruntime", "gdown", "wandb"])
        if not (ENV["CACHE_DIR"] / "pets_train_128.npy").exists():
            run([PY, "scripts/download_data.py", "--archive-dir", SCRATCH / "archives", "--data-dir", ENV["DATA_DIR"]])
            run([PY, "scripts/prepare_pets.py", "--pets-dir", ENV["DATA_DIR"] / "oxford_pets",
                 "--cache-dir", ENV["CACHE_DIR"]])
        run([PY, "scripts/prepare_fs2k.py"])
    if not os.environ.get("WANDB_API_KEY") and not smoke:
        raise SystemExit("WANDB_API_KEY is not set: add it as a Kaggle secret and load it before running.")


def pack():
    """Two zips in WORK: the ONNX models (needed by the app) and the outputs (figures, tables)."""
    for name, folder, skip in (("genai_models.zip", ENV["ONNX_DIR"], "smoke"),
                               ("genai_outputs.zip", ENV["OUTPUT_DIR"], "smoke")):
        target = WORK / name
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and skip not in path.relative_to(folder).parts:
                    zf.write(path, Path(folder.name) / path.relative_to(folder))
        say(f"{target} ({target.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run of every step (W&B disabled)")
    parser.add_argument("--pack", action="store_true", help="only zip the results for download")
    parser.add_argument("--gpus", type=int, help="override the number of GPUs used (testing)")
    parser.add_argument("--redo", action="store_true", help="forget finished-step markers")
    parser.add_argument("--plan", choices=("retrain", "initial"), default="retrain",
                        help="retrain: Task 1 skip ablation + larger Task 4 search (default); "
                             "initial: the first run (Task 2 final models, Tasks 3 and 4)")
    args = parser.parse_args()

    os.environ.update({k: str(v) for k, v in ENV.items()})
    for path in (*ENV.values(), LOG_DIR, MARKERS):
        Path(path).mkdir(parents=True, exist_ok=True)
    if args.pack:
        return pack()
    if args.redo:
        shutil.rmtree(MARKERS)
        MARKERS.mkdir()

    setup(args.smoke)
    if args.gpus is not None:
        n_gpus = args.gpus
    else:
        import torch

        n_gpus = torch.cuda.device_count()
    say(f"GPUs: {n_gpus}; logs in {LOG_DIR}; results in {WORK}")
    if n_gpus == 0 and not args.smoke:
        raise SystemExit("No GPU: in the notebook settings choose Accelerator -> GPU T4 x2.")

    jobs = (retrain_pipelines if args.plan == "retrain" else pipelines)(args.smoke)
    gpu0, gpu1, extra = jobs if len(jobs) == 3 else (*jobs, [])
    if n_gpus >= 2:
        b = Pipeline("gpu1", gpu1, gpu=1)
        a = Pipeline("gpu0", gpu0, gpu=0, others=[b])
        workers = [a, b] + ([Pipeline("gpu1b", extra, gpu=1)] if extra else [])
    else:
        # One GPU: run GPU 1's first block before GPU 0's wait so nothing waits forever
        cut0 = next((i for i, (name, _) in enumerate(gpu0) if name == "wait"), len(gpu0))
        cut1 = 0 if args.plan == "retrain" else 1
        order = gpu0[:cut0] + gpu1[:cut1] + gpu0[cut0:] + gpu1[cut1:] + extra
        workers = [Pipeline("gpu", order, gpu=0 if n_gpus else None)]
    for w in workers:
        w.start()
    while any(w.is_alive() for w in workers):
        time.sleep(1 if args.smoke else 120)
        if not args.smoke:
            for w in workers:
                say(f"[{w.name}] {w.current} | {last_line(w.log)}")
    failed = [f"{w.name}: {w.failed}" for w in workers if w.failed]
    if failed:
        raise SystemExit("Some steps failed:\n  " + "\n  ".join(failed))
    say("All steps finished. Next: python scripts/kaggle_run.py --pack, then download the zips.")


if __name__ == "__main__":
    main()
