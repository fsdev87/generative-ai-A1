"""Session setup for the Colab task notebooks (see docs/CONVENTIONS.md).

Mounts Drive, points every directory variable at it, installs requirements, makes
sure the preprocessed datasets exist and logs in to Weights & Biases. Environment
variables set here are inherited by the `!python -m ...` commands of the notebook.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DRIVE = "/content/drive/MyDrive/GenAI_A1"
REPO_DIR = Path(__file__).resolve().parents[2]


def _run(*args):
    subprocess.run([str(a) for a in args], check=True, cwd=REPO_DIR)


def setup(fs2k=False, pets=True):
    from google.colab import drive, userdata

    drive.mount("/content/drive")
    env = {
        "DATA_DIR": "/content/data",
        "CACHE_DIR": f"{PROJECT_DRIVE}/cache",
        "CKPT_DIR": f"{PROJECT_DRIVE}/checkpoints",
        "OPTUNA_DIR": f"{PROJECT_DRIVE}/optuna",
        "ONNX_DIR": f"{PROJECT_DRIVE}/onnx",
        "OUTPUT_DIR": f"{PROJECT_DRIVE}/outputs",
        "OPTUNA_LOCAL_DIR": "/content/optuna",
    }
    os.environ.update(env)
    _run(sys.executable, "-m", "pip", "install", "-q", "-r", REPO_DIR / "requirements.txt")

    archives = f"{PROJECT_DRIVE}/archives"
    if pets and not Path(env["CACHE_DIR"], "pets_train_128.npy").exists():
        _run(sys.executable, "scripts/download_data.py", "--only", "pets",
             "--archive-dir", archives, "--data-dir", env["DATA_DIR"])
        _run(sys.executable, "scripts/prepare_pets.py",
             "--pets-dir", f"{env['DATA_DIR']}/oxford_pets", "--cache-dir", env["CACHE_DIR"])
    if fs2k:
        _run(sys.executable, "scripts/download_data.py", "--only", "fs2k",
             "--archive-dir", archives, "--data-dir", env["DATA_DIR"])

    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU (Runtime > Change runtime type)"
    print(f"torch {torch.__version__} | {gpu}")
    for key, value in env.items():
        print(f"{key} = {value}")
