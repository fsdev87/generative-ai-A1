"""Tests for the Task 4 pipeline (src/task4 train / Optuna / evaluate / export).

End-to-end smoke runs on synthetic FS2K-shaped data, with every directory variable pointed
at a temporary folder and W&B disabled. Run: python -m pytest tests/test_task4_train.py -q
"""
import json

import numpy as np
import pytest
import torch

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.data.fs2k import NUM_STYLES
from src.task4 import evaluate, export_onnx, optuna_search, train
from src.task4.config import (
    DEFAULTS, SMOKE_OVERRIDES, best_config_path, checkpoint_dir, load_yaml_config, output_dir, resolve_config,
)
from src.task4.metrics import edge_energy, sketch_metrics, sketch_score
from src.task4.train import LOSS_KEYS


@pytest.fixture(scope="module", autouse=True)
def task4_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("task4")
    with pytest.MonkeyPatch.context() as mp:
        for var in ("CKPT_DIR", "OUTPUT_DIR", "ONNX_DIR", "OPTUNA_DIR", "OPTUNA_LOCAL_DIR", "CACHE_DIR"):
            mp.setenv(var, str(root / var.lower()))
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        mp.setenv("WANDB_DIR", str(root))
        yield root


@pytest.fixture(scope="module")
def trained():
    return train.main(["--smoke"])


@pytest.fixture(scope="module")
def evaluated(trained):
    return evaluate.main(["--smoke"])


# --------------------------------------------------------------------------- #
# Configuration, schedule and metrics
# --------------------------------------------------------------------------- #
def test_config_precedence_and_validation(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr_g: 0.0005\nbase_channels: 48\nlambda_l1: 50\n")
    config = resolve_config(path, cli={"lr_g": 0.001}, smoke=False)
    assert config["lr_g"] == 0.001 and config["base_channels"] == 48 and config["lambda_l1"] == 50
    smoke = resolve_config(path, smoke=True)
    assert smoke["base_channels"] == SMOKE_OVERRIDES["base_channels"] and smoke["epochs"] == 2
    assert smoke["lambda_l1"] == 50  # a YAML value survives the smoke overrides
    path.write_text("learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_yaml_config(path)


def test_generator_and_discriminator_share_width_and_style_dim():
    from src.task4.config import discriminator_config, generator_config

    config = resolve_config(cli={"base_channels": 48, "style_dim": 24})
    g, d = generator_config(config), discriminator_config(config)
    assert g["base_channels"] == d["base_channels"] == 48
    assert g["style_dim"] == d["style_dim"] == 24
    assert g["num_styles"] == d["num_styles"] == NUM_STYLES
    assert g["out_channels"] == d["sketch_channels"]


def test_linear_decay_schedule():
    factor = train.linear_decay(epochs=100, decay_fraction=0.5)
    assert factor(0) == 1.0 and factor(49) == 1.0
    assert factor(50) == pytest.approx(1.0) and factor(75) == pytest.approx(0.5)
    assert factor(99) == pytest.approx(0.02) and factor(100) == 0.0
    values = [factor(e) for e in range(100)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    short = train.linear_decay(epochs=2, decay_fraction=0.5)  # schedule stays valid for tiny runs
    assert short(0) == 1.0 and 0.0 <= short(1) <= 1.0


def test_sketch_metrics_and_score():
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    same = sketch_metrics(x, x)
    assert float(same["l1"].max()) == pytest.approx(0, abs=1e-6)
    assert float(same["ssim"].min()) == pytest.approx(1, abs=1e-5)
    assert float(same["edge_ratio"].mean()) == pytest.approx(1, abs=1e-4)
    assert sketch_score(1.0, 0.0) == 1.0 and sketch_score(0.0, 1.0) == 0.0
    # a blurred sketch keeps L1 small but loses edge energy: the sharpness diagnostic catches it
    sharp = torch.full((1, 1, 32, 32), -1.0)
    sharp[..., ::4] = 1.0
    blurred = torch.nn.functional.avg_pool2d(sharp, 3, 1, 1)
    assert float(edge_energy(blurred)) < float(edge_energy(sharp))
    assert float(sketch_metrics(blurred, sharp)["edge_ratio"]) < 1.0


# --------------------------------------------------------------------------- #
# Fixed validation samples
# --------------------------------------------------------------------------- #
def test_fixed_sample_indices_are_stable_and_cover_every_style():
    val = train.load_data(smoke=True)["val"]
    first = train.fixed_sample_indices(val)
    assert first == train.fixed_sample_indices(val)  # same entries in every epoch and every run
    styles = [int(val["style"][i]) for i in first]
    assert sorted(styles) == sorted(list(range(NUM_STYLES)) * train.SAMPLES_PER_STYLE)
    assert len(set(first)) == len(first)


def test_sample_grid_uses_the_same_photos_for_different_models():
    from src.task4.models import UNetGenerator

    val = train.load_data(smoke=True)["val"]
    idx = train.fixed_sample_indices(val)
    device = torch.device("cpu")
    torch.manual_seed(0)
    grid_a = train.sample_grid(UNetGenerator(base_channels=4, style_dim=4), val, idx, device)
    torch.manual_seed(1)
    grid_b = train.sample_grid(UNetGenerator(base_channels=4, style_dim=4), val, idx, device)
    assert grid_a.shape == grid_b.shape
    photo_and_gt = slice(0, 2 * (128 + 2) + 2)  # the first two columns: photo and ground truth
    assert np.array_equal(grid_a[:, photo_and_gt], grid_b[:, photo_and_gt])
    assert not np.array_equal(grid_a, grid_b)  # the generated columns do differ
    assert train.sample_caption(val, idx).startswith("photo | ground truth |")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_smoke_run_and_separate_losses(trained):
    assert trained["epochs_run"] == 2 and trained["best_epoch"] in (1, 2)
    assert set(trained["n_params"]) == {"generator", "discriminator"}
    for record in trained["history"]:
        for key in LOSS_KEYS:  # D real, D fake, G adversarial and G L1 are tracked separately
            assert np.isfinite(record[f"train_{key}"])
        assert record["train_d_real"] != record["train_d_fake"]
        for key in ("l1", "ssim", "psnr", "edge_ratio", "score", "style_gap"):
            assert f"val_{key}" in record
    ckpt_dir = checkpoint_dir(smoke=True)
    assert (ckpt_dir / "last.pt").exists() and (ckpt_dir / "best.pt").exists()


def test_checkpoint_contents_allow_rebuild_and_resume(trained):
    from src.common.checkpoint import build_model
    from src.task4.models import UNetGenerator

    last = load_checkpoint(checkpoint_dir(smoke=True) / "last.pt")
    best = load_checkpoint(checkpoint_dir(smoke=True) / "best.pt")
    generator = build_model(UNetGenerator, best)  # best.pt rebuilds the generator for the export
    assert generator.config == best["model_config"]
    assert last["finished"] is True and last["epoch"] == 2
    for key in ("d_state", "opt_g_state", "opt_d_state", "sch_g_state", "sch_d_state", "history",
                "best_epoch", "best_score", "global_step"):
        assert key in last
    assert last["global_step"] > 0


def test_resume_continues_from_an_interrupted_checkpoint(trained):
    path = checkpoint_dir(smoke=True) / "last.pt"
    finished = load_checkpoint(path)
    assert train.main(["--smoke", "--resume"])["already_finished"] is True

    interrupted = {**finished, "epoch": 1, "finished": False, "history": finished["history"][:1]}
    save_checkpoint(path, **interrupted)
    result = train.main(["--smoke", "--resume"])
    assert result.get("already_finished") is None
    assert result["epochs_run"] == 2 and [r["epoch"] for r in result["history"]] == [1, 2]
    save_checkpoint(path, **finished)  # leave the finished checkpoint for the other tests


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def test_optuna_smoke_search_writes_report_and_best_config():
    study, path = optuna_search.run_search(n_trials=2, epochs=1, smoke=True, fresh=True)
    assert study.study_name == "task4_cgan_smoke"
    assert len(study.get_trials(deepcopy=False)) == 2
    assert study.trials[0].params == optuna_search.REFERENCE_PARAMS  # the brief's starting point runs first
    assert path == best_config_path(smoke=True) and path.exists()

    config = load_yaml_config(path)
    for key in optuna_search.SEARCHED:  # every hyperparameter the brief asks for is tuned
        assert key in config
    assert {"lr_g", "lr_d", "batch_size", "base_channels", "dropout", "style_dim", "lambda_l1"} <= set(
        optuna_search.SEARCHED)
    assert "epochs" not in config  # the final run uses train.py's full schedule

    report = output_dir(True, "optuna", study.study_name)
    assert (report / "trials.csv").exists() and (report / "summary.json").exists()
    space = json.loads((report / "search_space.json").read_text())
    assert space["epochs_per_trial"] == 1
    assert "never used" in space["validation_split"]  # trials are scored on validation only
    assert set(space["params"]) == set(optuna_search.SEARCHED)


# --------------------------------------------------------------------------- #
# Evaluation and export
# --------------------------------------------------------------------------- #
def test_evaluation_smoke_run(evaluated):
    out = output_dir(smoke=True)
    assert evaluated["n_test_pairs"] == 12
    assert {r["style_name"] for r in evaluated["by_style"]} == {"Style 1", "Style 2", "Style 3"}
    assert evaluated["by_source"] and "style_sensitivity" in evaluated
    for metric in ("l1", "ssim", "psnr", "edge_ratio", "score"):
        assert np.isfinite(evaluated["overall"][metric])
    assert 0.0 <= evaluated["overall"]["l1"] <= 1.0
    records = (out / "eval" / "test_records.csv").read_text().strip().splitlines()
    assert len(records) == evaluated["n_test_pairs"] + 1
    for name in ("test_examples.png", "failure_cases.png", "style_swap.png"):
        assert (out / "figures" / name).stat().st_size > 0
    for name in ("by_style", "by_source", "by_source_style"):
        assert (out / "tables" / f"{name}.csv").exists() and (out / "tables" / f"{name}.tex").exists()
    assert (out / "eval" / "failure_cases.csv").exists()


def test_export_onnx_smoke_run(evaluated):
    import onnxruntime as ort

    card = export_onnx.export(smoke=True)
    assert card["parity"]["passed"] and card["parity"]["max_abs_diff"]["sketch"] < 1e-4
    assert card["opset"] == 17 and card["task"] == "task4_cgan"
    assert [i["name"] for i in card["inputs"]] == ["photo", "style"]
    assert [o["name"] for o in card["outputs"]] == ["sketch"]
    assert "centre square crop" in card["preprocessing"] and "[-1,1]" in card["preprocessing"]
    assert card["metrics"]["test"]["n_test_pairs"] == 12  # evaluate.py's results are in the card

    path = export_onnx.get_dir("ONNX_DIR", "smoke") / "generator.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [(i.name, i.type) for i in session.get_inputs()] == [
        ("photo", "tensor(float)"), ("style", "tensor(int64)")]
    for n in (1, 3):  # dynamic batch axis
        out = session.run(["sketch"], {"photo": np.zeros((n, 3, 128, 128), np.float32),
                                       "style": np.arange(n, dtype=np.int64) % NUM_STYLES})[0]
        assert out.shape == (n, 1, 128, 128) and np.abs(out).max() <= 1.0
