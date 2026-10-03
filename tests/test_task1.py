"""Tests for Task 1 (src/task1). Run: python -m pytest tests/test_task1.py -q

End-to-end smoke runs of train / Optuna / evaluate / export on synthetic data, with every
directory variable pointed at a temporary folder and W&B disabled.
"""
import json

import numpy as np
import optuna
import pytest
import torch
from PIL import Image

from src.common.checkpoint import build_model, load_checkpoint
from src.common.metrics import psnr
from src.data.corruptions import CLASSES, apply_spec, level_spec
from src.data.pets import load_pets
from src.models.autoencoder import ConvAutoencoder
from src.task1 import evaluate, export_onnx, optuna_search, train
from src.task1.baselines import oracle_restore
from src.task1.config import (
    SMOKE_OVERRIDES, bottleneck_info, checkpoint_dir, load_yaml_config, model_config, output_dir, resolve_config, variant_name,
)


@pytest.fixture(scope="module", autouse=True)
def task1_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("task1")
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
# Configuration and helpers
# --------------------------------------------------------------------------- #
def test_config_precedence_and_validation(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 0.002\nbase_channels: 48\nalpha: 0.5\n")
    config = resolve_config(path, cli={"lr": 0.005}, smoke=False)
    assert config["lr"] == 0.005 and config["base_channels"] == 48 and config["alpha"] == 0.5
    smoke = resolve_config(path, smoke=True)
    assert smoke["base_channels"] == 4 and smoke["epochs"] == 2 and smoke["alpha"] == 0.5
    path.write_text("learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_yaml_config(path)


def test_variant_names_and_bottleneck_size():
    assert variant_name([]) == "udae" and variant_name([32, 16]) == "udae_skip16_32"
    info = bottleneck_info(model_config(resolve_config(cli={"latent_channels": 32})))
    assert info["latent_shape"] == [32, 8, 8] and info["latent_dim"] == 2048 and info["compression_ratio"] == 24
    assert info["skip_values"] == 0
    skip = bottleneck_info(model_config(resolve_config(cli={"base_channels": 32, "skip_resolutions": [128]})))
    assert skip["skip_values"] == 32 * 128 * 128  # a full-resolution skip carries more values than the input
    model = ConvAutoencoder(**model_config(resolve_config()))
    assert model.latent_dim == bottleneck_info(model.config)["latent_dim"]


def test_warmup_cosine_schedule():
    f = train.warmup_cosine(warmup_steps=10, total_steps=110, min_ratio=0.01)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.505) and f(110) == pytest.approx(0.01) and f(500) == pytest.approx(0.01)
    decay = [f(s) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


# --------------------------------------------------------------------------- #
# Training and resuming
# --------------------------------------------------------------------------- #
def test_train_smoke_writes_checkpoints(trained):
    ckpt = checkpoint_dir("udae", smoke=True)
    last = load_checkpoint(ckpt / "last.pt")
    assert last["epoch"] == 2 and last["finished"] and len(last["history"]) == 2
    assert {f"val_{c}/psnr" for c in CLASSES} <= set(last["history"][-1])
    best = load_checkpoint(ckpt / "best.pt")
    assert best["epoch"] == trained["best_epoch"] and best["metrics"]["score"] == pytest.approx(trained["best_score"])
    model = build_model(ConvAutoencoder, ckpt / "best.pt").eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 3, 128, 128) and 0 <= out.min() and out.max() <= 1


def test_resume_continues_at_the_next_epoch(monkeypatch):
    args = ["--smoke", "--epochs", "2", "--output-subdir", "resume_test"]
    real_epoch, calls = train.train_one_epoch, []

    def crash_in_epoch_two(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated Colab disconnect")
        return real_epoch(*a, **k)

    monkeypatch.setattr(train, "train_one_epoch", crash_in_epoch_two)
    with pytest.raises(RuntimeError, match="simulated"):
        train.main(args)
    last_path = checkpoint_dir("resume_test", smoke=True) / "last.pt"
    interrupted = load_checkpoint(last_path)
    assert interrupted["epoch"] == 1 and not interrupted["finished"]

    monkeypatch.setattr(train, "train_one_epoch", real_epoch)
    result = train.main(args + ["--resume"])
    state = load_checkpoint(last_path)
    assert [h["epoch"] for h in state["history"]] == [1, 2] and state["finished"]
    assert result["epochs_run"] == 2
    steps = 2 * 4  # 32 synthetic images / batch 8, two epochs: the restored optimizer and schedule kept counting
    assert state["scheduler_state"]["last_epoch"] == steps
    assert all(s["step"] == steps for s in state["optimizer_state"]["state"].values())
    assert train.main(args + ["--resume"])["already_finished"]


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def test_optuna_smoke_writes_report_and_best_config(monkeypatch):
    reports = []

    def fake_report(study, out_dir):  # the shared plotting is tested in test_common and takes ~15 s on CPU
        out_dir.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
        reports.append(out_dir)

    monkeypatch.setattr(optuna_search, "save_study_report", fake_report)
    study, path = optuna_search.main(["--smoke", "--epochs", "1"])
    trials = study.trials
    assert len(trials) == 2 and all(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    assert trials[0].params == optuna_search.REFERENCE_PARAMS  # the brief's alpha = 0.8 reference
    assert {"latent_dim", "compression_ratio", "best_val", "epochs_run"} <= set(trials[1].user_attrs)
    report = output_dir(smoke=True) / "optuna" / "task1_udae_smoke"
    assert reports == [report]
    assert (report / "trials.csv").exists() and (report / "search_space.json").exists()
    space = json.loads((report / "search_space.json").read_text())
    assert set(space["params"]) == set(optuna_search.SEARCHED) and space["epochs_per_trial"] == 1
    config = load_yaml_config(path)
    assert {k: config[k] for k in optuna_search.SEARCHED} == study.best_trial.params
    assert "epochs" not in config  # the final run's length comes from train.py, not the short trials


def test_optuna_pruner_stops_trials():
    small = {k: SMOKE_OVERRIDES[k] for k in ("base_channels", "latent_channels", "batch_size")}
    objective = optuna_search.make_objective(resolve_config(smoke=True), train.load_data(smoke=True), epochs=2,
                                             device=torch.device("cpu"), forced=small)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=10.0))
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED and trial.user_attrs["pruned_reason"] == "pruner"
    assert trial.user_attrs["epochs_run"] == 1 and list(trial.intermediate_values) == [1]


@pytest.mark.parametrize("error, reason", [
    (torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "cuda_oom"),
    (train.NonFiniteLossError("training loss is nan"), "non_finite_loss"),
])
def test_optuna_oom_and_divergence_prune_the_trial(monkeypatch, error, reason):
    def failing_train_model(*args, **kwargs):
        raise error

    monkeypatch.setattr(optuna_search, "train_model", failing_train_model)
    objective = optuna_search.make_objective(resolve_config(smoke=True), data=None, epochs=1,
                                             device=torch.device("cpu"))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED and trial.user_attrs["pruned_reason"] == reason


# --------------------------------------------------------------------------- #
# Baselines and diagnostics
# --------------------------------------------------------------------------- #
def _psnr(a, b):
    return float(psnr(torch.from_numpy(a.transpose(2, 0, 1))[None], torch.from_numpy(b.transpose(2, 0, 1))[None]))


def test_classical_oracle_baselines_improve_psnr():
    rng = np.random.default_rng(0)
    blocks = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(blocks).resize((128, 128), Image.NEAREST), dtype=np.float32) / 255
    for ctype in CLASSES[1:]:
        for level in ("low", "high"):
            spec = level_spec(ctype, level, rng)
            corrupted = apply_spec(img, spec)
            restored = oracle_restore(corrupted, spec)
            assert restored.dtype == np.float32 and 0 <= restored.min() and restored.max() <= 1
            assert _psnr(restored, img) > _psnr(corrupted, img) + 0.5, (ctype, level)
    occluded = level_spec("occlusion", "high", rng)
    restored = oracle_restore(apply_spec(img, occluded), occluded)
    outside = apply_spec(np.ones_like(img), occluded) > 0
    assert np.array_equal(restored[outside], img[outside])  # inpainting only touches the known mask
    assert np.array_equal(oracle_restore(img, {"type": "clean"}), img)


def test_leakage_diagnostics_extremes():
    images, manifests = load_pets(smoke=True)
    identity = evaluate.leakage_diagnostics(lambda x: x, images["test"], manifests["test"], "cpu", 16, 0)
    assert identity["salt_impulse_survival"]["all"] == pytest.approx(1.0)
    assert identity["clean_detail_ratio"] == pytest.approx(1.0)
    gray = evaluate.leakage_diagnostics(lambda x: torch.full_like(x, 0.5), images["test"], manifests["test"],
                                        "cpu", 16, 0)
    assert gray["salt_impulse_survival"]["all"] == 0.0 and gray["clean_detail_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# Evaluation, ablation comparison, ONNX export
# --------------------------------------------------------------------------- #
def test_evaluate_smoke_outputs(evaluated):
    out = output_dir(smoke=True)
    records = evaluate.read_records(out / "eval" / "test_records.csv")
    assert len(records) == 40 and {r["type"] for r in records} == set(CLASSES)
    for name in ("by_type", "by_type_level", "by_level", "comparison_by_type", "comparison_by_type_level"):
        assert (out / "tables" / f"{name}.csv").exists() and (out / "tables" / f"{name}.tex").exists()
    comparison = (out / "tables" / "comparison_by_type.csv").read_text().splitlines()
    assert {"identity_psnr", "oracle_classical_psnr", "model_psnr"} <= set(comparison[0].split(","))
    assert [line.split(",")[0] for line in comparison[1:]] == [*CLASSES, "corrupted"]
    for name in ("representative_examples_1", "representative_examples_2", "failure_cases", "severity_curves",
                 "training_curves"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name

    rep = evaluated["representative_entries"]
    groups = [(records[e]["type"], records[e]["level"]) for e in rep]
    assert len(rep) == 12 and groups.count(("clean", "none")) == 3 and len(set(groups)) == 10
    failures = evaluated["failure_entries"]
    assert len(failures) >= 4 and [records[f["entry"]]["type"] for f in failures[:4]] == list(CLASSES)
    summary = json.loads((out / "eval" / "summary.json").read_text())
    assert summary["n_test_entries"] == 40 and summary["compression_ratio"] == 96  # smoke latent 8x8x8


def test_skip_variant_and_comparison(trained, evaluated):
    train.main(["--smoke", "--skip-resolutions", "128", "--epochs", "1"])
    assert (checkpoint_dir("udae_skip128", smoke=True) / "best.pt").exists()
    evaluate.evaluate_variant("udae_skip128", smoke=True, figures=False)
    assert (output_dir("udae_skip128", smoke=True) / "eval" / "test_records.csv").exists()
    rows = evaluate.main(["--smoke", "--compare", "udae", "udae_skip128"])
    assert [r["variant"] for r in rows] == ["udae", "udae_skip128"]
    assert rows[0]["skip_values_vs_input"] == 0 and rows[1]["skip_values_vs_input"] > 1
    tables = output_dir(smoke=True) / "tables"
    assert (tables / "skip_ablation.csv").exists() and (tables / "skip_ablation.tex").exists()
    assert (output_dir(smoke=True) / "figures" / "skip_ablation_examples.png").exists()


def test_export_onnx_parity_and_card(evaluated, task1_env):
    import onnxruntime as ort

    card = export_onnx.main(["--smoke"])
    onnx_path = task1_env / "onnx_dir" / "smoke" / "udae.onnx"
    assert onnx_path.exists() and onnx_path.with_suffix(".json").exists()
    parity = card["parity"]
    assert parity["passed"] and parity["n_inputs"] >= 32 and parity["single_image"]["passed"]
    assert card["preprocessing"] == "RGB, resized to 128x128, float32 in [0,1], NCHW"
    assert card["metrics"]["test"]["by_type"]["salt"]["n"] == 12
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["input"] and [o.name for o in session.get_outputs()] == ["output"]
    out = session.run(None, {"input": np.random.rand(3, 3, 128, 128).astype(np.float32)})[0]
    assert out.shape == (3, 3, 128, 128) and out.dtype == np.float32
