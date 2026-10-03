"""Tests for Task 2 (src/task2). Run: python -m pytest tests/test_task2.py -q

The module fixture runs every Task 2 script with --smoke, in pipeline order, into
temporary CKPT/OUTPUT/ONNX/OPTUNA directories with W&B disabled.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.checkpoint import build_model, load_checkpoint, save_checkpoint
from src.common.metrics import PSNR_CAP_DB
from src.data.corruptions import CLASSES
from src.data.pets import synthetic_images
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier
from src.task2 import (
    evaluate_classifier, evaluate_routing, export_onnx, optuna_classifier, optuna_specialists, train_classifier,
    train_specialist,
)
from src.task2.common import build_config
from src.task2.routing import HardRoutedRestorer

TYPES = ("salt", "blur", "occlusion")


def _set_env(mp, root):
    for env in ("CKPT_DIR", "OUTPUT_DIR", "ONNX_DIR", "OPTUNA_DIR", "OPTUNA_LOCAL_DIR", "WANDB_DIR"):
        (root / env.lower()).mkdir(parents=True, exist_ok=True)
        mp.setenv(env, str(root / env.lower()))
    mp.setenv("WANDB_MODE", "disabled")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("task2")
    with pytest.MonkeyPatch.context() as mp:
        _set_env(mp, root)
        out = root / "output_dir" / "task2"
        optuna_classifier.main(["--smoke", "--n-trials", "1", "--epochs", "1"])
        train_classifier.main(["--smoke", "--config", str(out / "task2_classifier_smoke_best_config.yaml")])
        evaluate_classifier.main(["--smoke", "--n-examples", "4"])
        optuna_specialists.main(["--smoke", "--n-trials", "1", "--epochs", "1"])
        for ctype in TYPES:
            train_specialist.main(["--smoke", "--type", ctype, "--epochs", "1",
                                   "--config", str(out / "task2_specialists_smoke_best_config.yaml")])
        evaluate_routing.main(["--smoke", "--n-failures", "3"])
        export_onnx.main(["--smoke", "--n-parity", "32"])
        yield root


# --------------------------------------------------------------------------- #
# End-to-end smoke pipeline
# --------------------------------------------------------------------------- #
def test_checkpoints_follow_the_task3_contract(pipeline):
    ckpt = pipeline / "ckpt_dir" / "task2"
    for name, cls in [("classifier", CorruptionClassifier)] + [(f"specialist_{t}", ConvAutoencoder) for t in TYPES]:
        for file in ("best.pt", "last.pt"):
            state = load_checkpoint(ckpt / name / file)
            assert {"model_config", "model_state"} <= set(state), (name, file)
        model = build_model(cls, ckpt / name / "best.pt")  # exactly how Task 3 loads them
        assert model.config == load_checkpoint(ckpt / name / "best.pt")["model_config"]
    assert load_checkpoint(ckpt / "classifier" / "last.pt")["finished"]


def test_optuna_reports_and_best_configs(pipeline):
    out = pipeline / "output_dir" / "task2"
    for study in ("task2_classifier_smoke", "task2_specialists_smoke"):
        report = out / "optuna" / study
        assert (report / "trials.csv").exists() and (report / "search_space.json").exists()
        assert (out / f"{study}_best_config.yaml").exists()
        assert (pipeline / "optuna_dir" / f"{study}.db").exists()
    trials = pd.read_csv(out / "optuna" / "task2_specialists_smoke" / "trials.csv")
    for t in TYPES:  # per-type scores stored as user attributes
        assert f"user_attrs_score_{t}" in trials.columns
    complete = trials[trials["state"] == "COMPLETE"]
    per_type = complete[[f"user_attrs_score_{t}" for t in TYPES]].mean(axis=1)
    assert np.allclose(per_type, complete["value"])  # objective = mean of the three scores


def test_classifier_evaluation_outputs(pipeline):
    out = pipeline / "output_dir" / "task2"
    metrics = json.loads((out / "tables" / "classifier_metrics.json").read_text())
    for split in ("test", "val"):
        cm = np.array(metrics[split]["confusion_normalized"])
        assert cm.shape == (4, 4) and np.allclose(cm.sum(axis=1), 1)
        assert set(metrics[split]["per_class"]) == set(CLASSES)
    preds = pd.read_csv(out / "tables" / "classifier_test_predictions.csv")
    assert len(preds) == 40 and np.allclose(preds[[f"prob_{c}" for c in CLASSES]].sum(axis=1), 1, atol=1e-5)
    assert preds.loc[preds["type"] == "blur", "blur_strength"].notna().all()
    blur = pd.read_csv(out / "tables" / "classifier_blur_strength.csv")
    assert {"clean", "low", "medium", "high"} <= set(blur[blur["source"] == "test"]["group"])
    for name in ("classifier_confusion_test.png", "classifier_blur_detection.png", "classifier_sharpness.png"):
        assert (out / "figures" / name).stat().st_size > 0


def test_routing_records_and_analysis(pipeline):
    out = pipeline / "output_dir" / "task2"
    records = {m: pd.read_csv(out / "eval" / f"test_records_{m}.csv") for m in ("oracle", "predicted")}
    for mode, df in records.items():
        assert len(df) == 40
        assert {"entry", "type", "level", "psnr", "ssim", "mse", "true_label", "pred_label", "expert"} <= set(df.columns)
        bypassed_clean = df[(df["route"] == 0) & (df["type"] == "clean")]  # identity output == clean target
        assert (bypassed_clean["psnr"] == PSNR_CAP_DB).all() and np.allclose(bypassed_clean["ssim"], 1.0)
        if mode == "oracle":
            assert ((df["route"] == 0) == (df["type"] == "clean")).all()
    oracle = records["oracle"]
    assert (oracle["route"] == oracle["true_label"]).all()
    assert (oracle["true_label"] == oracle["type"].map(CLASSES.index)).all()
    predicted = records["predicted"]
    assert (predicted["route"] == predicted["pred_label"]).all()
    # entries the classifier routes correctly get the same expert, hence the same result, in both modes
    same = predicted["pred_label"] == predicted["true_label"]
    assert np.allclose(predicted.loc[same, "psnr"], oracle.loc[same, "psnr"], atol=1e-3)
    comparison = pd.read_csv(out / "tables" / "routing_oracle_vs_predicted.csv")
    assert {"clean", "salt", "blur", "occlusion", "corrupted"} <= set(comparison["type"])
    misrouting = pd.read_csv(out / "tables" / "routing_misrouting.csv")
    assert misrouting["n"].sum() == (~same).sum()
    summary = json.loads((out / "eval" / "summary.json").read_text())
    assert summary["oracle"]["clean"]["n_exact"] == summary["oracle"]["clean"]["n"]


def test_onnx_parity_and_model_cards(pipeline):
    import onnxruntime as ort

    onnx_dir = pipeline / "onnx_dir"
    x = torch.rand(3, 3, 128, 128)
    for name, cls, output in [("classifier", CorruptionClassifier, "probs")] + \
                             [(f"specialist_{t}", ConvAutoencoder, "output") for t in TYPES]:
        card = json.loads((onnx_dir / f"{name}.json").read_text())
        assert card["parity"]["passed"] and "metrics" in card
        session = ort.InferenceSession(str(onnx_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        assert [o.name for o in session.get_outputs()] == [output]
        got = session.run(None, {"input": x.numpy()})[0]
        model = build_model(cls, pipeline / "ckpt_dir" / "task2" / name / "best.pt").eval()
        with torch.no_grad():
            ref = model(x)
        ref = torch.softmax(ref, dim=1) if output == "probs" else ref
        assert np.abs(got - ref.numpy()).max() < 1e-4
    assert json.loads((onnx_dir / "classifier.json").read_text())["metrics"]["test_macro_f1"] >= 0
    assert "test_psnr" in json.loads((onnx_dir / "specialist_blur.json").read_text())["metrics"]


# --------------------------------------------------------------------------- #
# Training details
# --------------------------------------------------------------------------- #
def test_classifier_batches_are_balanced():
    cfg = {**train_classifier.DEFAULTS, "batch_size": 12}
    loader, sampler = train_classifier.make_train_loader(synthetic_images(40), cfg, torch.device("cpu"), 0)
    assert len(sampler) == 3  # 40 // 12, incomplete batch dropped
    for epoch in range(2):
        sampler.rng = np.random.default_rng([cfg["seed"], epoch])
        for batch in loader:
            assert torch.equal(torch.bincount(batch["label"], minlength=4), torch.full((4,), 3))
    with pytest.raises(ValueError):
        train_classifier.make_train_loader(synthetic_images(8), {**cfg, "batch_size": 6}, torch.device("cpu"), 0)


def test_classifier_resumes_after_interruption(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    cfg = {**train_classifier.DEFAULTS, **train_classifier.SMOKE, "epochs": 2}
    train_classifier.train_classifier(cfg, smoke=True)
    path = tmp_path / "ckpt_dir" / "task2" / "classifier" / "last.pt"
    state = load_checkpoint(path)
    run_id = state["wandb_run_id"]
    # pretend the session died after the first epoch
    save_checkpoint(path, **{**state, "epoch": 0, "finished": False})
    train_classifier.train_classifier({**cfg, "epochs": 5}, smoke=True, resume=True)  # checkpoint config wins
    state = load_checkpoint(path)
    assert state["epoch"] == 1 and state["finished"] and state["wandb_run_id"] == run_id
    assert state["train_config"]["epochs"] == 2


def test_smoke_run_never_overwrites_real_checkpoints(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    ckpt = tmp_path / "ckpt_dir" / "task2" / "specialist_salt"
    model = ConvAutoencoder(base_channels=8, latent_channels=4)
    save_checkpoint(ckpt / "best.pt", model_config=model.config, model_state=model.state_dict(), smoke=False)
    with pytest.raises(RuntimeError, match="real training run"):
        train_specialist.main(["--smoke", "--type", "salt"])


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 0.01\nlearning_rate: 0.1\n")
    with pytest.raises(KeyError):
        build_config(train_classifier.DEFAULTS, path)
    assert build_config(train_classifier.DEFAULTS, overrides={"lr": 0.5, "epochs": None})["epochs"] == 60


def test_search_space_presets_build_models():
    for params in optuna_classifier.CHANNEL_PRESETS:
        cfg = optuna_classifier.apply_params(train_classifier.DEFAULTS, {"channels": params})
        assert train_classifier.build_classifier(cfg)(torch.rand(1, 3, 128, 128)).shape == (1, 4)
    for name, preset in optuna_specialists.LATENT_PRESETS.items():
        model = train_specialist.build_specialist({**train_specialist.DEFAULTS, "base_channels": 8, **preset})
        side, _, channels = (int(v) for v in name.split("x"))
        assert model.latent_dim == side * side * channels


# --------------------------------------------------------------------------- #
# Hard routing
# --------------------------------------------------------------------------- #
class FixedClassifier(nn.Module):
    """Predicts the given classes regardless of the input."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = torch.as_tensor(predictions)

    def forward(self, x):
        return 10.0 * F.one_hot(self.predictions, len(CLASSES)).float()


class ConstantExpert(nn.Module):
    """Returns a constant image and records the size of every batch it receives."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, x):
        self.calls.append(x.shape[0])
        return torch.full_like(x, self.value)


def _restorer(predictions, mode="predicted"):
    experts = {t: ConstantExpert(v) for t, v in zip(TYPES, (0.25, 0.5, 0.75))}
    return HardRoutedRestorer(FixedClassifier(predictions), experts, mode).eval(), experts


def test_identity_bypass_returns_the_exact_input():
    x = torch.rand(4, 3, 128, 128)
    restorer, experts = _restorer([0, 0, 0, 0])
    with torch.no_grad():
        out, info = restorer(x)
    assert torch.equal(out, x) and out.data_ptr() != x.data_ptr()
    assert all(not e.calls for e in experts.values())  # no expert executed
    assert torch.equal(info["route"], torch.zeros(4, dtype=torch.long))


def test_mixed_batch_routes_each_sample_to_its_expert():
    x = torch.rand(7, 3, 128, 128)
    predictions = [0, 1, 2, 3, 1, 0, 3]
    restorer, experts = _restorer(predictions)
    with torch.no_grad():
        out, info = restorer(x)
    for i, k in enumerate(predictions):
        if k == 0:
            assert torch.equal(out[i], x[i])
        else:
            assert torch.all(out[i] == experts[TYPES[k - 1]].value)
    # batch-friendly: each expert runs once on its whole sub-batch
    assert experts["salt"].calls == [2] and experts["blur"].calls == [1] and experts["occlusion"].calls == [2]
    assert torch.allclose(info["probs"].sum(1), torch.ones(7))


def test_oracle_mode_routes_by_the_true_label():
    x = torch.rand(4, 3, 128, 128)
    restorer, experts = _restorer([0, 0, 0, 0], mode="oracle")  # classifier always says "clean"
    labels = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        out, info = restorer(x, labels=labels)
    assert torch.equal(info["route"], labels) and torch.equal(info["predicted"], torch.zeros(4, dtype=torch.long))
    assert torch.all(out[0] == 0.5) and torch.equal(out[1], x[1])
    assert torch.all(out[2] == 0.75) and torch.all(out[3] == 0.25)
    with pytest.raises(ValueError):
        restorer(x)  # oracle without labels


def test_real_run_never_resumes_or_evaluates_a_smoke_checkpoint(tmp_path):
    """Regression: a --smoke quick check left a finished tiny model in the real checkpoint
    folder; the real `--resume` run then skipped training and evaluation used the tiny model."""
    import pytest

    from src.common.checkpoint import save_checkpoint
    from src.task2.common import load_resume_state, require_real_checkpoint

    smoke_ckpt, real_ckpt = tmp_path / "smoke.pt", tmp_path / "real.pt"
    save_checkpoint(smoke_ckpt, smoke=True, finished=True, epoch=1)
    save_checkpoint(real_ckpt, smoke=False, finished=False, epoch=7)

    assert load_resume_state(smoke_ckpt, smoke=False, tag="t") is None  # real run starts fresh
    assert load_resume_state(smoke_ckpt, smoke=True, tag="t")["epoch"] == 1  # smoke may resume smoke
    assert load_resume_state(real_ckpt, smoke=False, tag="t")["epoch"] == 7

    with pytest.raises(RuntimeError, match="--smoke checkpoint"):
        require_real_checkpoint(smoke_ckpt, smoke=False)
    require_real_checkpoint(smoke_ckpt, smoke=True)
    require_real_checkpoint(real_ckpt, smoke=False)
