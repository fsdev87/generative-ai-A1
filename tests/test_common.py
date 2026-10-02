"""Tests for the shared foundation (src/common). Run: python -m pytest tests/test_common.py -q"""
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import build_model, load_checkpoint, save_checkpoint
from src.common.evaluation import (
    evaluate_restoration, predict_entries, representative_entries, save_csv, save_latex_table,
    save_restoration_figure, standard_tables, worst_entries,
)
from src.common.losses import RestorationLoss, ssim
from src.common.metrics import PSNR_CAP_DB, psnr, restoration_score
from src.common.onnx_utils import check_parity, export_onnx, write_model_card
from src.common.optuna_utils import create_study, run_study, save_study_report
from src.common.tracking import image_grid
from src.common.utils import EarlyStopping, RunningMean
from src.data.corruptions import CLASSES
from src.data.pets import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset, load_pets
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier

torch.manual_seed(0)
X = torch.rand(4, 3, 64, 64)
Y = (X + 0.1 * torch.randn_like(X)).clamp(0, 1)


def test_ssim_matches_torchmetrics():
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim

    ours = ssim(X, Y)
    # torchmetrics reflect-pads and averages the SSIM map over the full image; we use the
    # valid region (Wang et al.'s reference code, scikit-image), i.e. its map minus a 5-pixel border
    _, full_map = tm_ssim(X, Y, data_range=1.0, return_full_image=True)
    assert abs(float(ours) - float(full_map[..., 5:-5, 5:-5].mean())) < 1e-6
    per_image = ssim(X, Y, reduction="none")
    assert per_image.shape == (4,)
    assert torch.allclose(per_image.mean(), ours)


def test_ssim_identical_and_autocast():
    assert abs(float(ssim(X, X)) - 1.0) < 1e-6
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = ssim(X, Y)
    assert inside.dtype == torch.float32
    assert abs(float(inside) - float(ssim(X, Y))) < 1e-6


def test_psnr_known_value_and_cap():
    target = torch.zeros(2, 3, 8, 8)
    noisy = target + 0.1  # MSE 0.01 -> 20 dB
    assert torch.allclose(psnr(noisy, target), torch.full((2,), 20.0), atol=1e-4)
    assert torch.all(psnr(target, target) == PSNR_CAP_DB)


def test_restoration_loss_and_score():
    loss, parts = RestorationLoss(alpha=1.0)(Y, X)
    assert torch.allclose(loss, (Y - X).abs().mean())
    loss, parts = RestorationLoss(alpha=0.0)(Y, X)
    assert torch.allclose(loss, 1 - parts["ssim"])
    assert restoration_score(40.0, 1.0) == 1.0


def test_running_mean_and_early_stopping():
    meter = RunningMean()
    meter.update({"loss": 1.0}, n=1)
    meter.update({"loss": 4.0}, n=3)
    assert meter.means()["loss"] == pytest.approx(3.25)
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.5) and not stopper.step(0.4) and not stopper.should_stop
    assert not stopper.step(0.3) and stopper.should_stop
    assert stopper.best == 0.5


def test_checkpoint_roundtrip(tmp_path):
    model = ConvAutoencoder(base_channels=8, latent_channels=4).eval()
    save_checkpoint(tmp_path / "best.pt", model_config=model.config, model_state=model.state_dict(), epoch=3)
    assert not (tmp_path / "best.pt.tmp").exists()
    rebuilt = build_model(ConvAutoencoder, tmp_path / "best.pt").eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 3


def test_optuna_study_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTUNA_DIR", str(tmp_path / "remote"))
    monkeypatch.setenv("OPTUNA_LOCAL_DIR", str(tmp_path / "local"))

    def objective(trial):
        return trial.suggest_float("x", -1, 1) ** 2

    study, sync = create_study("unit", direction="minimize", fresh=True)
    run_study(study, objective, n_trials=3, callbacks=[sync])
    assert (tmp_path / "remote" / "unit.db").exists()

    # Simulate a new Colab session: fresh empty local disk, only the Drive copy survives
    monkeypatch.setenv("OPTUNA_LOCAL_DIR", str(tmp_path / "local_new_session"))
    study, sync = create_study("unit", direction="minimize")
    assert len(study.trials) == 3
    run_study(study, objective, n_trials=5, callbacks=[sync])
    assert len(study.trials) == 5

    save_study_report(study, tmp_path / "report")
    assert (tmp_path / "report" / "trials.csv").exists()
    assert (tmp_path / "report" / "summary.json").exists()
    assert (tmp_path / "report" / "optimization_history.png").exists()


@pytest.mark.parametrize("model, names", [
    (ConvAutoencoder(base_channels=8, latent_channels=4, skip_resolutions=(16,)), ("output",)),
    (torch.nn.Sequential(CorruptionClassifier(channels=(8, 16)), torch.nn.Softmax(dim=1)), ("probs",)),
])
def test_onnx_export_parity_dynamic_batch(tmp_path, model, names):
    import onnxruntime as ort

    model = model.eval()
    path = export_onnx(model, (torch.rand(2, 3, 128, 128),), tmp_path / "m.onnx", ("input",), names)
    parity = check_parity(model, path, (torch.rand(5, 3, 128, 128),), ("input",), names)
    assert parity["passed"], parity
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.run(None, {"input": np.random.rand(3, 3, 128, 128).astype(np.float32)})[0].shape[0] == 3
    card = write_model_card(path, task="unit", parity=parity, model_config={"skips": (16,)})
    assert card["sha256"] and (tmp_path / "m.json").exists()


def test_load_pets_smoke_and_loaders():
    images, manifests = load_pets(smoke=True)
    assert images["train"].shape == (32, 128, 128, 3) and images["train"].dtype == np.uint8
    assert len(manifests["test"]["entries"]) == 4 * 10
    loader = DataLoader(RuntimeCorruptionDataset(images["train"]),
                        batch_sampler=BalancedBatchSampler(32, 8, seed=0))
    for batch in loader:
        assert torch.all(torch.bincount(batch["label"], minlength=4) == 2)
    blur_only = ManifestDataset(images["test"], manifests["test"], types=["blur"])
    assert len(blur_only) == 4 * 3 and all(e["type"] == "blur" for e in blur_only.entries)


def test_evaluation_pipeline(tmp_path):
    images, manifests = load_pets(smoke=True)
    dataset = ManifestDataset(images["test"], manifests["test"])
    identity = lambda x: (x, {"weights": torch.ones(x.shape[0], 4) / 4})  # noqa: E731
    records = evaluate_restoration(identity, dataset, device="cpu", batch_size=16)
    assert len(records) == 40 and records[0]["weights"] == [0.25] * 4

    clean = [r for r in records if r["type"] == "clean"]
    assert all(r["psnr"] == PSNR_CAP_DB and r["ssim"] == pytest.approx(1.0) for r in clean)

    tables = standard_tables(records)
    assert [r["type"] for r in tables["by_type"]] == [*CLASSES, "all"]
    assert len(tables["by_type_level"]) == 1 + 3 * 3
    assert [r["level"] for r in tables["by_level"]] == ["low", "medium", "high"]
    # Stronger corruption leaves the identity output further from the target
    salt = {r["level"]: r["psnr"] for r in tables["by_type_level"] if r["type"] == "salt"}
    assert salt["low"] > salt["medium"] > salt["high"]

    save_csv(records, tmp_path / "records.csv")
    save_latex_table(tables["by_type"], [("type", "Input"), ("psnr", "PSNR"), ("ssim", "SSIM")], tmp_path / "t.tex")
    assert r"\toprule" in (tmp_path / "t.tex").read_text()

    chosen = representative_entries(records, per_group=1)
    assert len(chosen) == 10
    worst = worst_entries(records, n=4, types=["occlusion"])
    assert all(records[i]["type"] == "occlusion" for i in worst)
    samples = predict_entries(identity, dataset, chosen[:3], device="cpu")
    save_restoration_figure(samples, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_image_grid_shape():
    rows = [[np.zeros((8, 8, 3)), np.ones((8, 8))], [np.ones((8, 8, 3))]]
    grid = image_grid(rows, pad=2)
    assert grid.shape == (2 * 10 + 2, 2 * 10 + 2, 3) and grid.dtype == np.uint8


def test_wandb_disabled_run_logs(monkeypatch, tmp_path):
    from src.common.tracking import log_artifact, log_image_grid, wandb_run

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    (tmp_path / "best.pt").write_bytes(b"x")
    with wandb_run("unit", "unit-tests", config={"a": 1}) as run:
        run.log({"loss": 1.0})
        log_image_grid(run, "samples", [[np.zeros((8, 8, 3))]])
        log_artifact(run, tmp_path / "best.pt", "unit-model")
    assert os.environ["WANDB_MODE"] == "disabled"
