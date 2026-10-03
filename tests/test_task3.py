"""Tests for Task 3, the soft mixture-of-experts restorer. Run: python -m pytest tests/test_task3.py -q

The Task 2 checkpoints are replaced by tiny randomly initialised stand-ins in temporary
directories; all scripts run in smoke mode with W&B disabled.
"""
import copy
import os

import numpy as np
import pytest
import torch

from src.common.checkpoint import build_model, load_checkpoint
from src.common.onnx_utils import check_parity, export_onnx
from src.data.corruptions import CLASSES
from src.task3 import train as train_mod
from src.task3.config import resolve_config
from src.task3.model import EXPERTS, SoftMoERestorer, load_task2_checkpoints, write_standin_task2
from src.task3.routing import balance_loss, detect_collapse, hard_route, moe_loss, routing_stats

TINY_GATE = dict(channels=(4, 8), convs_per_stage=1, dropout=0.1)
TINY_EXPERT = dict(base_channels=4, depth=2, latent_channels=2)
ENV_KEYS = ("CKPT_DIR", "OUTPUT_DIR", "ONNX_DIR", "OPTUNA_DIR", "OPTUNA_LOCAL_DIR", "WANDB_MODE", "WANDB_DIR")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Temporary output directories + stand-in Task 2 checkpoints at CKPT_DIR/task2/..."""
    root = tmp_path_factory.mktemp("task3")
    old = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS[:5]:
        os.environ[k] = str(root / k.lower())
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DIR"] = str(root)
    write_standin_task2(root / "ckpt_dir", gate_config=TINY_GATE, expert_config=TINY_EXPERT)
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def tiny_moe(tau=1.0, seed=0):
    torch.manual_seed(seed)
    return SoftMoERestorer(TINY_GATE, {k: TINY_EXPERT for k in EXPERTS}, tau=tau)


def balanced_batch(n_per_class=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(4).repeat_interleave(n_per_class)
    return {"input": torch.rand(len(labels), 3, 128, 128, generator=g),
            "target": torch.rand(len(labels), 3, 128, 128, generator=g), "label": labels}


def small_data(n_train=8, n_val=8):
    """A reduced smoke dataset (one batch per epoch) to keep the CPU tests fast."""
    data = train_mod.load_data(smoke=True)
    val = data["val"]
    keep = [i for c in range(4) for i in torch.nonzero(val["label"] == c).squeeze(1)[: n_val // 4].tolist()]
    return {"train_images": data["train_images"][:n_train],
            "val": {k: (v[keep] if torch.is_tensor(v) else [v[i] for i in keep]) for k, v in val.items()}}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_weights_are_a_distribution_and_output_is_the_weighted_sum():
    model = tiny_moe().eval()
    x = torch.rand(5, 3, 128, 128)
    with torch.no_grad():
        output, weights, branches = model(x)
    assert output.shape == (5, 3, 128, 128) and weights.shape == (5, 4) and branches.shape == (5, 4, 3, 128, 128)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.equal(branches[:, 0], x)  # branch 0 is the identity
    with torch.no_grad():
        for k, name in enumerate(EXPERTS, start=1):
            assert torch.allclose(branches[:, k], model.experts[name](x))
    expected = sum(weights[:, k, None, None, None] * branches[:, k] for k in range(4))
    assert torch.allclose(output, expected, atol=1e-6)


def test_small_tau_gives_nearly_one_hot_weights():
    x = torch.rand(6, 3, 128, 128)
    with torch.no_grad():
        sharp = tiny_moe(tau=1e-4).eval()(x)[1]
        soft = tiny_moe(tau=100.0).eval()(x)[1]
    assert torch.all(sharp.max(dim=1).values > 0.999)
    assert torch.all(soft.max(dim=1).values < 0.3)  # very large tau: almost uniform
    assert torch.equal(sharp.argmax(1), soft.argmax(1))  # tau rescales logits, never reorders them


def test_tau_is_a_saved_buffer_and_the_model_rebuilds(tmp_path):
    model = tiny_moe(tau=0.5).eval()
    assert "tau" in model.state_dict() and all(p is not model.tau for p in model.parameters())
    model.save(tmp_path / "moe.pt", epoch=1)
    rebuilt = build_model(SoftMoERestorer, tmp_path / "moe.pt").eval()
    assert rebuilt.config == model.config and float(rebuilt.tau) == 0.5
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(model(x), rebuilt(x)))


def test_from_task2_uses_the_trained_checkpoints(env):
    model = SoftMoERestorer.from_task2(env / "ckpt_dir", tau=2.0)
    gate_ckpt, expert_ckpts = load_task2_checkpoints(env / "ckpt_dir")
    assert all(torch.equal(v, gate_ckpt["model_state"][k]) for k, v in model.gate.state_dict().items())
    for name in EXPERTS:
        state = model.experts[name].state_dict()
        assert all(torch.equal(v, expert_ckpts[name]["model_state"][k]) for k, v in state.items())
    assert float(model.tau) == 2.0
    with pytest.raises(FileNotFoundError, match="train Task 2 first"):
        SoftMoERestorer.from_task2(env / "nowhere")


# --------------------------------------------------------------------------- #
# Training stages
# --------------------------------------------------------------------------- #
def _one_step(model, stage, config):
    train_mod.apply_stage(model, stage, config)
    optimizer, scheduler = train_mod.build_optimizer(model, stage, config, steps_per_epoch=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return train_mod.train_one_epoch(model, [balanced_batch()], optimizer, scheduler, scaler, config,
                                     torch.device("cpu"), use_amp=False, channels_last=False)


def _expert_bn_buffers(model):
    return {k: v.clone() for k, v in model.experts.state_dict().items() if "running" in k or "num_batches" in k}


def test_warmup_freezes_experts_parameters_and_batchnorm_statistics():
    model = tiny_moe()
    config = resolve_config(cli={"lr_warmup": 1e-2})
    experts_before = copy.deepcopy(model.experts.state_dict())
    gate_before = copy.deepcopy(model.gate.state_dict())
    stats = _one_step(model, "warmup", config)
    assert np.isfinite(stats["loss"])
    assert all(not p.requires_grad for p in model.experts.parameters())
    assert all(not m.training for m in model.experts.modules())  # dropout/BN in eval mode
    assert all(torch.equal(v, experts_before[k]) for k, v in model.experts.state_dict().items())
    changed = [k for k, v in model.gate.state_dict().items() if not torch.equal(v, gate_before[k])]
    assert any("weight" in k for k in changed) and any("running_mean" in k for k in changed)
    model.train()  # a later train() call must not unfreeze the experts' BatchNorm
    assert all(not m.training for m in model.experts.modules())


@pytest.mark.parametrize("freeze_bn", [True, False])
def test_joint_stage_updates_the_experts(freeze_bn):
    model = tiny_moe()
    config = resolve_config(cli={"lr": 1e-2, "freeze_expert_bn": freeze_bn})
    _one_step(model, "warmup", config)
    weights_before = {k: v.clone() for k, v in model.experts.named_parameters()}
    bn_before = _expert_bn_buffers(model)
    _one_step(model, "joint", config)
    assert all(p.requires_grad for p in model.experts.parameters())
    for name in EXPERTS:  # every expert receives gradient through its routing weight
        prefix = f"{name}."
        assert any(not torch.equal(v, weights_before[k]) for k, v in model.experts.named_parameters()
                   if k.startswith(prefix))
    bn_unchanged = all(torch.equal(v, bn_before[k]) for k, v in _expert_bn_buffers(model).items())
    assert bn_unchanged == freeze_bn


def test_joint_optimizer_uses_a_smaller_expert_learning_rate():
    model = tiny_moe()
    config = resolve_config()
    optimizer, _ = train_mod.build_optimizer(model, "joint", config, steps_per_epoch=10)
    gate_lr, expert_lr = (g["lr"] for g in optimizer.param_groups)
    assert gate_lr == config["lr"] and expert_lr == pytest.approx(config["lr"] * config["expert_lr_scale"])
    assert config["lr"] <= config["lr_warmup"]
    warm, _ = train_mod.build_optimizer(model, "warmup", config, steps_per_epoch=10)
    assert {id(p) for g in warm.param_groups for p in g["params"]} == {id(p) for p in model.gate.parameters()}


# --------------------------------------------------------------------------- #
# Losses and routing diagnostics
# --------------------------------------------------------------------------- #
def test_balance_loss_zero_for_perfect_routing_on_a_balanced_batch():
    labels = torch.arange(4).repeat(8)
    perfect = torch.nn.functional.one_hot(labels, 4).float()
    assert float(balance_loss(perfect)) == 0.0
    assert float(balance_loss(torch.full((32, 4), 0.25))) == 0.0
    collapsed = torch.zeros(32, 4)
    collapsed[:, 1] = 1
    assert float(balance_loss(collapsed)) == pytest.approx(0.75)
    unbalanced = torch.nn.functional.one_hot(torch.tensor([0, 0, 0, 1]), 4).float()
    assert float(balance_loss(unbalanced)) > 0  # why the batches must be balanced


def test_moe_loss_terms_and_ce_placement():
    torch.manual_seed(0)
    out, target = torch.rand(4, 3, 32, 32), torch.rand(4, 3, 32, 32)
    logits, labels = torch.randn(4, 4) * 3, torch.arange(4)
    weights = torch.softmax(logits / 2.0, dim=1)
    config = resolve_config(cli={"tau": 2.0})
    loss, parts = moe_loss(out, target, weights, logits, labels, config)
    expected = 0.8 * parts["l1"] + 0.2 * (1 - parts["ssim"]) + 0.1 * parts["ce"] + 0.01 * parts["balance"]
    assert torch.allclose(loss, expected)
    assert torch.allclose(parts["ce"], torch.nn.functional.cross_entropy(logits, labels))
    _, routed = moe_loss(out, target, weights, logits, labels, {**config, "ce_input": "routing"})
    assert torch.allclose(routed["ce"], torch.nn.functional.cross_entropy(logits / 2.0, labels))


def _weights_from_matrix(matrix, n_per_class=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), n_per_class)
    w = np.asarray(matrix)[labels] + noise * rng.random((len(labels), 4))
    return w / w.sum(axis=1, keepdims=True), labels


def test_collapse_criterion():
    healthy, labels = _weights_from_matrix(0.85 * np.eye(4) + 0.05, noise=0.05)
    stats = routing_stats(healthy, labels)
    assert not detect_collapse(stats)[0] and stats["accuracy"] == 1.0
    assert not detect_collapse(routing_stats(*_weights_from_matrix(np.full((4, 4), 0.25))))[0]  # uncertain != collapsed

    total = np.zeros((4, 4))
    total[:, 2] = 1  # everything to the blur expert
    collapsed, reason = detect_collapse(routing_stats(*_weights_from_matrix(total)))
    assert collapsed and "dead branch clean" in reason and "blur dominates" in reason

    dead = 0.9 * np.eye(4) + 0.025
    dead[1] = [0.05, 0.02, 0.03, 0.9]  # salt inputs go to the occlusion expert, salt expert unused
    dead[:, 1] = [0.02, 0.02, 0.02, 0.02]
    collapsed, reason = detect_collapse(routing_stats(*_weights_from_matrix(dead)))
    assert collapsed and "dead branch salt" in reason

    dominant = 0.4 * np.eye(4)
    dominant[:, 0] += 0.6  # the identity takes 60 % of every corrupted input
    collapsed, reason = detect_collapse(routing_stats(*_weights_from_matrix(dominant)))
    assert collapsed and "clean dominates" in reason and "dead" not in reason


def test_hard_route_identity_bypass_and_single_expert():
    model = tiny_moe().eval()
    x = torch.rand(4, 3, 128, 128)
    out = hard_route(x, torch.tensor([0, 1, 2, 3]), model.experts)
    assert torch.equal(out[0], x[0])
    with torch.no_grad():
        for k, name in enumerate(EXPERTS, start=1):
            assert torch.allclose(out[k], model.experts[name](x[k:k + 1])[0], atol=1e-6)


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def test_onnx_parity_of_the_full_pipeline(tmp_path):
    import onnxruntime as ort

    model = tiny_moe(tau=0.7).eval()
    names = ("output", "weights", "branch_outputs")
    path = export_onnx(model, (torch.rand(2, 3, 128, 128),), tmp_path / "moe.onnx", ("input",), names)
    parity = check_parity(model, path, (torch.rand(32, 3, 128, 128),), ("input",), names)
    assert parity["passed"] and parity["n_inputs"] == 32, parity
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == list(names)
    outs = session.run(None, {"input": np.random.rand(3, 3, 128, 128).astype(np.float32)})
    assert [o.shape for o in outs] == [(3, 3, 128, 128), (3, 4), (3, 4, 3, 128, 128)]


# --------------------------------------------------------------------------- #
# Training loop: resume and scripts
# --------------------------------------------------------------------------- #
def test_resume_after_interruption_matches_an_uninterrupted_run(env, tmp_path, monkeypatch):
    data = small_data()
    init = load_task2_checkpoints(env / "ckpt_dir") + ({"standin": True},)
    config = resolve_config(smoke=True, cli={"joint_epochs": 2})  # epochs: 1 warm-up, 2-3 joint

    full = train_mod.train_moe(config, data=data, init=init, ckpt_dir=tmp_path / "full", verbose=False)
    assert [h["stage"] for h in full["history"]] == ["warmup", "joint", "joint"]

    real_validate, calls = train_mod.validate, {"n": 0}

    def interrupted(*args, **kwargs):  # simulates a Colab disconnect during epoch 3
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(train_mod, "validate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train_mod.train_moe(config, data=data, init=init, ckpt_dir=tmp_path / "cut", verbose=False)
    monkeypatch.setattr(train_mod, "validate", real_validate)
    state = load_checkpoint(tmp_path / "cut" / "last.pt")
    assert state["epoch"] == 2 and state["stage"] == "joint" and not state["finished"]
    resumed = train_mod.train_moe(config, data=data, resume_state=state, ckpt_dir=tmp_path / "cut", verbose=False)
    assert resumed["epochs_run"] == 3 and resumed["stop_reason"] == "max_epochs"
    a = load_checkpoint(tmp_path / "full" / "last.pt")["model_state"]
    b = load_checkpoint(tmp_path / "cut" / "last.pt")["model_state"]
    assert all(torch.allclose(a[k].float(), b[k].float(), atol=1e-6) for k in a)


def test_optuna_prunes_on_routing_collapse(env, monkeypatch):
    import optuna

    monkeypatch.setattr(train_mod, "detect_collapse", lambda stats, *a: (True, "dead branch salt (test)"))
    study = optuna.create_study(direction="maximize")
    data = small_data()
    init = load_task2_checkpoints(env / "ckpt_dir") + ({"standin": True},)

    def objective(trial):
        return train_mod.train_moe(resolve_config(smoke=True), trial=trial, data=data, init=init, verbose=False)["best_score"]

    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["prune_reason"].startswith("collapse at epoch 1")


@pytest.fixture()
def fast_smoke_data(monkeypatch):
    """Scripts in smoke mode, on the reduced smoke data (same code path, fewer images)."""
    from src.task3 import optuna_search

    data = small_data()
    monkeypatch.setattr(train_mod, "load_data", lambda smoke=False: data)
    monkeypatch.setattr(optuna_search, "load_data", lambda smoke=False: data)


def test_smoke_runs_of_every_script(env, fast_smoke_data):
    from src.task3 import compare, evaluate, evaluate_mixed, export_onnx as export_mod, optuna_search
    from src.task3.config import best_config_path, checkpoint_dir, onnx_path, output_dir

    result = train_mod.main(["--smoke"])
    assert (checkpoint_dir(True) / "best.pt").exists() and result["epochs_run"] == 2
    assert train_mod.main(["--smoke", "--resume"])["already_finished"]

    study = optuna_search.main(["--smoke"])
    assert len(study.trials) == 2 and study.trials[0].params["tau"] == 1.0  # the brief's values first
    assert best_config_path(True).exists()
    assert (output_dir(True, "optuna", "task3_moe_smoke") / "trials.csv").exists()

    summary = evaluate.main(["--smoke"])
    records = (output_dir(True, "eval") / "test_records.csv").read_text().splitlines()
    assert len(records) == 1 + 40 and "w_occlusion" in records[0] and "entropy" in records[0]
    assert set(summary["routing"]["usage"]) == {"identity", "salt", "blur", "occlusion"}
    for name in ("routing_heatmap_type_level", "weight_distributions", "gating_dominant", "gating_spread", "failures"):
        assert (output_dir(True, "figures") / f"{name}.png").exists()

    assert compare.main(["--smoke"]) is not None  # Task 1/2 CSVs absent: skipped gracefully
    mixed = evaluate_mixed.main(["--smoke"])
    assert {r["combo"] for r in mixed["routing"]} == {"salt+blur", "blur+occlusion", "salt+occlusion"}

    card = export_mod.main(["--smoke"])
    assert card["parity"]["passed"] and card["parity"]["n_inputs"] >= 32 and onnx_path(True).exists()
    assert card["tau"] == 1.0 and card["routing"]["usage"]


def test_compare_pairs_entries_across_tasks(env, tmp_path, monkeypatch):
    import pandas as pd

    from src.task3 import compare

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    rows = [{"entry": i, "image_idx": i // 10, "type": CLASSES[min(i % 10, 3)], "level": "low" if i % 10 else "none",
             "psnr": 25.0 + i % 3, "ssim": 0.8, "mse": 0.01} for i in range(20)]
    moe = pd.DataFrame(rows)
    hard = moe.assign(psnr=np.where(moe["type"] == "clean", 100.0, moe["psnr"] - 1.0), classifier_correct=1)
    for task, name, df in (("task2", "test_records_predicted.csv", hard), ("task3", "test_records.csv", moe)):
        (tmp_path / task / "eval").mkdir(parents=True)
        df.to_csv(tmp_path / task / "eval" / name, index=False)
    result = compare.main([])
    clean = next(r for r in result["by_type"] if r["type"] == "clean")
    assert np.isnan(clean["predicted_psnr"]) and clean["predicted_n_exact"] == 2  # exact bypass not averaged
    gain = next(r for r in result["paired"] if r["versus"] == "predicted" and r["subset"] == "all corrupted")
    assert gain["mean_psnr_gain"] == pytest.approx(1.0) and gain["moe_wins_psnr"] == 1.0
