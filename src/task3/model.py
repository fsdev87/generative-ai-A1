"""Soft mixture-of-experts restorer (Task 3).

    w     = softmax(G(x) / tau)                                   gate = CorruptionClassifier
    x_hat = w0 * x + w1 * A_salt(x) + w2 * A_blur(x) + w3 * A_occlusion(x)

Branch order is CLASSES = (clean, salt, blur, occlusion): branch 0 is the identity, the
others are ConvAutoencoders. The gate and experts are initialised from the trained Task 2
classifier and specialists (`from_task2`), never from random weights.

forward(x) -> (output [N,3,H,W], weights [N,4], branch_outputs [N,4,3,H,W]): exactly the
ONNX contract of moe.onnx. forward_with_logits(x) also returns the gate logits G(x) for the
cross-entropy term, so the exported signature stays exactly (input) -> 3 outputs.
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.data.corruptions import CLASSES
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier

EXPERTS = CLASSES[1:]  # ("salt", "blur", "occlusion"): branches 1..3
BRANCHES = ("identity", *EXPERTS)  # display names of the four branches, CLASSES order
EXPERT_MODES = ("train", "frozen", "frozen_bn")


class SoftMoERestorer(nn.Module):
    """Gate + identity branch + three expert autoencoders, combined with softmax weights.

    gate_config / expert_configs are the constructor arguments of CorruptionClassifier and
    of each ConvAutoencoder (a dict keyed by "salt", "blur", "occlusion"), so the whole MoE
    is rebuilt from `model.config` with src.common.checkpoint.build_model.
    """

    def __init__(self, gate_config, expert_configs, tau=1.0):
        super().__init__()
        missing = set(EXPERTS) - set(expert_configs)
        if missing:
            raise ValueError(f"missing expert configs: {sorted(missing)}")
        if gate_config.get("num_classes", 4) != len(CLASSES):
            raise ValueError("the gate must output one logit per branch (4)")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.config = dict(gate_config=dict(gate_config),
                           expert_configs={k: dict(expert_configs[k]) for k in EXPERTS}, tau=float(tau))
        self.gate = CorruptionClassifier(**gate_config)
        self.experts = nn.ModuleDict({k: ConvAutoencoder(**expert_configs[k]) for k in EXPERTS})
        # A buffer (not a parameter): saved in the state dict, never trained, folded into the ONNX graph
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.expert_mode = "train"

    # ------------------------------------------------------------------ #
    # Construction from Task 2
    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoints(cls, gate_ckpt, expert_ckpts, tau=1.0):
        """Build from loaded checkpoint dicts ({model_config, model_state}) of the Task 2 models."""
        model = cls(gate_ckpt["model_config"], {k: expert_ckpts[k]["model_config"] for k in EXPERTS}, tau)
        model.gate.load_state_dict(gate_ckpt["model_state"])
        for k in EXPERTS:
            model.experts[k].load_state_dict(expert_ckpts[k]["model_state"])
        return model

    @classmethod
    def from_task2(cls, ckpt_dir, tau=1.0):
        """Initialise from CKPT_DIR/task2/classifier/best.pt and CKPT_DIR/task2/specialist_*/best.pt."""
        gate_ckpt, expert_ckpts = load_task2_checkpoints(ckpt_dir)
        return cls.from_checkpoints(gate_ckpt, expert_ckpts, tau)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def route(self, x):
        """Routing weights softmax(G(x) / tau) and the raw gate logits G(x), both float32."""
        logits = self.gate(x).float()  # float32 under autocast: the softmax and CE need the precision
        return torch.softmax(logits / self.tau, dim=1), logits

    def branch_outputs(self, x):
        """[N, 4, 3, H, W]: identity, A_salt(x), A_blur(x), A_occlusion(x)."""
        return torch.stack([x, *(self.experts[k](x).to(x.dtype) for k in EXPERTS)], dim=1)

    def forward_with_logits(self, x):
        """(output, weights, branch_outputs, gate logits): the training forward pass."""
        weights, logits = self.route(x)
        branches = self.branch_outputs(x)
        output = (weights[:, :, None, None, None] * branches).sum(dim=1)
        return output, weights, branches, logits

    def forward(self, x):
        """(output, weights, branch_outputs): the exported inference pipeline."""
        output, weights, branches, _ = self.forward_with_logits(x)
        return output, weights, branches

    # ------------------------------------------------------------------ #
    # Training stages
    # ------------------------------------------------------------------ #
    def set_expert_mode(self, mode):
        """How the experts behave while the MoE trains.

        "frozen":    warm-up. No gradients (requires_grad False) and always in eval mode, so
                     dropout is off and BatchNorm running statistics cannot change.
        "frozen_bn": joint stage (default). Weights and BN affine parameters train; BatchNorm
                     keeps the running statistics of the Task 2 specialist (eval-mode BN).
        "train":     everything trains, BatchNorm re-estimates its statistics on MoE batches.
        The gate always trains normally. The mode survives model.train()/model.eval() calls.
        """
        if mode not in EXPERT_MODES:
            raise ValueError(f"expert mode must be one of {EXPERT_MODES}, got {mode!r}")
        self.expert_mode = mode
        for p in self.experts.parameters():
            p.requires_grad_(mode != "frozen")
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.expert_mode == "frozen":
            self.experts.eval()
        elif mode and self.expert_mode == "frozen_bn":
            for m in self.experts.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def save(self, path, **extra):
        """Checkpoint with model_config + model_state (rebuild with build_model(SoftMoERestorer, path))."""
        save_checkpoint(path, model_config=self.config, model_state=self.state_dict(), **extra)


# --------------------------------------------------------------------------- #
# Task 2 checkpoints
# --------------------------------------------------------------------------- #
def task2_checkpoint_paths(ckpt_dir):
    """Fixed Task 2 locations under CKPT_DIR (docs/CONVENTIONS.md). Paths only, nothing is created."""
    root = Path(ckpt_dir) / "task2"
    return {"classifier": root / "classifier" / "best.pt",
            **{k: root / f"specialist_{k}" / "best.pt" for k in EXPERTS}}


def missing_task2_checkpoints(ckpt_dir):
    return [str(p) for p in task2_checkpoint_paths(ckpt_dir).values() if not p.exists()]


def load_task2_checkpoints(ckpt_dir):
    """(classifier checkpoint, {expert: specialist checkpoint}) as loaded dicts."""
    missing = missing_task2_checkpoints(ckpt_dir)
    if missing:
        raise FileNotFoundError("Task 3 starts from the trained Task 2 models, but these checkpoints are "
                                "missing (train Task 2 first): " + ", ".join(missing))
    paths = task2_checkpoint_paths(ckpt_dir)
    return load_checkpoint(paths["classifier"]), {k: load_checkpoint(paths[k]) for k in EXPERTS}


def write_standin_task2(ckpt_dir, seed=0, gate_config=None, expert_config=None):
    """Randomly initialised stand-ins for the Task 2 checkpoints (tests and --smoke before Task 2 exists).

    Tiny models by default so CPU tests stay fast. Never use them for real training.
    """
    torch.manual_seed(seed)
    gate_config = gate_config or dict(channels=(8, 16), convs_per_stage=1, dropout=0.1)
    expert_config = expert_config or dict(base_channels=8, depth=3, latent_channels=4)
    paths = task2_checkpoint_paths(ckpt_dir)
    gate = CorruptionClassifier(**gate_config)
    save_checkpoint(paths["classifier"], model_config=gate.config, model_state=gate.state_dict(), epoch=0,
                    metrics={}, standin=True)
    for k in EXPERTS:
        expert = ConvAutoencoder(**expert_config)
        save_checkpoint(paths[k], model_config=expert.config, model_state=expert.state_dict(), epoch=0,
                        metrics={}, standin=True)
    return paths
