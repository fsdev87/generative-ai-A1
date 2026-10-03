"""Hard-routed restoration (Task 2): classifier -> argmax -> one specialist, identity for clean.

    x_hat = x                 if r = clean   (identity bypass: no expert runs, output == input exactly)
            A_salt(x)         if r = salt
            A_blur(x)         if r = blur
            A_occlusion(x)    if r = occlusion

r is the classifier's argmax ("predicted" routing, the operational system) or the true
label from the deterministic manifest ("oracle" routing, the specialists' own ability).
The batch is split once per expert (each expert runs at most once per batch, on its
sub-batch), never per sample. Models must be in eval mode so BatchNorm uses its running
statistics and a sample's output does not depend on the rest of the batch.
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.common.checkpoint import build_model
from src.common.paths import get_dir
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier

from .common import SPECIALIST_TYPES, TASK, require_real_checkpoint

ROUTING_MODES = ("predicted", "oracle")
EXPERT_NAMES = ("identity", *SPECIALIST_TYPES)  # route index -> expert, same order as CLASSES


class HardRoutedRestorer(nn.Module):
    def __init__(self, classifier, specialists, routing_mode="predicted"):
        super().__init__()
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}, got {routing_mode!r}")
        missing = set(SPECIALIST_TYPES) - set(specialists)
        if missing:
            raise ValueError(f"missing specialists: {sorted(missing)}")
        self.classifier = classifier
        self.specialists = nn.ModuleDict({t: specialists[t] for t in SPECIALIST_TYPES})
        self.routing_mode = routing_mode

    def forward(self, x, labels=None):
        """Returns (output, info) with info = probs (N, 4), predicted (N,), route (N,).

        The classifier runs in both modes, so oracle evaluations still record its prediction.
        """
        probs = torch.softmax(self.classifier(x).float(), dim=1)
        predicted = probs.argmax(dim=1)
        if self.routing_mode == "oracle":
            if labels is None:
                raise ValueError("oracle routing needs the true labels")
            route = torch.as_tensor(labels, device=x.device).long()
        else:
            route = predicted
        output = x.clone()  # route 0 (clean) keeps the input bit for bit
        for k, ctype in enumerate(SPECIALIST_TYPES, start=1):
            idx = torch.nonzero(route == k).squeeze(1)
            if idx.numel():
                output[idx] = self.specialists[ctype](x[idx]).to(output.dtype)
        return output, {"probs": probs, "predicted": predicted, "route": route}


def checkpoint_paths():
    """Fixed checkpoint locations shared with Task 3 (docs/CONVENTIONS.md)."""
    paths = {"classifier": get_dir("CKPT_DIR", TASK, "classifier") / "best.pt"}
    paths.update({t: get_dir("CKPT_DIR", TASK, f"specialist_{t}") / "best.pt" for t in SPECIALIST_TYPES})
    return paths


def load_models(device="cpu", smoke=False):
    """Classifier and the three specialists from their best.pt, in eval mode.

    Outside smoke mode every checkpoint must come from a real training run (not --smoke).
    """
    paths = checkpoint_paths()
    missing = [str(p) for p in paths.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing Task 2 checkpoints (train the classifier and the specialists first): "
                                + ", ".join(missing))
    for path in paths.values():
        require_real_checkpoint(path, smoke)
    classifier = build_model(CorruptionClassifier, paths["classifier"]).to(device).eval()
    specialists = {t: build_model(ConvAutoencoder, paths[t]).to(device).eval() for t in SPECIALIST_TYPES}
    return classifier, specialists


def load_restorer(routing_mode="predicted", device="cpu"):
    classifier, specialists = load_models(device)
    return HardRoutedRestorer(classifier, specialists, routing_mode).to(device).eval()

