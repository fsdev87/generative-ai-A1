"""Inference for the four workspaces on preprocessed (H, W, 3) float32 images.

Restoration inputs and outputs are in [0, 1]. Every function returns plain arrays plus the
ONNX Runtime time in milliseconds; the routers turn them into responses.
"""
from dataclasses import dataclass

import numpy as np

from .corruption import CLASSES
from .models import ModelRegistry
from .preprocessing import from_batch, to_batch


@dataclass(frozen=True)
class Restoration:
    output: np.ndarray
    inference_ms: float


@dataclass(frozen=True)
class HardRouting:
    output: np.ndarray
    probabilities: np.ndarray  # (4,), CLASSES order
    predicted_class: str
    routed_class: str  # the class that chose the expert
    selected_expert: str  # "identity" or the specialist's model name
    classifier_ms: float
    expert_ms: float


@dataclass(frozen=True)
class SoftMixture:
    output: np.ndarray
    weights: np.ndarray  # (4,), CLASSES order; index 0 is the identity branch
    branch_outputs: np.ndarray  # (4, H, W, 3), same order
    dominant_branch: str
    inference_ms: float


@dataclass(frozen=True)
class Sketch:
    image: np.ndarray  # (H, W) for a one-channel generator, else (H, W, 3); in [0, 1]
    channels: int
    inference_ms: float


def universal(registry: ModelRegistry, x: np.ndarray) -> Restoration:
    """Task 1: the universal denoising autoencoder."""
    outputs, ms = registry.get("udae").run({"input": to_batch(x)})
    return Restoration(from_batch(outputs["output"]), ms)


def hard_routed(registry: ModelRegistry, x: np.ndarray, oracle_class: str | None = None) -> HardRouting:
    """Task 2: the classifier picks one specialist; a clean prediction bypasses the experts.

    In oracle mode `oracle_class` (the known corruption) chooses the expert instead of the
    prediction; the classifier still runs so that its probabilities can be compared.
    """
    batch = to_batch(x)
    outputs, classifier_ms = registry.get("classifier").run({"input": batch})
    probabilities = outputs["probs"][0]
    predicted = CLASSES[int(np.argmax(probabilities))]
    routed = oracle_class if oracle_class is not None else predicted
    if routed == "clean":  # identity bypass: no restoration expert is run
        return HardRouting(x.copy(), probabilities, predicted, routed, "identity", classifier_ms, 0.0)
    expert = f"specialist_{routed}"
    outputs, expert_ms = registry.get(expert).run({"input": batch})
    return HardRouting(from_batch(outputs["output"]), probabilities, predicted, routed, expert,
                       classifier_ms, expert_ms)


def soft_mixture(registry: ModelRegistry, x: np.ndarray) -> SoftMixture:
    """Task 3: the mixture model returns its output, routing weights and every branch output."""
    outputs, ms = registry.get("moe").run({"input": to_batch(x)})
    weights = outputs["weights"][0]
    branches = outputs["branch_outputs"][0].transpose(0, 2, 3, 1)
    dominant = CLASSES[int(np.argmax(weights))]
    return SoftMixture(from_batch(outputs["output"]), weights, branches, dominant, ms)


def sketch(registry: ModelRegistry, photo: np.ndarray, style_index: int) -> Sketch:
    """Task 4: the generator maps a photo in [-1, 1] and a style index (0-2) to a sketch in [-1, 1]."""
    feeds = {"photo": to_batch(photo * 2.0 - 1.0), "style": np.array([style_index], dtype=np.int64)}
    outputs, ms = registry.get("generator").run(feeds)
    result = (outputs["sketch"][0] + 1.0) / 2.0  # (C, H, W) in [0, 1]
    channels = result.shape[0]
    image = result[0] if channels == 1 else result.transpose(1, 2, 0)
    return Sketch(image, channels, ms)
