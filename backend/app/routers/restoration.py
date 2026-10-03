"""Restoration workspaces: universal autoencoder (Task 1), hard routing (Task 2) and soft
mixture of experts (Task 3).

Every endpoint takes an image (upload or sample) and, optionally, a corruption that the server
applies to the preprocessed clean image first. Without `corruption` the image is restored as
uploaded, e.g. an image that is already corrupted.
"""
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, Form

from ..dependencies import (InputImage, OptionalCorruption, corruption_params, elapsed_ms, get_registry,
                            image_input)
from ..errors import ApiError
from ..schemas import (BranchImages, ClassScores, CorruptionSettings, HardResponse, HardTiming, Metrics,
                       MoEResponse, RoutingMode, Timing, UniversalResponse, error_responses)
from ..services import inference
from ..services.corruption import CLASSES, AppliedCorruption, CorruptionParams, build_spec, corrupt
from ..services.images import psnr, ssim, to_data_url
from ..services.models import ModelRegistry
from ..services.preprocessing import restoration_input

router = APIRouter(prefix="/api/restore", tags=["restoration"])
ERRORS = error_responses(400, 404, 413, 415, 422, 500, 503)


@dataclass(frozen=True)
class ModelInput:
    x: np.ndarray  # what the model receives: (128, 128, 3) float32 in [0, 1]
    reference: np.ndarray | None  # the clean image, when the server applied the corruption
    applied: AppliedCorruption | None


def prepare(source: InputImage, corruption: str | None, params: CorruptionParams) -> ModelInput:
    """Preprocess as in training and apply the requested corruption, if any."""
    clean = restoration_input(source.image)
    if corruption is None:
        if not params.is_empty():
            raise ApiError(400, "level, p, k, sigma, n, cover and seed describe a corruption; "
                                "also send corruption=<type>, or leave them out.")
        return ModelInput(clean, None, None)
    applied = build_spec(corruption, params)
    return ModelInput(corrupt(clean, applied.spec), clean, applied)


def result_fields(model_input: ModelInput, output: np.ndarray, source: InputImage) -> dict:
    """The fields that all restoration responses share."""
    fields = {"input": to_data_url(model_input.x), "output": to_data_url(output),
              "reference": None, "corruption": None, "metrics": None, "source": source.info}
    if model_input.applied is not None:
        fields["reference"] = to_data_url(model_input.reference)
        fields["corruption"] = CorruptionSettings(**model_input.applied.describe())
        x, reference = model_input.x, model_input.reference
        fields["metrics"] = Metrics(psnr_input_db=psnr(x, reference), psnr_output_db=psnr(output, reference),
                                    ssim_input=ssim(x, reference), ssim_output=ssim(output, reference))
    return fields


def class_scores(values: np.ndarray) -> ClassScores:
    return ClassScores(**{name: float(value) for name, value in zip(CLASSES, values)})


@router.post("/universal", responses=ERRORS)
def restore_universal(
    source: Annotated[InputImage, Depends(image_input)],
    params: Annotated[CorruptionParams, Depends(corruption_params)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    corruption: OptionalCorruption = None,
) -> UniversalResponse:
    """Task 1: restore with the universal denoising autoencoder (udae.onnx)."""
    model_input = prepare(source, corruption, params)
    result = inference.universal(registry, model_input.x)
    fields = result_fields(model_input, result.output, source)
    timing = Timing(inference_ms=result.inference_ms, total_ms=elapsed_ms(source.started))
    return UniversalResponse(model="udae.onnx", timing=timing, **fields)


@router.post("/hard", responses=ERRORS)
def restore_hard(
    source: Annotated[InputImage, Depends(image_input)],
    params: Annotated[CorruptionParams, Depends(corruption_params)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    corruption: OptionalCorruption = None,
    routing_mode: Annotated[RoutingMode, Form(
        description="predicted: the classifier's argmax chooses the expert; oracle: the true corruption "
                    "chooses it (only when the server applies the corruption)")] = "predicted",
) -> HardResponse:
    """Task 2: classifier.onnx predicts the corruption, then one specialist restores the image;
    a clean prediction uses the identity bypass and runs no expert."""
    if routing_mode == "oracle" and corruption is None:
        raise ApiError(400, "Oracle routing needs the true corruption, which is only known when the server "
                            "applies it: send corruption=<type> (with a level or parameters), or use "
                            "routing_mode=predicted for an uploaded image.")
    model_input = prepare(source, corruption, params)
    result = inference.hard_routed(registry, model_input.x, oracle_class=corruption if routing_mode == "oracle" else None)
    fields = result_fields(model_input, result.output, source)
    timing = HardTiming(classifier_ms=result.classifier_ms, expert_ms=result.expert_ms,
                        inference_ms=result.classifier_ms + result.expert_ms, total_ms=elapsed_ms(source.started))
    return HardResponse(
        routing_mode=routing_mode,
        probabilities=class_scores(result.probabilities),
        predicted_class=result.predicted_class,
        true_class=corruption,
        routed_class=result.routed_class,
        selected_expert=result.selected_expert,
        timing=timing,
        **fields,
    )


@router.post("/moe", responses=ERRORS)
def restore_moe(
    source: Annotated[InputImage, Depends(image_input)],
    params: Annotated[CorruptionParams, Depends(corruption_params)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    corruption: OptionalCorruption = None,
) -> MoEResponse:
    """Task 3: the soft mixture of experts (moe.onnx) with its routing weights and the output
    of every branch."""
    model_input = prepare(source, corruption, params)
    result = inference.soft_mixture(registry, model_input.x)
    fields = result_fields(model_input, result.output, source)
    branches = BranchImages(**{name: to_data_url(image) for name, image in zip(CLASSES, result.branch_outputs)})
    timing = Timing(inference_ms=result.inference_ms, total_ms=elapsed_ms(source.started))
    return MoEResponse(
        model="moe.onnx",
        weights=class_scores(result.weights),
        dominant_branch=result.dominant_branch,
        branch_outputs=branches,
        timing=timing,
        **fields,
    )
