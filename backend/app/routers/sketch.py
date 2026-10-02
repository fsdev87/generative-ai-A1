"""POST /api/sketch: face-to-sketch generation (Task 4)."""
from typing import Annotated

from fastapi import APIRouter, Depends, Form

from ..dependencies import InputImage, elapsed_ms, get_registry, image_input
from ..schemas import SketchResponse, Timing, error_responses
from ..services import inference
from ..services.images import to_data_url
from ..services.models import ModelRegistry
from ..services.preprocessing import sketch_photo_input

router = APIRouter(prefix="/api", tags=["sketch"])


@router.post("/sketch", responses=error_responses(400, 404, 413, 415, 422, 500, 503))
def generate_sketch(
    source: Annotated[InputImage, Depends(image_input)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    style: Annotated[int, Form(ge=1, le=3, description="Sketch style shown to users: 1, 2 or 3")],
) -> SketchResponse:
    """Task 4: generator.onnx turns a face photo into a sketch in the chosen style."""
    photo = sketch_photo_input(source.image)
    result = inference.sketch(registry, photo, style - 1)
    photo_url, sketch_url = to_data_url(photo), to_data_url(result.image)
    return SketchResponse(
        model="generator.onnx",
        style=style,
        style_index=style - 1,
        photo=photo_url,
        sketch=sketch_url,
        sketch_channels=result.channels,
        source=source.info,
        timing=Timing(inference_ms=result.inference_ms, total_ms=elapsed_ms(source.started)),
    )
