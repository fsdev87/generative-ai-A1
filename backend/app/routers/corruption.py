"""POST /api/corrupt (runtime corruption) and GET /api/options (choices for the UI controls)."""
from typing import Annotated

from fastapi import APIRouter, Depends, Form

from ..config import MB, Settings
from ..dependencies import InputImage, corruption_params, get_settings, image_input
from ..schemas import (CorruptionOption, CorruptionSettings, CorruptionType, CorruptResponse, OptionsResponse,
                       ParamOption, error_responses)
from ..services.corruption import (CLASSES, CUSTOM_PARAMS, INTEGER_PARAMS, LEVELS, TEST_LEVELS, CorruptionParams,
                                   build_spec, corrupt)
from ..services.images import to_data_url
from ..services.preprocessing import IMAGE_SIZE, restoration_input

router = APIRouter(prefix="/api", tags=["corruption"])


@router.post("/corrupt", responses=error_responses(400, 404, 413, 415, 422))
def corrupt_image(
    source: Annotated[InputImage, Depends(image_input)],
    params: Annotated[CorruptionParams, Depends(corruption_params)],
    corruption: Annotated[CorruptionType, Form(description="clean, salt, blur or occlusion")],
) -> CorruptResponse:
    """Apply one corruption with the training code (src/data/corruptions.py) to the
    preprocessed 128x128 image: a fixed level or custom parameters, and an optional seed."""
    clean = restoration_input(source.image)
    applied = build_spec(corruption, params)
    return CorruptResponse(
        image=to_data_url(corrupt(clean, applied.spec)),
        clean=to_data_url(clean),
        corruption=CorruptionSettings(**applied.describe()),
        source=source.info,
    )


@router.get("/options")
def options(settings: Annotated[Settings, Depends(get_settings)]) -> OptionsResponse:
    """Corruption types, fixed levels and custom parameter ranges, sketch styles and limits,
    taken from the training code so the UI controls always match it."""
    corruption_options = [CorruptionOption(type="clean", params=[], levels={})]
    for ctype, params in CUSTOM_PARAMS.items():
        param_options = [
            ParamOption(name=name, min=min(allowed), max=max(allowed),
                        choices=list(allowed) if name in INTEGER_PARAMS else None)
            for name, allowed in params.items()
        ]
        corruption_options.append(
            CorruptionOption(type=ctype, params=param_options, levels=TEST_LEVELS[ctype]))
    return OptionsResponse(
        classes=list(CLASSES),
        levels=list(LEVELS),
        corruptions=corruption_options,
        styles=[1, 2, 3],
        image_size=IMAGE_SIZE,
        max_upload_mb=settings.max_upload_bytes / MB,
        max_image_megapixels=settings.max_image_pixels / 1e6,
    )
