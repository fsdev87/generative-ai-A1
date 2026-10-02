"""FastAPI dependencies shared by the routers: app state, the input image and the corruption fields.

All POST endpoints take multipart/form-data. The image is either an uploaded `file` or the id
of a bundled sample (`sample_id`), never both.
"""
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, File, Form, Request, UploadFile
from PIL import Image

from .config import MB, Settings
from .errors import ApiError
from .schemas import CorruptionType, Level, SourceInfo
from .services.corruption import CUSTOM_PARAMS, CorruptionParams
from .services.images import decode_image
from .services.models import ModelRegistry
from .services.samples import SampleCatalog


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_samples(request: Request) -> SampleCatalog:
    return request.app.state.samples


@dataclass(frozen=True)
class InputImage:
    image: Image.Image  # upright RGB, original size
    info: SourceInfo
    started: float  # time.perf_counter() when reading began; the start of total_ms


def image_input(
    settings: Annotated[Settings, Depends(get_settings)],
    samples: Annotated[SampleCatalog, Depends(get_samples)],
    file: Annotated[UploadFile | None, File(
        description="Image file: JPEG, PNG, WEBP or BMP, at most MAX_UPLOAD_MB (default 10 MB)")] = None,
    sample_id: Annotated[str | None, Form(
        description="Id of a bundled sample (GET /api/samples), instead of a file")] = None,
) -> InputImage:
    """Read and decode the request image (400 / 404 / 413 / 415 on failure)."""
    started = time.perf_counter()
    if file is not None and sample_id is not None:
        raise ApiError(400, "Send either an image file or a sample_id, not both.")
    if file is not None:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(413, f"The image file is larger than {settings.max_upload_bytes / MB:g} MB.")
        kind, name = "upload", file.filename or "upload"
    elif sample_id is not None:
        sample = samples.get(sample_id)
        data = sample.path.read_bytes()
        kind, name = "sample", sample.id
    else:
        raise ApiError(400, "No image: send an image file in the 'file' field or a 'sample_id'.")
    image, image_format = decode_image(data, settings.max_image_pixels)
    info = SourceInfo(kind=kind, name=name, format=image_format, width=image.width, height=image.height)
    return InputImage(image, info, started)


def corruption_params(
    level: Annotated[Level | None, Form(description="Fixed test severity: low, medium or high")] = None,
    p: Annotated[float | None, Form(
        description=f"Custom salt-and-pepper probability, range {CUSTOM_PARAMS['salt']['p']}")] = None,
    k: Annotated[int | None, Form(
        description=f"Custom blur kernel size, one of {CUSTOM_PARAMS['blur']['k']}")] = None,
    sigma: Annotated[float | None, Form(
        description=f"Custom blur sigma, range {CUSTOM_PARAMS['blur']['sigma']}")] = None,
    n: Annotated[int | None, Form(
        description=f"Custom number of occlusion rectangles, one of {CUSTOM_PARAMS['occlusion']['n']}")] = None,
    cover: Annotated[float | None, Form(
        description=f"Custom occluded fraction, range {CUSTOM_PARAMS['occlusion']['cover']}")] = None,
    seed: Annotated[int | None, Form(
        description="Seed for the random parts of the corruption (0 to 2^31-1); random when omitted, "
                    "always returned")] = None,
) -> CorruptionParams:
    """Corruption fields other than the type: a level or custom parameters, and a seed."""
    return CorruptionParams(level=level, p=p, k=k, sigma=sigma, n=n, cover=cover, seed=seed)


# Restoration endpoints: the corruption to apply on the server first; omit it for an image to be
# used as uploaded (for example one that is already corrupted).
OptionalCorruption = Annotated[CorruptionType | None, Form(
    description="Corruption applied by the server before restoration (clean, salt, blur or occlusion); "
                "omit to use the image as uploaded")]


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
