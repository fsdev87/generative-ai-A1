"""GET /api/samples and GET /api/samples/{id}: the bundled clean sample images."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..dependencies import get_samples
from ..schemas import SampleInfo, SampleList, error_responses
from ..services.samples import MEDIA_TYPES, SampleCatalog

router = APIRouter(prefix="/api/samples", tags=["samples"])


@router.get("")
def list_samples(
    samples: Annotated[SampleCatalog, Depends(get_samples)],
    category: Literal["pets", "faces"] | None = None,
) -> SampleList:
    """Bundled clean images (pets for restoration, faces for sketches). Send an id as sample_id."""
    return SampleList(samples=[
        SampleInfo(id=s.id, category=s.category, name=s.path.name, url=f"/api/samples/{s.id}")
        for s in samples.select(category)
    ])


@router.get(
    "/{sample_id}",
    response_class=FileResponse,
    responses={200: {"content": {media: {} for media in sorted(set(MEDIA_TYPES.values()))},
                     "description": "The original image file"},
               **error_responses(404)},
)
def get_sample(sample_id: str, samples: Annotated[SampleCatalog, Depends(get_samples)]) -> FileResponse:
    """The sample's original image file, for display."""
    sample = samples.get(sample_id)
    return FileResponse(sample.path, media_type=sample.media_type)
