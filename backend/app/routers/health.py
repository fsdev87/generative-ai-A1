"""GET /api/health: service status, runtime versions and the state of every model."""
import platform
import time
from typing import Annotated

import onnxruntime as ort
from fastapi import APIRouter, Depends, Request

from .. import __version__
from ..dependencies import get_registry, get_samples
from ..schemas import HealthResponse, ModelStatus, RuntimeInfo, TensorInfo, Workspaces, WorkspaceStatus
from ..services.models import PROVIDERS, WORKSPACE_MODELS, LoadedModel, ModelRegistry
from ..services.samples import SampleCatalog

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(
    request: Request,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    samples: Annotated[SampleCatalog, Depends(get_samples)],
) -> HealthResponse:
    """Always 200 while the server runs; `status` is "degraded" when a model is missing or broken."""
    models = [_model_status(model) for model in registry.models.values()]
    workspaces = {}
    for workspace, names in WORKSPACE_MODELS.items():
        missing = registry.missing(names)
        workspaces[workspace] = WorkspaceStatus(ready=not missing, missing=missing)
    return HealthResponse(
        status="ok" if all(model.loaded for model in models) else "degraded",
        version=__version__,
        started_at=request.app.state.started_at,
        uptime_s=round(time.monotonic() - request.app.state.started, 1),
        runtime=RuntimeInfo(
            python=platform.python_version(),
            onnxruntime=ort.__version__,
            available_providers=ort.get_available_providers(),
            providers=PROVIDERS,
        ),
        model_dir=str(registry.model_dir),
        models=models,
        workspaces=Workspaces(**workspaces),
        samples=samples.counts(),
    )


def _model_status(model: LoadedModel) -> ModelStatus:
    session = model.session
    return ModelStatus(
        name=model.contract.name,
        file=model.contract.file,
        description=model.contract.description,
        present=model.present,
        loaded=model.ready,
        size_bytes=model.size_bytes,
        load_ms=model.load_ms,
        inputs=_tensors(session.get_inputs()) if session else [],
        outputs=_tensors(session.get_outputs()) if session else [],
        metadata=model.metadata,
        error=model.error,
        warnings=model.warnings,
    )


def _tensors(nodes) -> list[TensorInfo]:
    return [TensorInfo(name=node.name, type=node.type, shape=list(node.shape)) for node in nodes]
