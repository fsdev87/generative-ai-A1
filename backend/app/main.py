"""Application factory: settings, model loading at startup, middleware, error handlers and routers.

Run locally from the repository root (the backend imports src/data/corruptions.py):
    python -m uvicorn app.main:app --app-dir backend --port 8000
Interactive API docs: http://localhost:8000/api/docs
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from . import __version__
from .config import MB, Settings
from .errors import ApiError
from .routers import corruption, health, restoration, samples, sketch
from .services.models import ModelRegistry
from .services.samples import SampleCatalog

FORM_OVERHEAD_BYTES = MB  # multipart framing and the small form fields next to the image file


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = ModelRegistry(settings.model_dir)
        app.state.samples = SampleCatalog(settings.samples_dir)
        app.state.started_at = datetime.now(timezone.utc)
        app.state.started = time.monotonic()
        yield

    app = FastAPI(
        title="GenAI Assignment 1 API",
        version=__version__,
        description="Image restoration (Tasks 1-3) and face-to-sketch generation (Task 4) with ONNX Runtime.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.add_middleware(RequestSizeLimit, max_upload_bytes=settings.max_upload_bytes)
    # Added last, so it is the outermost layer and error responses get CORS headers too.
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    for module in (health, samples, corruption, restoration, sketch):
        app.include_router(module.router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/api/docs")

    return app


class RequestSizeLimit:
    """ASGI middleware: answers 413 before reading a body whose declared size is far above the
    upload limit. The exact per-file limit is checked when the file is read (dependencies.py)."""

    def __init__(self, app, max_upload_bytes: int):
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > self.max_upload_bytes + FORM_OVERHEAD_BYTES:
                detail = (f"The request is {int(length) / MB:.1f} MB; "
                          f"image files may be at most {self.max_upload_bytes / MB:g} MB.")
                await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with a readable `detail` string, like every other error, plus the per-field errors."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    summary = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in errors)
    return JSONResponse({"detail": f"Invalid request: {summary}", "errors": errors}, status_code=422)


app = create_app()
