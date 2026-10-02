"""Backend settings, read from environment variables.

MODEL_DIR             folder with the ONNX files and their sidecar JSONs (default: backend/models;
                      /models in Docker)
SAMPLES_DIR           folder with the bundled clean sample images (default: backend/app/samples)
CORS_ORIGINS          comma-separated browser origins allowed to call the API, or "*"
                      (default: http://localhost:5173,http://localhost:3000)
MAX_UPLOAD_MB         largest accepted image file in MB (default 10)
MAX_IMAGE_MEGAPIXELS  largest accepted decoded image; guards against decompression bombs (default 50)
"""
import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
MB = 1024 * 1024


@dataclass(frozen=True)
class Settings:
    model_dir: Path = BACKEND_DIR / "models"
    samples_dir: Path = APP_DIR / "samples"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:3000")
    max_upload_bytes: int = 10 * MB
    max_image_pixels: int = 50_000_000

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        default = cls()
        origins = tuple(o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            model_dir=Path(env.get("MODEL_DIR") or default.model_dir),
            samples_dir=Path(env.get("SAMPLES_DIR") or default.samples_dir),
            cors_origins=origins or default.cors_origins,
            max_upload_bytes=int(_positive(env, "MAX_UPLOAD_MB", default.max_upload_bytes / MB) * MB),
            max_image_pixels=int(_positive(env, "MAX_IMAGE_MEGAPIXELS", default.max_image_pixels / 1e6) * 1e6),
        )


def _positive(env, name, default):
    """A positive number from the environment; a typo stops the server with a clear message."""
    raw = env.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise ValueError(f"environment variable {name} must be a positive number, got {raw!r}")
    return value
