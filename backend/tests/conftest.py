"""Shared fixtures for the backend tests. Run from the repository root:
    python -m pytest backend/tests -q

The dummy ONNX models are written once per session with backend/scripts/make_dummy_models.py
(needs torch); sample images are generated at test time.
"""
import base64
import importlib.util
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
for path in (str(REPO_ROOT), str(BACKEND_DIR)):  # `src` (training code) and `app` (backend)
    if path not in sys.path:
        sys.path.insert(0, path)

with warnings.catch_warnings():  # starlette's TestClient warns that it prefers httpx2
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

RNG = np.random.default_rng(0)
PET_SAMPLE = "pets-test_pet"
FACE_SAMPLE = "faces-test_face"


def _load_dummy_script():
    spec = importlib.util.spec_from_file_location("make_dummy_models", BACKEND_DIR / "scripts" / "make_dummy_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def dummy_models():
    return _load_dummy_script()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, dummy_models):
    path = tmp_path_factory.mktemp("models")
    dummy_models.write_dummy_models(path)
    return path


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    """pets/test_pet.jpg (300x200 photo-like image) and faces/test_face.png (200x260)."""
    root = tmp_path_factory.mktemp("samples")
    (root / "pets").mkdir()
    (root / "faces").mkdir()
    Image.fromarray(photo_like(200, 300)).save(root / "pets" / "test_pet.jpg", quality=95)
    Image.fromarray(photo_like(260, 200)).save(root / "faces" / "test_face.png")
    (root / "pets" / "notes.txt").write_text("not an image")  # ignored by the catalog
    return root


def make_client(model_dir, samples_dir, **overrides):
    settings = Settings(model_dir=Path(model_dir), samples_dir=Path(samples_dir), **overrides)
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def client(model_dir, samples_dir):
    with make_client(model_dir, samples_dir) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def photo_like(height, width, seed=0):
    """Smooth colour gradients plus mild noise: a uint8 (H, W, 3) image."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(height, width)
    base = np.stack([0.5 + 0.4 * np.sin(6 * x), 0.5 + 0.4 * np.cos(5 * y), 0.5 + 0.3 * np.sin(4 * (x + y))], axis=-1)
    return np.clip((base + rng.normal(0, 0.03, base.shape)) * 255, 0, 255).astype(np.uint8)


def solid(colour, size=(64, 64)):
    return np.full((size[1], size[0], 3), colour, dtype=np.uint8)


def encode(array_or_image, fmt="PNG", **save_args) -> bytes:
    image = array_or_image if isinstance(array_or_image, Image.Image) else Image.fromarray(array_or_image)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_args)
    return buffer.getvalue()


def upload(data: bytes, name="image.png", content_type="image/png"):
    return {"file": (name, data, content_type)}


def decode_data_url(url: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    with Image.open(io.BytesIO(base64.b64decode(url[len(prefix):]))) as image:
        assert image.format == "PNG"
        return np.asarray(image)
