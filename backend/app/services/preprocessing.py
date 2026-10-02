"""Model-input preprocessing. Each function must match the preprocessing used in training."""
import numpy as np
from PIL import Image

IMAGE_SIZE = 128


def restoration_input(image: Image.Image) -> np.ndarray:
    """Tasks 1-3, as in src/data/pets.py: RGB, plain bicubic resize to 128x128 (no crop, so the
    aspect ratio is not kept), float32 in [0, 1]. Returns an (H, W, 3) array."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def sketch_photo_input(image: Image.Image) -> np.ndarray:
    """Task 4 photo preprocessing; must match the Task 4 training pipeline (FS2K loader).

    Currently: centre crop to a square, bicubic resize to 128x128, float32 in [0, 1], (H, W, 3).
    The generator itself takes [-1, 1]; that scaling is done in services/inference.py.
    """
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    square = image.convert("RGB").crop((left, top, left + side, top + side))
    resized = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def to_batch(x: np.ndarray) -> np.ndarray:
    """(H, W, C) image -> contiguous float32 (1, C, H, W) batch for ONNX Runtime."""
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def from_batch(y: np.ndarray) -> np.ndarray:
    """First item of an (N, C, H, W) model output -> (H, W, C) image."""
    return y[0].transpose(1, 2, 0)
