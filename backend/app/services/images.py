"""Decoding and validating uploaded images, and encoding results as PNG data URLs."""
import base64
import io
import math

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors import ApiError

# Pillow format -> name reported to the client. MPO is the multi-picture JPEG written by some
# phones and cameras.
ACCEPTED_FORMATS = {"JPEG": "JPEG", "MPO": "JPEG", "PNG": "PNG", "WEBP": "WEBP", "BMP": "BMP"}


def decode_image(data: bytes, max_pixels: int) -> tuple[Image.Image, str]:
    """Decode a file with Pillow (the declared content type is not trusted).

    Returns the upright RGB image and its format. Raises 400 for an empty or corrupt file,
    413 when the image has more than `max_pixels` pixels (decompression bomb) and 415 for
    anything that is not a JPEG, PNG, WEBP or BMP image.
    """
    if not data:
        raise ApiError(400, "The image file is empty.")
    try:
        image = Image.open(io.BytesIO(data))  # reads the header only
    except Image.DecompressionBombError as exc:
        raise ApiError(413, f"The image has too many pixels: {exc}") from exc
    except UnidentifiedImageError as exc:
        raise ApiError(415, "The file is not a supported image; use JPEG, PNG, WEBP or BMP.") from exc
    except Exception as exc:
        raise ApiError(400, f"The image file could not be read: {exc}") from exc
    if image.format not in ACCEPTED_FORMATS:
        raise ApiError(415, f"{image.format} images are not supported; use JPEG, PNG, WEBP or BMP.")
    width, height = image.size
    if width * height > max_pixels:
        raise ApiError(413, f"The image is {width}x{height} pixels; the limit is {max_pixels / 1e6:g} megapixels.")
    try:
        image.load()  # truncated or corrupt data only shows up when the pixels are decoded
        rgb = to_rgb(image)
    except Exception as exc:
        raise ApiError(400, f"The image file is corrupt or truncated: {exc}") from exc
    return rgb, ACCEPTED_FORMATS[image.format]


def to_rgb(image: Image.Image) -> Image.Image:
    """Upright RGB: applies the EXIF orientation (phone and webcam photos), scales 16-bit
    greyscale to 8 bits and puts transparent images on a white background."""
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # malformed EXIF block: keep the stored orientation
        pass
    if image.mode == "I" or image.mode.startswith("I;16"):
        # 16-bit greyscale PNG: convert("RGB") would clip every value above 255 to white
        levels = np.asarray(image, dtype=np.float32) / 257.0
        image = Image.fromarray(np.clip(np.rint(levels), 0, 255).astype(np.uint8))
    if image.has_transparency_data:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return image.convert("RGB")


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Float image in [0, 1] -> uint8, rounding to the nearest level (NaN -> 0)."""
    return np.clip(np.rint(np.nan_to_num(x) * 255.0), 0, 255).astype(np.uint8)


def quantize(x: np.ndarray) -> np.ndarray:
    """Round a float image to the 256 levels per channel that a PNG stores."""
    return to_uint8(x).astype(np.float32) / 255.0


def to_data_url(x: np.ndarray) -> str:
    """Float image in [0, 1], (H, W, 3) or greyscale (H, W), as a base64 PNG data URL."""
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(x)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def psnr(x: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB for images in [0, 1], capped at 100 dB as in src.common.metrics.psnr."""
    mse = float(np.mean((np.clip(x, 0.0, 1.0).astype(np.float64) - reference) ** 2))
    return 100.0 if mse <= 1e-10 else min(100.0, 10.0 * math.log10(1.0 / mse))


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """1-D normalised Gaussian; the 2-D window of src.common.losses is its outer product."""
    t = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(t**2) / (2.0 * sigma**2))
    return g / g.sum()


def _filter_valid(image: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Separable Gaussian filter of an (H, W, C) image over the valid region only (no padding),
    like F.conv2d without padding: output (H - k + 1, W - k + 1, C)."""
    k = len(g)
    height, width = image.shape[:2]
    rows = sum(g[i] * image[i:height - k + 1 + i] for i in range(k))
    return sum(g[j] * rows[:, j:width - k + 1 + j] for j in range(k))


def ssim(x: np.ndarray, reference: np.ndarray, data_range: float = 1.0, window_size: int = 11,
         sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> float:
    """SSIM (Wang et al., 2004) of an image against the reference, both (H, W, C) or (H, W) in
    [0, 1]. Same definition as src.common.losses.ssim (no torch here): 11x11 Gaussian window with
    sigma 1.5, valid region only, mean over channels and positions. 1.0 for identical images."""
    x = np.clip(x, 0.0, 1.0).astype(np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    if x.shape != y.shape or min(x.shape[:2]) < window_size:
        raise ValueError(f"ssim needs two images of the same size, at least {window_size} pixels")
    g = _gaussian_window(window_size, sigma)
    mu_x, mu_y = _filter_valid(x, g), _filter_valid(y, g)
    var_x = _filter_valid(x * x, g) - mu_x**2
    var_y = _filter_valid(y * y, g) - mu_y**2
    cov = _filter_valid(x * y, g) - mu_x * mu_y
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())
