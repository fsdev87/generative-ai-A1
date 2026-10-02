"""Upload validation: formats, sizes, decompression bombs, EXIF orientation and colour modes."""
import io

import numpy as np
from PIL import Image, ImageOps

from conftest import PET_SAMPLE, decode_data_url, encode, make_client, photo_like, upload

CLEAN = {"corruption": "clean"}


def preprocess(image: Image.Image) -> np.ndarray:
    """Training preprocessing (src/data/pets.py) as uint8."""
    return np.asarray(image.convert("RGB").resize((128, 128), Image.BICUBIC))


def test_accepted_formats(client):
    for fmt, mime in (("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp"), ("BMP", "image/bmp")):
        response = client.post("/api/corrupt", data=CLEAN, files=upload(encode(photo_like(90, 120), fmt), "x", mime))
        assert response.status_code == 200, (fmt, response.json())
        assert response.json()["source"] == {"kind": "upload", "name": "x", "format": fmt, "width": 120, "height": 90}


def test_content_type_is_not_trusted(client):
    png_named_as_jpeg = upload(encode(photo_like(64, 64), "PNG"), "photo.jpg", "image/jpeg")
    assert client.post("/api/corrupt", data=CLEAN, files=png_named_as_jpeg).json()["source"]["format"] == "PNG"
    text_named_as_png = upload(b"hello, I am not an image", "x.png", "image/png")
    response = client.post("/api/corrupt", data=CLEAN, files=text_named_as_png)
    assert response.status_code == 415 and "JPEG, PNG, WEBP or BMP" in response.json()["detail"]


def test_unsupported_formats_are_415(client):
    for fmt in ("GIF", "TIFF"):
        response = client.post("/api/corrupt", data=CLEAN, files=upload(encode(photo_like(32, 32), fmt)))
        assert response.status_code == 415
        assert fmt in response.json()["detail"]


def test_empty_and_truncated_files_are_400(client):
    assert client.post("/api/corrupt", data=CLEAN, files=upload(b"", "x.png")).status_code == 400
    truncated = encode(photo_like(128, 128), "JPEG")[:700]
    response = client.post("/api/corrupt", data=CLEAN, files=upload(truncated, "x.jpg", "image/jpeg"))
    assert response.status_code == 400 and "corrupt or truncated" in response.json()["detail"]


def test_file_and_sample_id_are_exclusive(client):
    data = encode(photo_like(32, 32))
    both = client.post("/api/corrupt", data={**CLEAN, "sample_id": PET_SAMPLE}, files=upload(data))
    assert both.status_code == 400 and "not both" in both.json()["detail"]
    neither = client.post("/api/corrupt", data=CLEAN)
    assert neither.status_code == 400 and "No image" in neither.json()["detail"]
    unknown = client.post("/api/corrupt", data={**CLEAN, "sample_id": "pets-nope"})
    assert unknown.status_code == 404


def test_size_limits(model_dir, samples_dir):
    noise = np.random.default_rng(1).integers(0, 256, (400, 400, 3), dtype=np.uint8)
    big_file = encode(noise, "PNG")  # ~480 KB, incompressible
    with make_client(model_dir, samples_dir, max_upload_bytes=200_000) as client:
        response = client.post("/api/corrupt", data=CLEAN, files=upload(big_file))
        assert response.status_code == 413 and "larger than" in response.json()["detail"]
    # a body far above the limit is rejected from its Content-Length before it is read
    with make_client(model_dir, samples_dir, max_upload_bytes=1000) as client:
        huge = np.random.default_rng(2).integers(0, 256, (800, 800, 3), dtype=np.uint8)
        response = client.post("/api/corrupt", data=CLEAN, files=upload(encode(huge, "PNG")))
        assert response.status_code == 413 and "may be at most" in response.json()["detail"]


def test_decompression_bomb_is_413(model_dir, samples_dir):
    bomb = encode(Image.new("L", (3000, 3000)), "PNG")  # a few KB on disk, 9 megapixels decoded
    assert len(bomb) < 100_000
    with make_client(model_dir, samples_dir, max_image_pixels=4_000_000) as client:
        response = client.post("/api/corrupt", data=CLEAN, files=upload(bomb))
        assert response.status_code == 413 and "megapixels" in response.json()["detail"]


def test_exif_orientation_is_applied(client):
    stored = photo_like(60, 160)  # stored landscape; orientation 6 means "rotate 90 degrees clockwise"
    exif = Image.Exif()
    exif[0x0112] = 6
    data = encode(stored, "JPEG", exif=exif, quality=95)
    body = client.post("/api/corrupt", data=CLEAN, files=upload(data, "phone.jpg", "image/jpeg")).json()
    assert (body["source"]["width"], body["source"]["height"]) == (60, 160)  # upright portrait
    with Image.open(io.BytesIO(data)) as image:
        expected = preprocess(ImageOps.exif_transpose(image))
    assert np.array_equal(decode_data_url(body["clean"]), expected)


def test_colour_modes_become_rgb(client):
    grey = Image.fromarray(photo_like(64, 64)[..., 0])
    palette = Image.fromarray(photo_like(64, 64)).convert("P")
    for image in (grey, palette, Image.fromarray(photo_like(64, 64)).convert("CMYK")):
        fmt = "JPEG" if image.mode == "CMYK" else "PNG"
        response = client.post("/api/corrupt", data=CLEAN, files=upload(encode(image, fmt)))
        assert response.status_code == 200, image.mode
        assert decode_data_url(response.json()["clean"]).shape == (128, 128, 3)

    rgba = np.zeros((64, 64, 4), dtype=np.uint8)  # fully transparent black ...
    rgba[:32, :, :] = (255, 0, 0, 255)  # ... except an opaque red top half
    clean = decode_data_url(client.post("/api/corrupt", data=CLEAN, files=upload(encode(rgba))).json()["clean"])
    assert np.all(clean[100:, :] == 255)  # transparent pixels are composited on white
    assert np.all(clean[:20, :] == (255, 0, 0))


def test_sixteen_bit_greyscale_is_scaled(client):
    levels = np.tile(np.linspace(0, 65535, 64), (64, 1)).astype(np.uint16)
    body = client.post("/api/corrupt", data=CLEAN, files=upload(encode(Image.fromarray(levels)))).json()
    clean = decode_data_url(body["clean"])
    assert clean[:, :8].mean() < 30 and clean[:, -8:].mean() > 225  # a ramp, not clipped to white
