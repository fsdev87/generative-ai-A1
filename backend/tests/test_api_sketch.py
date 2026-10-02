"""POST /api/sketch with the dummy generator (one- and three-channel variants)."""
import numpy as np
import pytest
from PIL import Image

from conftest import FACE_SAMPLE, decode_data_url, encode, make_client, photo_like, upload


def sketch(client, **fields):
    return client.post("/api/sketch", data={"sample_id": FACE_SAMPLE, **fields})


def test_sketch_styles(client):
    outputs = []
    for style in (1, 2, 3):
        response = sketch(client, style=style)
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["style"] == style and body["style_index"] == style - 1
        assert body["model"] == "generator.onnx" and body["sketch_channels"] == 1
        assert decode_data_url(body["sketch"]).shape == (128, 128)  # greyscale PNG
        assert decode_data_url(body["photo"]).shape == (128, 128, 3)
        assert body["timing"]["inference_ms"] > 0
        outputs.append(body["sketch"])
    assert len(set(outputs)) == 3  # the style changes the result


def test_photo_is_centre_cropped(client):
    """A 300x100 photo whose centre square is white and sides black becomes all white."""
    photo = np.zeros((100, 300, 3), dtype=np.uint8)
    photo[:, 100:200] = 255
    body = client.post("/api/sketch", data={"style": 1}, files=upload(encode(photo))).json()
    assert np.all(decode_data_url(body["photo"]) == 255)
    assert (body["source"]["width"], body["source"]["height"]) == (300, 100)


@pytest.mark.parametrize("style", ["0", "4", "two", ""])
def test_invalid_style(client, style):
    response = sketch(client, style=style)
    assert response.status_code == 422 and "style" in response.json()["detail"]


def test_three_channel_generator(tmp_path, dummy_models, samples_dir):
    dummy_models.write_dummy_models(tmp_path, ["generator"], sketch_channels=3)
    with make_client(tmp_path, samples_dir) as client:
        body = client.post("/api/sketch", data={"style": 2},
                           files=upload(encode(Image.fromarray(photo_like(140, 120))))).json()
        assert body["sketch_channels"] == 3
        assert decode_data_url(body["sketch"]).shape == (128, 128, 3)
