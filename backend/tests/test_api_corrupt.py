"""POST /api/corrupt: identical pixels to src/data/corruptions.py, determinism and validation."""
import numpy as np
import pytest
from PIL import Image

from conftest import PET_SAMPLE, decode_data_url
from src.data.corruptions import (BLUR_KERNELS, LEVELS, TEST_LEVELS, apply_spec, level_spec,
                                  severity_level)


@pytest.fixture(scope="module")
def clean_sample(samples_dir):
    """The pet sample preprocessed exactly as in src/data/pets.py: float32 (128, 128, 3)."""
    with Image.open(samples_dir / "pets" / "test_pet.jpg") as image:
        return np.asarray(image.convert("RGB").resize((128, 128), Image.BICUBIC)).astype(np.float32) / 255.0


def to_png_levels(x):
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def corrupt(client, **fields):
    return client.post("/api/corrupt", data={"sample_id": PET_SAMPLE, **fields})


@pytest.mark.parametrize("ctype", ["salt", "blur", "occlusion"])
@pytest.mark.parametrize("level", LEVELS)
def test_levels_match_the_training_code(client, clean_sample, ctype, level):
    response = corrupt(client, corruption=ctype, level=level, seed=1234)
    assert response.status_code == 200, response.json()
    body = response.json()
    spec = level_spec(ctype, level, np.random.default_rng(1234))  # what the API documents for a seed
    assert body["corruption"]["spec"] == spec
    assert np.array_equal(decode_data_url(body["image"]), to_png_levels(apply_spec(clean_sample, spec)))
    assert np.array_equal(decode_data_url(body["clean"]), to_png_levels(clean_sample))
    settings = body["corruption"]
    assert settings["type"] == ctype and settings["level"] == level and settings["custom"] is False
    assert settings["seed"] == 1234
    expected_params = dict(TEST_LEVELS[ctype][level])
    if ctype == "occlusion":
        expected_params["cover"] = spec["cover"]  # the cover actually achieved
        assert settings["rects"] == spec["rects"] and len(spec["rects"]) == spec["n"]
    else:
        assert settings["rects"] is None
    assert settings["params"] == expected_params


@pytest.mark.parametrize("fields", [
    {"corruption": "salt", "p": 0.05},
    {"corruption": "blur", "k": 7, "sigma": 0.9},
    {"corruption": "occlusion", "n": 2, "cover": 0.3},
])
def test_custom_parameters_match_the_training_code(client, clean_sample, fields):
    body = corrupt(client, seed=99, **fields).json()
    settings = body["corruption"]
    assert settings["custom"] is True and settings["level"] == severity_level(settings["spec"])
    assert np.array_equal(decode_data_url(body["image"]), to_png_levels(apply_spec(clean_sample, settings["spec"])))
    for name, value in fields.items():
        if name not in ("corruption", "cover"):
            assert settings["params"][name] == value
    if fields["corruption"] == "occlusion":
        assert abs(settings["params"]["cover"] - 0.3) <= 0.011


def test_custom_parameters_equal_to_a_level_reproduce_it(client):
    """Custom specs draw their random parts like level_spec, so equal parameters give equal pixels."""
    level = corrupt(client, corruption="salt", level="medium", seed=5).json()
    custom = corrupt(client, corruption="salt", p=TEST_LEVELS["salt"]["medium"]["p"], seed=5).json()
    assert custom["image"] == level["image"]


def test_clean_is_the_preprocessed_image(client, clean_sample):
    body = corrupt(client, corruption="clean").json()
    assert body["image"] == body["clean"]
    assert body["corruption"]["level"] is None and body["corruption"]["params"] == {}


def test_same_seed_same_result(client):
    for ctype in ("salt", "occlusion"):
        first = corrupt(client, corruption=ctype, level="high", seed=42).json()
        second = corrupt(client, corruption=ctype, level="high", seed=42).json()
        other = corrupt(client, corruption=ctype, level="high", seed=43).json()
        assert first["image"] == second["image"] and first["corruption"] == second["corruption"]
        assert first["image"] != other["image"]


def test_random_seed_is_returned_and_reproducible(client):
    first = corrupt(client, corruption="occlusion", level="medium").json()
    seed = first["corruption"]["seed"]
    assert 0 <= seed < 2**31
    again = corrupt(client, corruption="occlusion", level="medium", seed=seed).json()
    assert again["image"] == first["image"]


@pytest.mark.parametrize("fields, status, message", [
    ({"corruption": "salt"}, 400, "needs a level"),
    ({"corruption": "salt", "level": "low", "p": 0.05}, 400, "not both"),
    ({"corruption": "clean", "level": "low"}, 400, "takes no level"),
    ({"corruption": "salt", "k": 3}, 400, "not a parameter"),
    ({"corruption": "blur", "k": 5}, 400, "missing sigma"),
    ({"corruption": "salt", "p": 0.5}, 422, "training range"),
    ({"corruption": "blur", "k": 4, "sigma": 1.0}, 422, "one of 3, 5, 7"),
    ({"corruption": "blur", "k": 3, "sigma": 3.0}, 422, "training range"),
    ({"corruption": "occlusion", "n": 4, "cover": 0.2}, 422, "one of 1, 2, 3"),
    ({"corruption": "occlusion", "n": 1, "cover": 0.05}, 422, "training range"),
    ({"corruption": "salt", "level": "low", "seed": -1}, 422, "seed"),
    ({"corruption": "salt", "p": "nan"}, 422, "training range"),
    ({"corruption": "fog", "level": "low"}, 422, "corruption"),
    ({"corruption": "salt", "level": "extreme"}, 422, "level"),
    ({"corruption": "salt", "p": "a lot"}, 422, "p"),
    ({"level": "low"}, 422, "corruption"),
])
def test_invalid_requests(client, fields, status, message):
    response = corrupt(client, **fields)
    assert response.status_code == status, response.json()
    assert message in response.json()["detail"]


def test_blur_kernels_follow_the_training_code(client):
    assert sorted({corrupt(client, corruption="blur", k=k, sigma=1.0).status_code for k in BLUR_KERNELS}) == [200]
