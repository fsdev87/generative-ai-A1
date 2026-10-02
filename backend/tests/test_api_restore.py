"""POST /api/restore/{universal,hard,moe} with the dummy models.

The dummy classifier and gate score the mean colour (grey -> clean, red -> salt, green -> blur,
blue -> occlusion), so solid-colour uploads choose the predicted class.
"""
import numpy as np
import pytest

from conftest import PET_SAMPLE, decode_data_url, encode, solid, upload

GREY, RED, GREEN, BLUE = (128, 128, 128), (220, 30, 30), (30, 220, 30), (30, 30, 220)
ENDPOINTS = ("/api/restore/universal", "/api/restore/hard", "/api/restore/moe")


def post(client, endpoint, image=None, **fields):
    files = upload(encode(solid(image))) if image is not None else None
    data = fields if image is not None else {"sample_id": PET_SAMPLE, **fields}
    return client.post(endpoint, data=data, files=files)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_server_side_corruption(client, endpoint):
    response = post(client, endpoint, corruption="blur", level="medium", seed=3)
    assert response.status_code == 200, response.json()
    body = response.json()
    reference, model_input, output = (decode_data_url(body[k]) for k in ("reference", "input", "output"))
    assert reference.shape == model_input.shape == output.shape == (128, 128, 3)
    assert body["corruption"]["type"] == "blur" and body["corruption"]["seed"] == 3
    # the model input is exactly what /api/corrupt returns for the same request
    corrupted = client.post("/api/corrupt", data={"sample_id": PET_SAMPLE, "corruption": "blur",
                                                  "level": "medium", "seed": 3}).json()
    assert body["input"] == corrupted["image"] and body["reference"] == corrupted["clean"]
    assert 0 < body["metrics"]["psnr_input_db"] <= 100
    assert body["timing"]["total_ms"] >= body["timing"]["inference_ms"] > 0
    assert body["source"]["kind"] == "sample"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_uploaded_image_is_used_as_is(client, endpoint):
    body = post(client, endpoint, image=GREY).json()
    assert body["reference"] is None and body["corruption"] is None and body["metrics"] is None
    assert np.all(decode_data_url(body["input"]) == 128)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_corruption_fields_need_a_corruption_type(client, endpoint):
    response = post(client, endpoint, image=GREY, level="high")
    assert response.status_code == 400 and "corruption=<type>" in response.json()["detail"]


def test_reuploading_the_corrupted_png_gives_the_same_result(client):
    server_side = post(client, "/api/restore/universal", corruption="blur", level="high", seed=8).json()
    png = decode_data_url(server_side["input"])
    uploaded = client.post("/api/restore/universal", files=upload(encode(png))).json()
    assert uploaded["input"] == server_side["input"] and uploaded["output"] == server_side["output"]


def test_universal_reports_model(client):
    body = post(client, "/api/restore/universal", corruption="salt", level="low").json()
    assert body["model"] == "udae.onnx"
    assert set(body["timing"]) == {"inference_ms", "total_ms"}


# --------------------------------------------------------------------------- #
# Hard routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("colour, expected", [(RED, "salt"), (GREEN, "blur"), (BLUE, "occlusion")])
def test_hard_predicted_routing_uses_the_argmax(client, colour, expected):
    body = post(client, "/api/restore/hard", image=colour).json()
    probabilities = body["probabilities"]
    assert list(probabilities) == ["clean", "salt", "blur", "occlusion"]
    assert abs(sum(probabilities.values()) - 1) < 1e-4
    assert body["predicted_class"] == max(probabilities, key=probabilities.get) == expected
    assert body["routed_class"] == expected and body["selected_expert"] == f"specialist_{expected}"
    assert body["routing_mode"] == "predicted" and body["true_class"] is None
    timing = body["timing"]
    assert timing["expert_ms"] > 0
    assert timing["inference_ms"] == pytest.approx(timing["classifier_ms"] + timing["expert_ms"])


def test_hard_clean_prediction_bypasses_the_experts(client):
    body = post(client, "/api/restore/hard", image=GREY).json()
    assert body["predicted_class"] == "clean" and body["selected_expert"] == "identity"
    assert body["timing"]["expert_ms"] == 0
    assert body["output"] == body["input"]


@pytest.mark.parametrize("ctype", ["clean", "salt", "blur", "occlusion"])
def test_hard_oracle_routing_uses_the_true_class(client, ctype):
    fields = {"corruption": ctype} if ctype == "clean" else {"corruption": ctype, "level": "high"}
    body = post(client, "/api/restore/hard", routing_mode="oracle", **fields).json()
    assert body["routing_mode"] == "oracle" and body["true_class"] == ctype == body["routed_class"]
    assert body["selected_expert"] == ("identity" if ctype == "clean" else f"specialist_{ctype}")
    if ctype == "clean":
        assert body["output"] == body["input"] == body["reference"]
        assert body["metrics"]["psnr_output_db"] == 100


def test_hard_oracle_differs_from_prediction(client):
    """A red image corrupted with blur: predicted salt, oracle blur."""
    red = upload(encode(solid(RED, (128, 128))))
    fields = {"corruption": "blur", "level": "low"}
    predicted = client.post("/api/restore/hard", data=fields, files=red).json()
    red = upload(encode(solid(RED, (128, 128))))
    oracle = client.post("/api/restore/hard", data={**fields, "routing_mode": "oracle"}, files=red).json()
    assert predicted["selected_expert"] == "specialist_salt" and predicted["true_class"] == "blur"
    assert oracle["selected_expert"] == "specialist_blur" and oracle["predicted_class"] == "salt"


def test_hard_oracle_needs_a_server_side_corruption(client):
    response = post(client, "/api/restore/hard", image=GREY, routing_mode="oracle")
    assert response.status_code == 400 and "Oracle routing" in response.json()["detail"]
    assert post(client, "/api/restore/hard", image=GREY, routing_mode="psychic").status_code == 422


# --------------------------------------------------------------------------- #
# Soft mixture of experts
# --------------------------------------------------------------------------- #
def test_moe_weights_and_branches(client):
    body = post(client, "/api/restore/moe", corruption="occlusion", level="medium", seed=1).json()
    weights = body["weights"]
    assert list(weights) == ["clean", "salt", "blur", "occlusion"]
    assert abs(sum(weights.values()) - 1) < 1e-4
    assert body["dominant_branch"] == max(weights, key=weights.get)
    assert body["model"] == "moe.onnx"
    branches = {name: decode_data_url(url).astype(np.float64) for name, url in body["branch_outputs"].items()}
    assert list(branches) == list(weights)
    assert np.array_equal(branches["clean"], decode_data_url(body["input"]))  # identity branch
    mixture = sum(weights[name] * branches[name] for name in weights)
    assert np.abs(mixture - decode_data_url(body["output"])).max() <= 1.5  # 8-bit rounding only


@pytest.mark.parametrize("colour, expected", [(GREY, "clean"), (BLUE, "occlusion")])
def test_moe_dominant_branch(client, colour, expected):
    assert post(client, "/api/restore/moe", image=colour).json()["dominant_branch"] == expected
