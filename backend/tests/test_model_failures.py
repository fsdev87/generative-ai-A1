"""Missing, partial and broken models: the app keeps running, health explains, endpoints answer 503."""
import json
import shutil

import pytest

from conftest import PET_SAMPLE, encode, make_client, solid, upload

REQUESTS = {
    "/api/restore/universal": ("udae.onnx", {"corruption": "salt", "level": "low"}),
    "/api/restore/hard": ("classifier.onnx", {"corruption": "salt", "level": "low"}),
    "/api/restore/moe": ("moe.onnx", {"corruption": "salt", "level": "low"}),
    "/api/sketch": ("generator.onnx", {"style": 1}),
}


def copy_models(model_dir, target, names):
    for name in names:
        for suffix in (".onnx", ".json"):
            shutil.copy(model_dir / f"{name}{suffix}", target / f"{name}{suffix}")


def test_no_models(tmp_path, samples_dir):
    with make_client(tmp_path / "empty", samples_dir) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "degraded"
        assert not any(m["present"] or m["loaded"] for m in health["models"])
        assert health["workspaces"]["hard"]["missing"] == [
            "classifier.onnx", "specialist_salt.onnx", "specialist_blur.onnx", "specialist_occlusion.onnx"]
        for endpoint, (missing_file, fields) in REQUESTS.items():
            response = client.post(endpoint, data={"sample_id": PET_SAMPLE, **fields})
            assert response.status_code == 503
            assert missing_file in response.json()["detail"]
        # endpoints without models keep working
        assert client.post("/api/corrupt", data={"sample_id": PET_SAMPLE, "corruption": "blur",
                                                 "level": "low"}).status_code == 200
        assert client.get("/api/samples").status_code == 200


def test_hard_routing_needs_only_the_selected_specialist(tmp_path, model_dir, samples_dir):
    copy_models(model_dir, tmp_path, ["classifier", "specialist_blur"])
    with make_client(tmp_path, samples_dir) as client:
        grey = client.post("/api/restore/hard", files=upload(encode(solid((128, 128, 128)))))
        assert grey.status_code == 200 and grey.json()["selected_expert"] == "identity"
        green = client.post("/api/restore/hard", files=upload(encode(solid((30, 220, 30)))))
        assert green.status_code == 200 and green.json()["selected_expert"] == "specialist_blur"
        red = client.post("/api/restore/hard", files=upload(encode(solid((220, 30, 30)))))
        assert red.status_code == 503 and "specialist_salt.onnx" in red.json()["detail"]
        workspace = client.get("/api/health").json()["workspaces"]["hard"]
        assert workspace == {"ready": False, "missing": ["specialist_salt.onnx", "specialist_occlusion.onnx"]}


def test_contract_violations_are_reported(tmp_path, model_dir, samples_dir):
    shutil.copy(model_dir / "classifier.onnx", tmp_path / "udae.onnx")  # output is 'probs', not 'output'
    (tmp_path / "moe.onnx").write_bytes(b"this is not an ONNX model")
    copy_models(model_dir, tmp_path, ["generator"])
    (tmp_path / "generator.json").write_text("{not json")
    with make_client(tmp_path, samples_dir) as client:
        models = {m["name"]: m for m in client.get("/api/health").json()["models"]}
        assert models["udae"]["present"] and not models["udae"]["loaded"]
        assert "missing output 'output'" in models["udae"]["error"]
        assert models["udae"]["outputs"][0]["name"] == "probs"  # shows what the file contains
        assert not models["moe"]["loaded"] and "could not load" in models["moe"]["error"]
        assert models["generator"]["loaded"] and models["generator"]["metadata"] is None
        assert any("generator.json" in w for w in models["generator"]["warnings"])

        response = client.post("/api/restore/universal", data={"sample_id": PET_SAMPLE})
        assert response.status_code == 503 and "missing output 'output'" in response.json()["detail"]
        assert client.post("/api/restore/moe", data={"sample_id": PET_SAMPLE}).status_code == 503


def test_sidecar_nan_values_do_not_break_health(tmp_path, model_dir, samples_dir):
    copy_models(model_dir, tmp_path, ["udae"])
    card = json.loads((tmp_path / "udae.json").read_text())
    (tmp_path / "udae.json").write_text(json.dumps({**card, "test_psnr": float("nan")}))  # writes NaN
    with make_client(tmp_path, samples_dir) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["models"][0]["metadata"]["test_psnr"] is None


@pytest.mark.parametrize("value", ["abc", "-1", "0"])
def test_invalid_environment_values_fail_fast(value):
    from app.config import Settings
    with pytest.raises(ValueError, match="MAX_UPLOAD_MB"):
        Settings.from_env({"MAX_UPLOAD_MB": value})


def test_settings_from_environment(tmp_path):
    from app.config import Settings
    settings = Settings.from_env({"MODEL_DIR": str(tmp_path), "CORS_ORIGINS": "http://a.test, http://b.test",
                                  "MAX_UPLOAD_MB": "5"})
    assert settings.model_dir == tmp_path
    assert settings.cors_origins == ("http://a.test", "http://b.test")
    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert Settings.from_env({}) == Settings()
