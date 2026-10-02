"""GET /api/health, /api/options, /api/samples and the OpenAPI document."""
import onnxruntime as ort

from conftest import FACE_SAMPLE, PET_SAMPLE

EXPECTED_FILES = ["udae.onnx", "classifier.onnx", "specialist_salt.onnx", "specialist_blur.onnx",
                  "specialist_occlusion.onnx", "moe.onnx", "generator.onnx"]


def test_health_reports_every_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runtime"]["onnxruntime"] == ort.__version__
    assert "CPUExecutionProvider" in body["runtime"]["available_providers"]
    assert body["uptime_s"] >= 0
    assert [m["file"] for m in body["models"]] == EXPECTED_FILES
    for model in body["models"]:
        assert model["present"] and model["loaded"] and model["error"] is None
        assert model["size_bytes"] > 0
        assert model["metadata"]["dummy"] is True  # the sidecar JSON
        assert model["inputs"][0]["shape"][0] == "batch"  # dynamic batch axis
    udae = body["models"][0]
    assert udae["inputs"] == [{"name": "input", "type": "tensor(float)", "shape": ["batch", 3, 128, 128]}]
    assert all(w["ready"] and w["missing"] == [] for w in body["workspaces"].values())
    assert body["samples"] == {"pets": 1, "faces": 1}


def test_options_come_from_the_training_code(client):
    body = client.get("/api/options").json()
    assert body["classes"] == ["clean", "salt", "blur", "occlusion"]
    by_type = {c["type"]: c for c in body["corruptions"]}
    assert by_type["salt"]["params"] == [{"name": "p", "min": 0.02, "max": 0.15, "choices": None}]
    assert by_type["blur"]["params"][0]["choices"] == [3, 5, 7]
    assert by_type["occlusion"]["levels"]["high"] == {"n": 3, "cover": 0.35}
    assert body["styles"] == [1, 2, 3] and body["max_upload_mb"] == 10


def test_samples_listing_and_files(client):
    samples = client.get("/api/samples").json()["samples"]
    assert {s["id"] for s in samples} == {PET_SAMPLE, FACE_SAMPLE}
    pets = client.get("/api/samples", params={"category": "pets"}).json()["samples"]
    assert [s["id"] for s in pets] == [PET_SAMPLE]
    image = client.get(pets[0]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert client.get("/api/samples/pets-missing").status_code == 404
    assert client.get("/api/samples", params={"category": "cars"}).status_code == 422


def test_openapi_and_root(client):
    paths = client.get("/api/openapi.json").json()["paths"]
    for path in ("/api/health", "/api/samples", "/api/corrupt", "/api/restore/universal",
                 "/api/restore/hard", "/api/restore/moe", "/api/sketch"):
        assert path in paths
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/api/docs"


def test_cors_headers(client):
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
