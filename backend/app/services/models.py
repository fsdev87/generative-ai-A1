"""ONNX models: the contract from docs/CONVENTIONS.md, loading with self-checks, and inference.

Every model is loaded once at startup and checked against the contract: input/output names and
element types, then one test inference that checks the output shapes, finiteness and value
ranges. A missing or broken file never stops the app: /api/health reports it and the endpoints
that need the model answer 503 with the reason.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..errors import ApiError

log = logging.getLogger("uvicorn.error")  # configured by uvicorn, so messages reach the server log

PROVIDERS = ["CPUExecutionProvider"]
FLOAT, INT64 = "tensor(float)", "tensor(int64)"


@dataclass(frozen=True)
class TensorSpec:
    """Expected tensor. In `shape`, None is the batch axis and a tuple lists the allowed sizes."""

    dtype: str
    shape: tuple
    value_range: tuple[float, float]
    softmax: bool = False  # values sum to 1 over axis 1


@dataclass(frozen=True)
class ModelContract:
    name: str
    description: str
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]

    @property
    def file(self) -> str:
        return f"{self.name}.onnx"


IMAGE = TensorSpec(FLOAT, (None, 3, 128, 128), (0.0, 1.0))
PROBS = TensorSpec(FLOAT, (None, 4), (0.0, 1.0), softmax=True)


def _restorer(name, description):
    return ModelContract(name, description, {"input": IMAGE}, {"output": IMAGE})


CONTRACTS = {c.name: c for c in (
    _restorer("udae", "Task 1: universal denoising autoencoder"),
    ModelContract("classifier", "Task 2: corruption classifier", {"input": IMAGE}, {"probs": PROBS}),
    _restorer("specialist_salt", "Task 2: salt-and-pepper specialist"),
    _restorer("specialist_blur", "Task 2: Gaussian-blur specialist"),
    _restorer("specialist_occlusion", "Task 2: occlusion specialist"),
    ModelContract("moe", "Task 3: soft mixture of experts", {"input": IMAGE}, {
        "output": IMAGE,
        "weights": PROBS,
        "branch_outputs": TensorSpec(FLOAT, (None, 4, 3, 128, 128), (0.0, 1.0)),
    }),
    ModelContract("generator", "Task 4: face-to-sketch generator", {
        "photo": TensorSpec(FLOAT, (None, 3, 128, 128), (-1.0, 1.0)),
        "style": TensorSpec(INT64, (None,), (0, 2)),
    }, {"sketch": TensorSpec(FLOAT, (None, (1, 3), 128, 128), (-1.0, 1.0))}),
)}

# Models each workspace uses (hard routing runs the classifier and at most one specialist).
WORKSPACE_MODELS = {
    "universal": ("udae",),
    "hard": ("classifier", "specialist_salt", "specialist_blur", "specialist_occlusion"),
    "moe": ("moe",),
    "sketch": ("generator",),
}


@dataclass
class LoadedModel:
    contract: ModelContract
    path: Path
    size_bytes: int | None = None
    metadata: dict | None = None
    session: ort.InferenceSession | None = None
    load_ms: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return self.size_bytes is not None

    @property
    def ready(self) -> bool:
        return self.session is not None and self.error is None

    def run(self, feeds: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        """Run one batch: the outputs by name and the duration of session.run in ms."""
        names = list(self.contract.outputs)
        start = time.perf_counter()
        try:
            values = self.session.run(names, feeds)
        except Exception as exc:
            raise ApiError(500, f"{self.contract.file} failed during inference: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        outputs = dict(zip(names, values))
        batch = len(next(iter(feeds.values())))
        problems = _output_problems(self.contract, outputs, batch)
        if problems:
            raise ApiError(500, f"{self.contract.file} returned unexpected output: {'; '.join(problems)}")
        return outputs, elapsed_ms


class ModelRegistry:
    """All models of the ONNX contract, loaded from one directory at startup."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            log.warning("model directory %s does not exist", self.model_dir)
        self.models = {name: load_model(contract, self.model_dir) for name, contract in CONTRACTS.items()}
        for model in self.models.values():
            if model.ready:
                log.info("loaded %s (%.1f MB) in %.0f ms", model.contract.file, model.size_bytes / 1e6, model.load_ms)
            else:
                log.warning("%s is unavailable: %s", model.contract.file, model.error)
            for warning in model.warnings:
                log.warning("%s: %s", model.contract.file, warning)

    def get(self, name: str) -> LoadedModel:
        """A ready model, or a 503 error naming the file and the reason."""
        model = self.models[name]
        if model.ready:
            return model
        if not model.present:
            raise ApiError(503, f"Model file {model.contract.file} is missing from {self.model_dir}. "
                                "Add it and restart the backend.")
        raise ApiError(503, f"Model {model.contract.file} is unavailable: {model.error}")

    def missing(self, names) -> list[str]:
        """Files among the named models that are missing or failed to load."""
        return [self.models[name].contract.file for name in names if not self.models[name].ready]


def load_model(contract: ModelContract, model_dir: Path) -> LoadedModel:
    """Load one model and check it against its contract; problems are recorded, never raised."""
    model = LoadedModel(contract, model_dir / contract.file)
    if not model.path.is_file():
        model.error = f"{contract.file} not found in {model_dir}"
        return model
    model.size_bytes = model.path.stat().st_size
    model.metadata = _read_sidecar(model.path.with_suffix(".json"), model.warnings)
    start = time.perf_counter()
    try:
        model.session = ort.InferenceSession(str(model.path), providers=PROVIDERS)
    except Exception as exc:
        model.error = f"ONNX Runtime could not load {contract.file}: {exc}"
        return model
    problems = _signature_problems(contract, model.session, model.warnings) or _self_test(model)
    model.error = "; ".join(problems) or None
    model.load_ms = (time.perf_counter() - start) * 1000
    return model


def _read_sidecar(path: Path, warnings: list[str]) -> dict | None:
    """The model card next to the .onnx file. NaN/Infinity become null (they are not valid JSON)."""
    if not path.is_file():
        warnings.append(f"no sidecar {path.name}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_constant=lambda _: None)
    except (OSError, ValueError) as exc:
        warnings.append(f"sidecar {path.name} could not be read: {exc}")
        return None
    if not isinstance(data, dict):
        warnings.append(f"sidecar {path.name} is not a JSON object")
        return None
    return data


def _signature_problems(contract: ModelContract, session, warnings: list[str]) -> list[str]:
    """Input/output names and element types; a fixed batch axis is only a warning."""
    problems = []
    declared = {
        "input": {t.name: t for t in session.get_inputs()},
        "output": {t.name: t for t in session.get_outputs()},
    }
    for kind, expected in (("input", contract.inputs), ("output", contract.outputs)):
        for name, spec in expected.items():
            tensor = declared[kind].get(name)
            if tensor is None:
                problems.append(f"missing {kind} '{name}' (the model has {', '.join(declared[kind])})")
            elif tensor.type != spec.dtype:
                problems.append(f"{kind} '{name}' is {tensor.type}, expected {spec.dtype}")
            elif kind == "input" and tensor.shape and isinstance(tensor.shape[0], int):
                warnings.append(f"input '{name}' has a fixed batch size of {tensor.shape[0]}; "
                                "the contract asks for a dynamic batch axis")
    extra = [name for name in declared["input"] if name not in contract.inputs]
    if extra:
        problems.append(f"unexpected input(s) {', '.join(extra)}")
    return problems


def _self_test(model: LoadedModel) -> list[str]:
    """One inference on a synthetic batch of one (a ramp over the input range)."""
    feeds = {}
    for name, spec in model.contract.inputs.items():
        shape = tuple(1 if size is None else size for size in spec.shape)
        low, high = spec.value_range
        if spec.dtype == INT64:
            feeds[name] = np.full(shape, int(low), dtype=np.int64)
        else:
            feeds[name] = np.linspace(low, high, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    try:
        outputs, _ = model.run(feeds)
    except ApiError as exc:
        return [f"test inference failed: {exc.detail}"]
    model.warnings.extend(_range_warnings(model.contract, outputs))
    return []


def _output_problems(contract: ModelContract, outputs: dict, batch: int) -> list[str]:
    problems = []
    for name, spec in contract.outputs.items():
        value = outputs[name]
        if not _shape_matches(value.shape, spec.shape, batch):
            problems.append(f"'{name}' has shape {list(value.shape)}, expected {_format_shape(spec.shape)}")
        elif not np.isfinite(value).all():
            problems.append(f"'{name}' contains NaN or infinite values")
    return problems


def _range_warnings(contract: ModelContract, outputs: dict) -> list[str]:
    warnings = []
    for name, spec in contract.outputs.items():
        value = outputs[name]
        low, high = spec.value_range
        if value.min() < low - 1e-4 or value.max() > high + 1e-4:
            warnings.append(f"output '{name}' spans [{value.min():.4g}, {value.max():.4g}] on the test input, "
                            f"expected [{low:g}, {high:g}]")
        if spec.softmax and not np.allclose(value.sum(axis=1), 1.0, atol=1e-3):
            warnings.append(f"output '{name}' does not sum to 1 (softmax expected)")
    return warnings


def _allowed_sizes(axis, batch: int) -> tuple:
    if axis is None:
        return (batch,)
    return axis if isinstance(axis, tuple) else (axis,)


def _shape_matches(shape, expected, batch: int) -> bool:
    return len(shape) == len(expected) and all(
        size in _allowed_sizes(axis, batch) for size, axis in zip(shape, expected))


def _format_shape(shape) -> str:
    def axis_text(axis):
        if axis is None:
            return "N"
        return " or ".join(map(str, axis)) if isinstance(axis, tuple) else str(axis)

    return "[" + ", ".join(axis_text(axis) for axis in shape) + "]"
