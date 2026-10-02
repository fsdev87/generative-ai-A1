"""Response models of every endpoint. They define the OpenAPI schema (/api/docs, /api/openapi.json)
that the frontend is built from; docs/api.md describes the same API with examples."""
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

CorruptionType = Literal["clean", "salt", "blur", "occlusion"]
Level = Literal["low", "medium", "high"]
RoutingMode = Literal["predicted", "oracle"]
Expert = Literal["identity", "specialist_salt", "specialist_blur", "specialist_occlusion"]
DataUrl = Annotated[str, Field(description="PNG image as a base64 data URL (data:image/png;base64,...)")]


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message")
    errors: list[dict[str, Any]] | None = Field(None, description="Per-field errors (422 only)")


def error_responses(*codes: int) -> dict:
    """OpenAPI entries for the error status codes an endpoint can return."""
    return {code: {"model": ErrorResponse} for code in codes}


# --------------------------------------------------------------------------- #
# Shared parts
# --------------------------------------------------------------------------- #
class SourceInfo(BaseModel):
    kind: Literal["upload", "sample"]
    name: str = Field(description="Uploaded file name or sample id")
    format: str = Field(description="Decoded file format: JPEG, PNG, WEBP or BMP")
    width: int = Field(description="Width of the original image (after EXIF rotation)")
    height: int = Field(description="Height of the original image (after EXIF rotation)")


class CorruptionSettings(BaseModel):
    type: CorruptionType
    level: Level | None = Field(
        description="The fixed test level, or the severity level that the custom parameters fall into; "
                    "null for clean")
    custom: bool = Field(description="True when custom parameters were given instead of a level")
    seed: int = Field(description="Request seed; sending it again with the same fields reproduces the corruption")
    params: dict[str, float | int] = Field(
        description="salt: p; blur: k, sigma; occlusion: n, cover (the cover actually achieved); clean: {}")
    rects: list[list[int]] | None = Field(
        description="Occlusion rectangles as [y, x, height, width] in the 128x128 image; null otherwise")
    spec: dict[str, Any] = Field(
        description="The exact spec given to src.data.corruptions.apply_spec (same format as the manifests)")


class Metrics(BaseModel):
    """Present only when the server applied the corruption, so the clean reference is known."""

    psnr_input_db: float = Field(description="PSNR of the model input against the clean reference (max 100)")
    psnr_output_db: float = Field(description="PSNR of the output against the clean reference (max 100)")


class Timing(BaseModel):
    inference_ms: float = Field(description="Time spent in ONNX Runtime (session.run of every model used)")
    total_ms: float = Field(description="Server time from reading the image to the finished response")


class HardTiming(Timing):
    classifier_ms: float
    expert_ms: float = Field(description="0 when the identity bypass was used")


class ClassScores(BaseModel):
    """One value per class in CLASSES order; clean is the identity branch."""

    clean: float
    salt: float
    blur: float
    occlusion: float


class BranchImages(BaseModel):
    """Output of every mixture branch; clean is the identity branch (the input itself)."""

    clean: DataUrl
    salt: DataUrl
    blur: DataUrl
    occlusion: DataUrl


# --------------------------------------------------------------------------- #
# Corruption, restoration and sketch responses
# --------------------------------------------------------------------------- #
class CorruptResponse(BaseModel):
    image: DataUrl = Field(description="The corrupted 128x128 image")
    clean: DataUrl = Field(description="The preprocessed clean 128x128 image before corruption")
    corruption: CorruptionSettings
    source: SourceInfo


class RestorationResult(BaseModel):
    input: DataUrl = Field(description="The 128x128 image the model received")
    output: DataUrl = Field(description="The restored 128x128 image")
    reference: DataUrl | None = Field(
        description="The clean 128x128 image before the server-side corruption; null for an image used as uploaded")
    corruption: CorruptionSettings | None = Field(description="The server-side corruption, if one was applied")
    metrics: Metrics | None
    source: SourceInfo


class UniversalResponse(RestorationResult):
    model: str
    timing: Timing


class HardResponse(RestorationResult):
    routing_mode: RoutingMode
    probabilities: ClassScores = Field(description="Classifier softmax output")
    predicted_class: CorruptionType = Field(description="argmax of the probabilities")
    true_class: CorruptionType | None = Field(description="Known only when the server applied the corruption")
    routed_class: CorruptionType = Field(
        description="Class that chose the expert: the prediction, or the true class in oracle mode")
    selected_expert: Expert = Field(description="identity (clean bypass, no expert runs) or the specialist model")
    timing: HardTiming


class MoEResponse(RestorationResult):
    model: str
    weights: ClassScores = Field(description="Gate routing weights (softmax)")
    dominant_branch: CorruptionType = Field(description="Branch with the largest weight")
    branch_outputs: BranchImages
    timing: Timing


class SketchResponse(BaseModel):
    model: str
    style: int = Field(description="Style as shown to users: 1, 2 or 3")
    style_index: int = Field(description="Index given to the generator: style - 1")
    photo: DataUrl = Field(description="The preprocessed 128x128 photo the generator received")
    sketch: DataUrl = Field(description="The generated 128x128 sketch (greyscale PNG for a one-channel generator)")
    sketch_channels: int = Field(description="Channels of the generator output: 1 or 3")
    source: SourceInfo
    timing: Timing


# --------------------------------------------------------------------------- #
# Health, samples and options
# --------------------------------------------------------------------------- #
class TensorInfo(BaseModel):
    name: str
    type: str = Field(description="ONNX element type, e.g. tensor(float)")
    shape: list[int | str | None] = Field(description="Symbolic (dynamic) axes are strings, e.g. 'batch'")


class ModelStatus(BaseModel):
    name: str
    file: str
    description: str
    present: bool = Field(description="The file exists in MODEL_DIR")
    loaded: bool = Field(description="Loaded and passed the contract checks; usable by the endpoints")
    size_bytes: int | None
    load_ms: float | None
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    metadata: dict[str, Any] | None = Field(description="The sidecar <name>.json (model card), if present")
    error: str | None = Field(description="Why the model is not usable")
    warnings: list[str]


class WorkspaceStatus(BaseModel):
    ready: bool
    missing: list[str] = Field(description="Model files of this workspace that are missing or failed to load")


class Workspaces(BaseModel):
    universal: WorkspaceStatus
    hard: WorkspaceStatus
    moe: WorkspaceStatus
    sketch: WorkspaceStatus


class RuntimeInfo(BaseModel):
    python: str
    onnxruntime: str
    available_providers: list[str]
    providers: list[str] = Field(description="Execution providers the models use")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(description="degraded when any model is missing or broken")
    version: str
    started_at: datetime
    uptime_s: float
    runtime: RuntimeInfo
    model_dir: str
    models: list[ModelStatus]
    workspaces: Workspaces
    samples: dict[str, int] = Field(description="Number of bundled samples per category")


class SampleInfo(BaseModel):
    id: str
    category: Literal["pets", "faces"]
    name: str = Field(description="File name")
    url: str = Field(description="Path of the image file: GET it to display the sample")


class SampleList(BaseModel):
    samples: list[SampleInfo]


class ParamOption(BaseModel):
    name: str
    min: float
    max: float
    choices: list[int] | None = Field(None, description="Allowed values of an integer parameter")


class CorruptionOption(BaseModel):
    type: CorruptionType
    params: list[ParamOption] = Field(description="Custom parameters with their training ranges")
    levels: dict[str, dict[str, float | int]] = Field(description="Parameters of the fixed test levels")


class OptionsResponse(BaseModel):
    classes: list[CorruptionType] = Field(description="Class order of probabilities and weights")
    levels: list[Level]
    corruptions: list[CorruptionOption]
    styles: list[int]
    image_size: int
    max_upload_mb: float
    max_image_megapixels: float
