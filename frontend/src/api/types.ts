// TypeScript mirror of the backend response models (backend/app/schemas.py, docs/api.md).
// Keep the two in sync: every field here is read from the API, nothing is invented.

/** Class order used everywhere: clean (identity branch), salt, blur, occlusion. */
export const CLASSES = ["clean", "salt", "blur", "occlusion"] as const;
export type CorruptionType = (typeof CLASSES)[number];
export type Level = "low" | "medium" | "high";
export type RoutingMode = "predicted" | "oracle";
export type Expert = "identity" | "specialist_salt" | "specialist_blur" | "specialist_occlusion";
/** PNG image as a base64 data URL (data:image/png;base64,...). */
export type DataUrl = string;

export type ClassScores = Record<CorruptionType, number>;

export interface SourceInfo {
  kind: "upload" | "sample";
  name: string;
  format: string;
  width: number;
  height: number;
}

export interface CorruptionSettings {
  type: CorruptionType;
  level: Level | null;
  custom: boolean;
  seed: number;
  params: Record<string, number>;
  rects: number[][] | null;
  spec: Record<string, unknown>;
}

export interface Metrics {
  psnr_input_db: number;
  psnr_output_db: number;
}

export interface Timing {
  inference_ms: number;
  total_ms: number;
}

export interface HardTiming extends Timing {
  classifier_ms: number;
  expert_ms: number;
}

export interface CorruptResponse {
  image: DataUrl;
  clean: DataUrl;
  corruption: CorruptionSettings;
  source: SourceInfo;
}

export interface RestorationResult {
  input: DataUrl;
  output: DataUrl;
  reference: DataUrl | null;
  corruption: CorruptionSettings | null;
  metrics: Metrics | null;
  source: SourceInfo;
}

export interface UniversalResponse extends RestorationResult {
  model: string;
  timing: Timing;
}

export interface HardResponse extends RestorationResult {
  routing_mode: RoutingMode;
  probabilities: ClassScores;
  predicted_class: CorruptionType;
  true_class: CorruptionType | null;
  routed_class: CorruptionType;
  selected_expert: Expert;
  timing: HardTiming;
}

export interface MoEResponse extends RestorationResult {
  model: string;
  weights: ClassScores;
  dominant_branch: CorruptionType;
  branch_outputs: Record<CorruptionType, DataUrl>;
  timing: Timing;
}

export interface SketchResponse {
  model: string;
  style: number;
  style_index: number;
  photo: DataUrl;
  sketch: DataUrl;
  sketch_channels: number;
  source: SourceInfo;
  timing: Timing;
}

export interface TensorInfo {
  name: string;
  type: string;
  shape: (number | string | null)[];
}

export interface ModelStatus {
  name: string;
  file: string;
  description: string;
  present: boolean;
  loaded: boolean;
  size_bytes: number | null;
  load_ms: number | null;
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  metadata: Record<string, unknown> | null;
  error: string | null;
  warnings: string[];
}

export interface WorkspaceStatus {
  ready: boolean;
  missing: string[];
}

export type WorkspaceKey = "universal" | "hard" | "moe" | "sketch";

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  started_at: string;
  uptime_s: number;
  runtime: {
    python: string;
    onnxruntime: string;
    available_providers: string[];
    providers: string[];
  };
  model_dir: string;
  models: ModelStatus[];
  workspaces: Record<WorkspaceKey, WorkspaceStatus>;
  samples: Record<string, number>;
}

export interface SampleInfo {
  id: string;
  category: "pets" | "faces";
  name: string;
  url: string;
}

export interface ParamOption {
  name: string;
  min: number;
  max: number;
  choices: number[] | null;
}

export interface CorruptionOption {
  type: CorruptionType;
  params: ParamOption[];
  levels: Record<string, Record<string, number>>;
}

export interface OptionsResponse {
  classes: CorruptionType[];
  levels: Level[];
  corruptions: CorruptionOption[];
  styles: number[];
  image_size: number;
  max_upload_mb: number;
  max_image_megapixels: number;
}
