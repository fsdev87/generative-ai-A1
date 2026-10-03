// API responses shaped like docs/api.md, and a fetch mock that answers by URL.
import { vi } from "vitest";
import type {
  HardResponse,
  HealthResponse,
  ModelStatus,
  MoEResponse,
  OptionsResponse,
  SampleInfo,
  SketchResponse,
  UniversalResponse,
} from "../api/types";

export const PNG = "data:image/png;base64,iVBORw0KGgo=";

const model = (name: string, loaded = true): ModelStatus => ({
  name,
  file: `${name}.onnx`,
  description: `model ${name}`,
  present: loaded,
  loaded,
  size_bytes: loaded ? 1234 : null,
  load_ms: loaded ? 4.2 : null,
  inputs: [{ name: "input", type: "tensor(float)", shape: ["batch", 3, 128, 128] }],
  outputs: [{ name: "output", type: "tensor(float)", shape: ["batch", 3, 128, 128] }],
  metadata: loaded ? { task: "test", parity: { max_abs_diff: 3e-7 } } : null,
  error: null,
  warnings: [],
});

const NAMES = ["udae", "classifier", "specialist_salt", "specialist_blur", "specialist_occlusion", "moe", "generator"];

export function health(missing: string[] = []): HealthResponse {
  const ws = (names: string[]) => {
    const absent = names.filter((n) => missing.includes(n)).map((n) => `${n}.onnx`);
    return { ready: absent.length === 0, missing: absent };
  };
  return {
    status: missing.length ? "degraded" : "ok",
    version: "1.0.0",
    started_at: "2026-10-02T21:05:11.402Z",
    uptime_s: 84.2,
    runtime: {
      python: "3.11.9",
      onnxruntime: "1.30.0",
      available_providers: ["CPUExecutionProvider"],
      providers: ["CPUExecutionProvider"],
    },
    model_dir: "/models",
    models: NAMES.map((n) => model(n, !missing.includes(n))),
    workspaces: {
      universal: ws(["udae"]),
      hard: ws(["classifier", "specialist_salt", "specialist_blur", "specialist_occlusion"]),
      moe: ws(["moe"]),
      sketch: ws(["generator"]),
    },
    samples: { pets: 2, faces: 1 },
  };
}

export const options: OptionsResponse = {
  classes: ["clean", "salt", "blur", "occlusion"],
  levels: ["low", "medium", "high"],
  corruptions: [
    { type: "clean", params: [], levels: {} },
    {
      type: "salt",
      params: [{ name: "p", min: 0.02, max: 0.15, choices: null }],
      levels: { low: { p: 0.03 }, medium: { p: 0.08 }, high: { p: 0.15 } },
    },
    {
      type: "blur",
      params: [
        { name: "k", min: 3, max: 7, choices: [3, 5, 7] },
        { name: "sigma", min: 0.5, max: 2.5, choices: null },
      ],
      levels: { low: { k: 3, sigma: 0.7 }, medium: { k: 5, sigma: 1.5 }, high: { k: 7, sigma: 2.5 } },
    },
    {
      type: "occlusion",
      params: [
        { name: "n", min: 1, max: 3, choices: [1, 2, 3] },
        { name: "cover", min: 0.1, max: 0.35, choices: null },
      ],
      levels: { low: { n: 1, cover: 0.1 }, medium: { n: 2, cover: 0.2 }, high: { n: 3, cover: 0.35 } },
    },
  ],
  styles: [1, 2, 3],
  image_size: 128,
  max_upload_mb: 10,
  max_image_megapixels: 50,
};

export const pets: SampleInfo[] = [
  { id: "pets-abyssinian_12", category: "pets", name: "Abyssinian_12.jpg", url: "/api/samples/pets-abyssinian_12" },
  { id: "pets-beagle_3", category: "pets", name: "beagle_3.jpg", url: "/api/samples/pets-beagle_3" },
];

export const faces: SampleInfo[] = [
  { id: "faces-face_01", category: "faces", name: "face_01.jpg", url: "/api/samples/faces-face_01" },
];

const source = { kind: "sample" as const, name: "pets-abyssinian_12", format: "JPEG", width: 500, height: 375 };
const blur = {
  type: "blur" as const,
  level: "medium" as const,
  custom: false,
  seed: 7,
  params: { k: 5, sigma: 1.5 },
  rects: null,
  spec: { type: "blur", k: 5, sigma: 1.5, level: "medium" },
};

export const universalResult: UniversalResponse = {
  input: PNG,
  output: PNG,
  reference: PNG,
  corruption: blur,
  metrics: { psnr_input_db: 27.41, psnr_output_db: 30.02 },
  source,
  model: "udae.onnx",
  timing: { inference_ms: 6.2, total_ms: 18.3 },
};

export const hardResult: HardResponse = {
  input: PNG,
  output: PNG,
  reference: PNG,
  corruption: { ...blur, type: "salt", params: { p: 0.15 }, level: "high", spec: { type: "salt" } },
  metrics: { psnr_input_db: 12.98, psnr_output_db: 21.25 },
  source,
  routing_mode: "predicted",
  probabilities: { clean: 0.691, salt: 0.1627, blur: 0.0551, occlusion: 0.0912 },
  predicted_class: "clean",
  true_class: "salt",
  routed_class: "clean",
  selected_expert: "identity",
  timing: { inference_ms: 2.2, total_ms: 15.5, classifier_ms: 2.2, expert_ms: 0 },
};

export const moeResult: MoEResponse = {
  input: PNG,
  output: PNG,
  reference: null,
  corruption: null,
  metrics: null,
  source: { kind: "upload", name: "upload.jpg", format: "JPEG", width: 400, height: 300 },
  model: "moe.onnx",
  weights: { clean: 0.1528, salt: 0.2512, blur: 0.4837, occlusion: 0.1123 },
  dominant_branch: "blur",
  branch_outputs: { clean: PNG, salt: PNG, blur: PNG, occlusion: PNG },
  timing: { inference_ms: 11.9, total_ms: 22.9 },
};

export const sketchResult: SketchResponse = {
  model: "generator.onnx",
  style: 3,
  style_index: 2,
  photo: PNG,
  sketch: PNG,
  sketch_channels: 1,
  source: { kind: "sample", name: "faces-face_01", format: "JPEG", width: 250, height: 250 },
  timing: { inference_ms: 35.2, total_ms: 52.0 },
};

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

type Handler = (init?: RequestInit) => Response | Promise<Response>;

/**
 * Replaces fetch: GET /api/health, /api/options and /api/samples answer with the fixtures above
 * unless overridden; POST paths must be given. Returns the mock to inspect the calls.
 */
export function mockApi(handlers: Record<string, Handler> = {}) {
  const defaults: Record<string, Handler> = {
    "/api/health": () => json(health()),
    "/api/options": () => json(options),
    "/api/samples?category=pets": () => json({ samples: pets }),
    "/api/samples?category=faces": () => json({ samples: faces }),
  };
  const routes = { ...defaults, ...handlers };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const handler = routes[url];
    if (!handler) throw new Error(`Unexpected request: ${url}`);
    return handler(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The FormData sent in the last call to `path`. */
export function sentForm(fetchMock: ReturnType<typeof mockApi>, path: string): FormData {
  const call = fetchMock.mock.calls.filter(([url]) => String(url) === path).at(-1);
  if (!call) throw new Error(`No request to ${path}`);
  return call[1]?.body as FormData;
}
