// HTTP client for the FastAPI backend (docs/api.md).
// The app always calls the same origin under /api: the Vite dev server (development) or nginx
// (Docker) forwards those requests to the backend, so no CORS setup is needed.
import type {
  CorruptResponse,
  CorruptionType,
  HardResponse,
  HealthResponse,
  Level,
  MoEResponse,
  OptionsResponse,
  RoutingMode,
  SampleInfo,
  SketchResponse,
  UniversalResponse,
} from "./types";

/** Prefix for API paths; empty means same origin. Override with VITE_API_BASE at build time. */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

/** An error response from the backend, or a network failure (status 0). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** 503: a model file the request needs is missing or broken on the server. */
  get isModelUnavailable(): boolean {
    return this.status === 503;
  }
}

/** The message to show for a failed response. The backend always sends {"detail": "..."}; the
 * fallbacks cover responses that did not come from it (e.g. nginx while the backend restarts). */
export function errorDetail(status: number, body: unknown, statusText = ""): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string" && detail) return detail;
  }
  if (status === 413) return "The file is too large: images may be at most 10 MB.";
  if (status === 502 || status === 504) {
    return "The backend is not reachable. Check that it is running and try again.";
  }
  return `Request failed with status ${status}${statusText ? ` ${statusText}` : ""}.`;
}

export const NETWORK_ERROR =
  "Cannot reach the server. Check that the application is running and try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_BASE + path, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(0, NETWORK_ERROR);
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null; // not JSON, e.g. an HTML error page from a proxy
  }
  if (!response.ok) throw new ApiError(response.status, errorDetail(response.status, body, response.statusText));
  if (body === null) throw new ApiError(response.status, "The server returned an invalid response.");
  return body as T;
}

const post = <T>(path: string, form: FormData, signal?: AbortSignal) =>
  request<T>(path, { method: "POST", body: form, signal });

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/health", { signal }),
  options: () => request<OptionsResponse>("/api/options"),
  samples: (category: "pets" | "faces") =>
    request<{ samples: SampleInfo[] }>(`/api/samples?category=${category}`).then((r) => r.samples),
  corrupt: (form: FormData, signal?: AbortSignal) => post<CorruptResponse>("/api/corrupt", form, signal),
  universal: (form: FormData, signal?: AbortSignal) =>
    post<UniversalResponse>("/api/restore/universal", form, signal),
  hard: (form: FormData, signal?: AbortSignal) => post<HardResponse>("/api/restore/hard", form, signal),
  moe: (form: FormData, signal?: AbortSignal) => post<MoEResponse>("/api/restore/moe", form, signal),
  sketch: (form: FormData, signal?: AbortSignal) => post<SketchResponse>("/api/sketch", form, signal),
};

/** URL of a bundled sample image, for <img src>. */
export const sampleImageUrl = (sample: SampleInfo) => API_BASE + sample.url;

// --------------------------------------------------------------------------------------------
// Multipart request builders (pure functions, unit tested)
// --------------------------------------------------------------------------------------------

/** The image of a request: an uploaded file (or webcam frame) or a bundled sample. */
export type ImageSource = { kind: "upload"; file: File } | { kind: "sample"; id: string };

/** Corruption picked in the UI; "none" means no artificial corruption. */
export type CorruptionChoice = "none" | Exclude<CorruptionType, "clean">;

export interface CorruptionRequest {
  choice: CorruptionChoice;
  level: Level;
  /** Custom parameters of the chosen type (e.g. {k: 5, sigma: 1.2}); null to use `level`. */
  custom: Record<string, number> | null;
  /** Seed for reproducible corruptions; null lets the server pick one (it is always returned). */
  seed: number | null;
}

/**
 * The `corruption` form field for a request, or null to omit it.
 * - A corruption type is applied by the server to the (clean) image.
 * - "none" on a sample sends `clean`: the server then knows the true class, so oracle routing and
 *   PSNR against the reference are available.
 * - "none" on an upload omits the field: the image is restored as uploaded (for example an image
 *   that is already corrupted) and its true condition is unknown.
 */
export function corruptionField(source: ImageSource["kind"], choice: CorruptionChoice): CorruptionType | null {
  if (choice !== "none") return choice;
  return source === "sample" ? "clean" : null;
}

/** Oracle routing needs the true corruption, which is known only when the server applies it. */
export function oracleAvailable(source: ImageSource["kind"] | null, choice: CorruptionChoice): boolean {
  return choice !== "none" || source === "sample";
}

function appendImage(form: FormData, source: ImageSource): void {
  if (source.kind === "upload") form.append("file", source.file, source.file.name);
  else form.append("sample_id", source.id);
}

/** Form for /api/restore/{universal,hard,moe} and /api/corrupt; `routingMode` only for hard. */
export function buildRestoreForm(
  source: ImageSource,
  corruption: CorruptionRequest,
  routingMode?: RoutingMode,
): FormData {
  const form = new FormData();
  appendImage(form, source);
  const type = corruptionField(source.kind, corruption.choice);
  // "clean" takes no other corruption field; every other type needs a level or its parameters.
  if (type !== null) {
    form.append("corruption", type);
    if (type !== "clean") {
      if (corruption.custom) {
        for (const [name, value] of Object.entries(corruption.custom)) form.append(name, String(value));
      } else {
        form.append("level", corruption.level);
      }
      if (corruption.seed !== null) form.append("seed", String(corruption.seed));
    }
  }
  if (routingMode) form.append("routing_mode", routingMode);
  return form;
}

/** Form for /api/sketch. */
export function buildSketchForm(source: ImageSource, style: number): FormData {
  const form = new FormData();
  appendImage(form, source);
  form.append("style", String(style));
  return form;
}
