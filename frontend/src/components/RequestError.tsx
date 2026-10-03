// Shows a failed request with the backend's `detail` message as is (docs/api.md, "Errors").
import type { ApiError } from "../api/client";
import { Alert } from "./Alert";

const TITLES: Record<number, string> = {
  0: "Server not reachable",
  400: "Invalid request",
  404: "Not found",
  413: "File too large",
  415: "Unsupported image",
  422: "Invalid value",
  500: "Inference failed",
  502: "Backend not reachable",
  503: "Model not available",
  504: "Backend not reachable",
};

export function RequestError({ error }: { error: ApiError | null }) {
  if (!error) return null;
  const title = TITLES[error.status] ?? "Request failed";
  return (
    <Alert tone="error" title={error.status ? `${title} (${error.status})` : title}>
      {error.detail}
    </Alert>
  );
}
