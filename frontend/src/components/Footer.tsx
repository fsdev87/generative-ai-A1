// Page footer (Stitch footer). The design's "API v2.4" and "Latency: ~420ms" were placeholders;
// the version and runtime shown here come from GET /api/health.
import { API_BASE } from "../api/client";
import type { HealthResponse } from "../api/types";

export function Footer({ health }: { health: HealthResponse | null }) {
  return (
    <footer className="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-md mt-space-xl">
      <div className="w-full px-gutter flex flex-col sm:flex-row items-center justify-between gap-space-sm text-on-surface-variant font-label-sm text-label-sm">
        <span>GenAI Studio • Generative AI Assignment 1</span>
        <div className="flex flex-wrap items-center justify-center gap-x-space-lg gap-y-space-xs">
          {health && <span>API v{health.version}</span>}
          {health && <span>ONNX Runtime {health.runtime.onnxruntime}</span>}
          <a className="hover:text-primary underline-offset-2 hover:underline" href={`${API_BASE}/api/docs`} target="_blank" rel="noreferrer">
            API docs
          </a>
        </div>
      </div>
    </footer>
  );
}
