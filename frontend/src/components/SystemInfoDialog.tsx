// "System information" dialog: everything GET /api/health reports (backend status, runtime,
// workspace readiness, every ONNX model with its tensors and its model card from the sidecar JSON).
import { useEffect, useRef, type ReactNode } from "react";
import type { ModelStatus, TensorInfo } from "../api/types";
import { useBackend } from "../hooks/backend";
import { formatBytes, formatDuration, formatMs } from "../lib/format";
import { WORKSPACES } from "../lib/routes";
import { Alert } from "./Alert";
import { Icon } from "./Icon";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[10rem_minmax(0,1fr)] gap-space-sm py-1 border-b border-surface-container last:border-0">
      <dt className="font-label-md text-label-md text-on-surface-variant">{label}</dt>
      <dd className="font-code-sm text-code-sm text-on-surface break-words">{children}</dd>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-space-xs">
      <h3 className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">{title}</h3>
      {children}
    </section>
  );
}

const tensors = (list: TensorInfo[]) =>
  list.map((t) => `${t.name} ${t.type} [${t.shape.map((d) => d ?? "?").join(", ")}]`).join("; ") || "-";

/** The model card: scalar fields as rows, nested objects as formatted JSON. */
function Metadata({ metadata }: { metadata: Record<string, unknown> }) {
  return (
    <dl>
      {Object.entries(metadata).map(([key, value]) => (
        <Row key={key} label={key}>
          {value !== null && typeof value === "object" ? (
            <pre className="whitespace-pre-wrap text-[11px] leading-4">{JSON.stringify(value, null, 2)}</pre>
          ) : typeof value === "string" && /^https?:\/\//.test(value) ? (
            <a className="text-primary underline" href={value} target="_blank" rel="noreferrer">
              {value}
            </a>
          ) : (
            String(value)
          )}
        </Row>
      ))}
    </dl>
  );
}

function ModelEntry({ model }: { model: ModelStatus }) {
  const state = model.loaded ? "loaded" : model.present ? "failed to load" : "missing";
  return (
    <details className="rounded-lg bg-surface-container-low px-space-md py-space-sm">
      <summary className="cursor-pointer flex items-center gap-space-sm">
        <Icon
          name={model.loaded ? "check_circle" : "error"}
          className={`text-[18px] ${model.loaded ? "text-emerald-600" : "text-error"}`}
        />
        <span className="font-label-md text-label-md text-on-surface">{model.file}</span>
        <span className="font-body-sm text-body-sm text-on-surface-variant truncate">{model.description}</span>
        <span className={`ml-auto font-label-sm text-label-sm ${model.loaded ? "text-emerald-700" : "text-error"}`}>
          {state}
        </span>
      </summary>
      <dl className="mt-space-sm">
        {model.error && <Row label="Error">{model.error}</Row>}
        {model.warnings.length > 0 && <Row label="Warnings">{model.warnings.join("; ")}</Row>}
        {model.size_bytes !== null && <Row label="File size">{formatBytes(model.size_bytes)}</Row>}
        {model.load_ms !== null && <Row label="Load time">{formatMs(model.load_ms)}</Row>}
        <Row label="Inputs">{tensors(model.inputs)}</Row>
        <Row label="Outputs">{tensors(model.outputs)}</Row>
      </dl>
      <div className="mt-space-sm">
        <p className="font-label-md text-label-md text-on-surface mb-1">Model card ({model.name}.json)</p>
        {model.metadata ? (
          <Metadata metadata={model.metadata} />
        ) : (
          <p className="font-body-sm text-body-sm text-on-surface-variant">No model card found next to the model.</p>
        )}
      </div>
    </details>
  );
}

export function SystemInfoDialog({ onClose }: { onClose: () => void }) {
  const { status, health, healthError, refresh } = useBackend();
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-inverse-surface/40 backdrop-blur-sm flex items-start justify-center p-space-md overflow-y-auto"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="system-info-title"
        className="w-full max-w-3xl my-space-xl bg-surface-container-lowest rounded-xl shadow-lg p-space-lg flex flex-col gap-space-lg"
      >
        <div className="flex items-center justify-between gap-space-md">
          <div className="flex items-center gap-space-sm">
            <Icon name="monitor_heart" className="text-[20px] text-primary" />
            <h2 id="system-info-title" className="font-headline-md text-headline-md text-on-surface">
              System information
            </h2>
          </div>
          <div className="flex items-center gap-space-xs">
            <button
              type="button"
              onClick={refresh}
              className="py-1.5 px-space-md rounded-lg bg-surface-container-low hover:bg-surface-container text-on-surface font-label-md text-label-md flex items-center gap-1"
            >
              <Icon name="refresh" className="text-[16px]" /> Refresh
            </button>
            <button
              ref={closeButton}
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="w-8 h-8 rounded-lg hover:bg-surface-container-low flex items-center justify-center text-on-surface-variant"
            >
              <Icon name="close" className="text-[20px]" />
            </button>
          </div>
        </div>

        {status === "offline" && (
          <Alert tone="error" title="Backend offline">
            {healthError}
          </Alert>
        )}
        {status === "checking" && <p className="font-body-md text-body-md">Checking the backend…</p>}

        {health && (
          <>
            <Section title="Backend">
              <dl>
                <Row label="Status">{health.status === "ok" ? "ok (all models loaded)" : "degraded (some models are not loaded)"}</Row>
                <Row label="API version">{health.version}</Row>
                <Row label="Started">{new Date(health.started_at).toLocaleString()}</Row>
                <Row label="Uptime">{formatDuration(health.uptime_s)}</Row>
                <Row label="Model folder">{health.model_dir}</Row>
                <Row label="Bundled samples">
                  {Object.entries(health.samples).map(([k, v]) => `${k}: ${v}`).join(", ") || "none"}
                </Row>
              </dl>
            </Section>
            <Section title="Runtime">
              <dl>
                <Row label="Python">{health.runtime.python}</Row>
                <Row label="ONNX Runtime">{health.runtime.onnxruntime}</Row>
                <Row label="Providers in use">{health.runtime.providers.join(", ") || "-"}</Row>
                <Row label="Available providers">{health.runtime.available_providers.join(", ") || "-"}</Row>
              </dl>
            </Section>
            <Section title="Workspaces">
              <dl>
                {WORKSPACES.map((w) => {
                  const ws = health.workspaces[w.key];
                  return (
                    <Row key={w.key} label={w.title}>
                      {ws.ready ? "ready" : `not ready, missing: ${ws.missing.join(", ")}`}
                    </Row>
                  );
                })}
              </dl>
            </Section>
            <Section title="Models">
              <div className="flex flex-col gap-space-xs">
                {health.models.map((model) => (
                  <ModelEntry key={model.name} model={model} />
                ))}
              </div>
            </Section>
          </>
        )}
      </div>
    </div>
  );
}
