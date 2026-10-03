// Workspace title row (Stitch "Universal Restoration Workbench" header): icon tile, title, task
// tag and description on the left; live chips on the right. The design's chips ("CUDA: 0 (RTX
// 4090)", "FP16 TensorRT", "v2.1") were invented; these show the real model files and runtime.
import type { ReactNode } from "react";
import type { WorkspaceKey } from "../api/types";
import { useBackend } from "../hooks/backend";
import { Icon } from "./Icon";

/** ONNX model names (GET /api/health `models[].name`) each workspace needs. */
export const WORKSPACE_MODELS: Record<WorkspaceKey, string[]> = {
  universal: ["udae"],
  hard: ["classifier", "specialist_salt", "specialist_blur", "specialist_occlusion"],
  moe: ["moe"],
  sketch: ["generator"],
};

function Chip({ icon, children, tone = "normal", title }: { icon: string; children: ReactNode; tone?: "normal" | "error"; title?: string }) {
  return (
    <div
      title={title}
      className={`flex items-center gap-space-xs bg-surface-container px-space-md py-1 rounded-md font-code-sm text-code-sm ${
        tone === "error" ? "text-error" : "text-on-surface-variant"
      }`}
    >
      <Icon name={icon} className={`text-[16px] ${tone === "error" ? "text-error" : "text-primary"}`} />
      <span>{children}</span>
    </div>
  );
}

export function WorkspaceHeader({
  workspace,
  icon,
  title,
  tag,
  subtitle,
  actions,
}: {
  workspace: WorkspaceKey;
  icon: string;
  title: string;
  tag: string;
  subtitle: string;
  actions?: ReactNode;
}) {
  const { health } = useBackend();
  const models = health ? health.models.filter((m) => WORKSPACE_MODELS[workspace].includes(m.name)) : [];
  const loaded = models.filter((m) => m.loaded).length;
  const modelTitle = models.map((m) => `${m.file}: ${m.loaded ? "loaded" : m.present ? "failed to load" : "missing"}`).join("\n");

  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-space-sm mb-space-lg">
      <div className="flex items-center gap-space-md min-w-0">
        <div className="w-9 h-9 rounded-lg bg-surface-container flex items-center justify-center text-primary shrink-0">
          <Icon name={icon} className="text-[20px]" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-space-xs">
            <h1 className="font-headline-sm text-headline-sm text-on-surface">{title}</h1>
            <span className="font-label-sm text-label-sm bg-primary-fixed text-on-primary-fixed px-space-xs py-0.5 rounded whitespace-nowrap">
              {tag}
            </span>
          </div>
          <p className="font-body-sm text-body-sm text-on-surface-variant">{subtitle}</p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-space-sm">
        {health && models.length === 1 && (
          <Chip icon={models[0].loaded ? "dataset" : "error"} tone={models[0].loaded ? "normal" : "error"} title={modelTitle}>
            {models[0].file} · {models[0].loaded ? "loaded" : "not loaded"}
          </Chip>
        )}
        {health && models.length > 1 && (
          <Chip icon={loaded === models.length ? "dataset" : "error"} tone={loaded === models.length ? "normal" : "error"} title={modelTitle}>
            {loaded}/{models.length} models loaded
          </Chip>
        )}
        {health && (
          <Chip icon="memory" title={`Available: ${health.runtime.available_providers.join(", ")}`}>
            ONNX Runtime {health.runtime.onnxruntime} · {health.runtime.providers.join(", ") || "no provider"}
          </Chip>
        )}
        {actions}
      </div>
    </div>
  );
}
