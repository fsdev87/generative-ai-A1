// Warning shown in a workspace whose model files are missing or failed to load (from
// GET /api/health), so the evaluator knows why a request returns 503 before trying.
import type { WorkspaceKey } from "../api/types";
import { useBackend } from "../hooks/backend";
import { Alert } from "./Alert";
import { WORKSPACE_MODELS } from "./WorkspaceHeader";

export function ModelStatusNotice({ workspace }: { workspace: WorkspaceKey }) {
  const { health } = useBackend();
  if (!health || health.workspaces[workspace].ready) return null;
  const problems = health.models.filter((m) => WORKSPACE_MODELS[workspace].includes(m.name) && !m.loaded);
  return (
    <Alert tone="warning" title="Model not available">
      <ul className="list-disc pl-4">
        {problems.map((m) => (
          <li key={m.name}>
            <span className="font-semibold">{m.file}</span>: {m.present ? (m.error ?? "failed to load") : `missing from ${health.model_dir}`}
          </li>
        ))}
      </ul>
      <p className="mt-1">
        Put the ONNX files (and their .json model cards) into <code>./models</code> and restart the backend.
        {workspace === "hard"
          ? " Requests routed to a missing specialist fail; routes whose models are loaded still work."
          : " Requests in this workspace fail until then."}
      </p>
    </Alert>
  );
}
