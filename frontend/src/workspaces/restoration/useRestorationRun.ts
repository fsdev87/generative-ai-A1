// Sends the restoration request of a workspace, or a corruption preview (POST /api/corrupt).
// The view is either the preview or the latest restoration result.
import { useEffect } from "react";
import { api, buildRestoreForm } from "../../api/client";
import type { CorruptResponse, RestorationResult, RoutingMode } from "../../api/types";
import { useBackend } from "../../hooks/backend";
import { useRequest } from "../../hooks/useRequest";
import type { RestorationForm } from "./useRestorationForm";

export type RestorationView<T> = { kind: "preview"; data: CorruptResponse } | { kind: "result"; data: T };

export function useRestorationRun<T extends RestorationResult>(
  form: RestorationForm,
  restore: (body: FormData, signal: AbortSignal) => Promise<T>,
) {
  const { options } = useBackend();
  const request = useRequest<RestorationView<T>>();
  const { reset } = request;
  const selected = form.image.selected;

  // A new image makes the previous result meaningless.
  useEffect(() => reset(), [selected, reset]);

  const run = (routingMode?: RoutingMode) => {
    const corruption = form.buildRequest(options);
    if (!selected || !corruption) return;
    const body = buildRestoreForm(selected.source, corruption, routingMode);
    void request.run((signal) => restore(body, signal).then((data) => ({ kind: "result" as const, data })));
  };

  const preview = () => {
    const corruption = form.buildRequest(options);
    if (!selected || !corruption || corruption.choice === "none") return;
    const body = buildRestoreForm(selected.source, corruption);
    void request.run((signal) =>
      api.corrupt(body, signal).then((data) => {
        // Keep the seed, so "Restore" next restores exactly this corrupted image.
        form.setSeedText(String(data.corruption.seed));
        return { kind: "preview" as const, data };
      }),
    );
  };

  const view = request.data;
  return {
    view,
    result: view?.kind === "result" ? view.data : null,
    loading: request.loading,
    error: request.error,
    run,
    preview,
  };
}
