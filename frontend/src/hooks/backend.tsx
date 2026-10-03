// Backend state shared by the whole app: the health check (polled), the UI options and the
// bundled samples. Workspaces read it with useBackend() / useSamples().
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, ApiError, NETWORK_ERROR } from "../api/client";
import type { HealthResponse, OptionsResponse, SampleInfo } from "../api/types";

export type BackendStatus = "checking" | "online" | "degraded" | "offline";

export interface BackendState {
  status: BackendStatus;
  /** Last successful health response; null while checking or when the backend is offline. */
  health: HealthResponse | null;
  healthError: string | null;
  /** Corruption types, levels, parameter ranges, styles and limits from GET /api/options. */
  options: OptionsResponse | null;
  refresh: () => void;
}

const BackendContext = createContext<BackendState | null>(null);

export function useBackend(): BackendState {
  const state = useContext(BackendContext);
  if (!state) throw new Error("useBackend must be used inside <BackendProvider>");
  return state;
}

const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError";
const messageOf = (error: unknown) => (error instanceof ApiError ? error.detail : NETWORK_ERROR);

export function BackendProvider({ children, pollMs = 10_000 }: { children: ReactNode; pollMs?: number }) {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);

  // Health: checked at start and every `pollMs`, so the pill follows backend restarts.
  useEffect(() => {
    let controller: AbortController | null = null;
    const check = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const result = await api.health(controller.signal);
        setHealth(result);
        setStatus(result.status === "ok" ? "online" : "degraded");
        setHealthError(null);
      } catch (error) {
        if (isAbort(error)) return;
        setHealth(null);
        setStatus("offline");
        setHealthError(messageOf(error));
      }
    };
    check();
    const timer = setInterval(check, pollMs);
    return () => {
      clearInterval(timer);
      controller?.abort();
    };
  }, [pollMs, refreshCount]);

  // Options: loaded once the backend answers, and again after a backend restart.
  const startedAt = health?.started_at;
  useEffect(() => {
    if (!startedAt) return;
    let active = true;
    api.options().then(
      (result) => active && setOptions(result),
      () => undefined, // keep the previous options; the health pill already shows the problem
    );
    return () => {
      active = false;
    };
  }, [startedAt]);

  const value: BackendState = {
    status,
    health,
    healthError,
    options,
    refresh: () => setRefreshCount((n) => n + 1),
  };
  return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>;
}

// Samples are cached per category and backend start, so the three restoration workspaces share
// one request. The backend scans its sample folder at startup only.
const sampleCache = new Map<string, Promise<SampleInfo[]>>();

/** Forget cached sample lists (used by the tests). */
export const clearSampleCache = () => sampleCache.clear();

export function useSamples(category: "pets" | "faces") {
  const { health, status } = useBackend();
  const startedAt = health?.started_at;
  const [samples, setSamples] = useState<SampleInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!startedAt) return;
    const key = `${category}@${startedAt}`;
    let request = sampleCache.get(key);
    if (!request) {
      request = api.samples(category);
      sampleCache.set(key, request);
      request.catch(() => sampleCache.delete(key)); // retry on the next health change
    }
    let active = true;
    request.then(
      (result) => {
        if (!active) return;
        setSamples(result);
        setError(null);
      },
      (err: unknown) => active && setError(messageOf(err)),
    );
    return () => {
      active = false;
    };
  }, [category, startedAt]);

  const offline = status === "offline" && samples === null;
  return { samples, error: error ?? (offline ? "the backend is offline." : null) };
}
