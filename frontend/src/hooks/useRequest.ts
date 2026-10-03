// Runs one API request at a time for a workspace: loading flag, result, error, and cancellation
// of a previous request that is still running (only the latest answer is shown).
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, NETWORK_ERROR } from "../api/client";

export interface RequestState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

export function useRequest<T>() {
  const [state, setState] = useState<RequestState<T>>({ data: null, error: null, loading: false });
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const run = useCallback(async (call: (signal: AbortSignal) => Promise<T>) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await call(current.signal);
      if (!current.signal.aborted) setState({ data, error: null, loading: false });
    } catch (error) {
      if (current.signal.aborted) return;
      // Clear the old result so it is never mistaken for the answer to this request.
      const apiError = error instanceof ApiError ? error : new ApiError(0, NETWORK_ERROR);
      setState({ data: null, error: apiError, loading: false });
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    setState({ data: null, error: null, loading: false });
  }, []);

  return { ...state, run, reset };
}
