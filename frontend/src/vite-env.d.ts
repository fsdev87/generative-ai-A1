/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional API origin, e.g. http://localhost:8000; empty = same origin (default). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
