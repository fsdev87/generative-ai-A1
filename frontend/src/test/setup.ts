// Vitest setup: jest-dom matchers, DOM cleanup between tests, and the browser APIs jsdom lacks.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import { clearSampleCache } from "../hooks/backend";

afterEach(() => {
  cleanup();
  clearSampleCache();
});

// jsdom does not implement object URLs; the upload preview only needs them to exist.
if (!URL.createObjectURL) {
  URL.createObjectURL = () => "blob:preview";
  URL.revokeObjectURL = () => undefined;
}
