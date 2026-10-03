// Behaviour of the workspaces against a mocked backend (fetch is replaced, see test/fixtures.ts).
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { BackendProvider } from "../hooks/backend";
import {
  health,
  hardResult,
  json,
  mockApi,
  moeResult,
  PNG,
  sentForm,
  sketchResult,
  universalResult,
} from "../test/fixtures";
import { HardRoutedWorkspace } from "./HardRoutedWorkspace";
import { MoEWorkspace } from "./MoEWorkspace";
import { SketchWorkspace } from "./SketchWorkspace";
import { UniversalWorkspace } from "./UniversalWorkspace";

const renderWithBackend = (ui: ReactNode) => render(<BackendProvider pollMs={60_000}>{ui}</BackendProvider>);
const photo = () => new File(["jpeg bytes"], "photo.jpg", { type: "image/jpeg" });
const upload = (file: File) => fireEvent.change(screen.getByLabelText("Upload image"), { target: { files: [file] } });
const radio = (name: string) => screen.getByRole("radio", { name });
/** jsdom has neither a secure-context flag we can rely on nor navigator.mediaDevices. */
function setCamera(secure: boolean, getUserMedia?: () => Promise<MediaStream>) {
  Object.defineProperty(window, "isSecureContext", { configurable: true, get: () => secure });
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: getUserMedia ? { getUserMedia } : undefined,
  });
}
const pickSample = async (name = "Abyssinian_12.jpg") =>
  fireEvent.click(await screen.findByRole("button", { name: `Sample ${name}` }));

describe("Universal Restoration", () => {
  it("restores a sample and shows the settings, timing and download", async () => {
    const fetchMock = mockApi({ "/api/restore/universal": () => json(universalResult) });
    renderWithBackend(<UniversalWorkspace />);
    await pickSample();
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    const form = sentForm(fetchMock, "/api/restore/universal");
    expect(Object.fromEntries(form)).toEqual({ sample_id: "pets-abyssinian_12", corruption: "blur", level: "medium", seed: "7" });

    expect(await screen.findByText("Gaussian blur · kernel 5 · σ 1.5")).toBeInTheDocument();
    expect(screen.getByText("udae.onnx")).toBeInTheDocument();
    expect(screen.getByText("6.20 ms")).toBeInTheDocument();
    expect(screen.getByText("PSNR: 30.02 dB")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Clean original image before the corruption" })).toHaveAttribute("src", PNG);
    const download = screen.getByRole("link", { name: "Download result" });
    expect(download).toHaveAttribute("download", "pets-abyssinian_12_restored-universal.png");
    expect(download).toHaveAttribute("href", PNG);
  });

  it("previews the corruption and keeps its seed for the restore", async () => {
    const fetchMock = mockApi({
      "/api/corrupt": () =>
        json({ image: PNG, clean: PNG, corruption: { ...universalResult.corruption, seed: 1234 }, source: universalResult.source }),
    });
    renderWithBackend(<UniversalWorkspace />);
    await pickSample();
    fireEvent.click(screen.getByRole("button", { name: "Preview corruption" }));
    await waitFor(() => expect(screen.getByLabelText("Seed")).toHaveValue("1234"));
    expect(sentForm(fetchMock, "/api/corrupt").get("corruption")).toBe("blur");
    expect(screen.getByText("Press Restore to run the model on this input.")).toBeInTheDocument();
  });

  it("shows the backend's 503 message and the missing model", async () => {
    const detail = "Model file udae.onnx is missing from /models. Add it and restart the backend.";
    mockApi({
      "/api/health": () => json(health(["udae"])),
      "/api/restore/universal": () => json({ detail }, 503),
    });
    renderWithBackend(<UniversalWorkspace />);
    expect(await screen.findByText("Model not available")).toBeInTheDocument();
    await pickSample();
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Model not available (503)");
    expect(alert).toHaveTextContent(detail);
  });

  it("shows a rejected upload's 415 detail", async () => {
    const detail = "The file is not a JPEG, PNG, WEBP or BMP image.";
    const fetchMock = mockApi({ "/api/restore/universal": () => json({ detail }, 415) });
    renderWithBackend(<UniversalWorkspace />);
    upload(new File(["hello"], "notes.txt", { type: "text/plain" }));
    fireEvent.click(radio("None"));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(`Unsupported image (415)${detail}`);
    // An upload with "None" is restored as is: no corruption field.
    expect([...sentForm(fetchMock, "/api/restore/universal").keys()]).toEqual(["file"]);
  });

  it("rejects a file above the upload limit before sending it", async () => {
    const fetchMock = mockApi();
    renderWithBackend(<UniversalWorkspace />);
    await screen.findByText(/up to 10 MB/);
    const big = photo();
    Object.defineProperty(big, "size", { value: 11 * 1024 * 1024 });
    upload(big);
    expect(screen.getByRole("alert")).toHaveTextContent("photo.jpg is 11.0 MB; images may be at most 10 MB.");
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith("/api/restore"))).toBe(false);
  });

  it("sends custom parameters when enabled", async () => {
    const fetchMock = mockApi({ "/api/restore/universal": () => json(universalResult) });
    renderWithBackend(<UniversalWorkspace />);
    await pickSample();
    fireEvent.click(radio("Salt & Pepper"));
    fireEvent.click(screen.getByLabelText("Custom parameters"));
    fireEvent.change(screen.getByLabelText("Noise probability p"), { target: { value: "0.1" } });
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(Object.fromEntries(sentForm(fetchMock, "/api/restore/universal"))).toEqual({
      sample_id: "pets-abyssinian_12",
      corruption: "salt",
      p: "0.1",
    });
  });
});

describe("Hard-Routed Restoration", () => {
  it("allows oracle routing only when the server applies the corruption", async () => {
    mockApi();
    renderWithBackend(<HardRoutedWorkspace />);
    await screen.findByRole("button", { name: "Sample Abyssinian_12.jpg" });
    expect(radio("Oracle")).toBeEnabled(); // Blur selected by default

    fireEvent.click(radio("None"));
    expect(radio("Oracle")).toBeDisabled(); // no image yet
    expect(screen.getByText(/Oracle routing needs the true corruption/)).toBeInTheDocument();

    await pickSample();
    expect(radio("Oracle")).toBeEnabled(); // a sample with None is sent as clean

    upload(photo());
    expect(radio("Oracle")).toBeDisabled(); // an upload used as is has no known class

    fireEvent.click(radio("Salt & Pepper"));
    expect(radio("Oracle")).toBeEnabled();
  });

  it("falls back to predicted routing when oracle becomes invalid", async () => {
    const fetchMock = mockApi({ "/api/restore/hard": () => json(hardResult) });
    renderWithBackend(<HardRoutedWorkspace />);
    upload(photo());
    fireEvent.click(radio("Oracle"));
    expect(radio("Oracle")).toHaveAttribute("aria-checked", "true");
    fireEvent.click(radio("None"));
    expect(radio("Predicted")).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(Object.fromEntries(sentForm(fetchMock, "/api/restore/hard"))).toMatchObject({ routing_mode: "predicted" });
    expect(sentForm(fetchMock, "/api/restore/hard").has("corruption")).toBe(false);
  });

  it("sends oracle routing and shows probabilities, prediction and expert", async () => {
    const fetchMock = mockApi({ "/api/restore/hard": () => json(hardResult) });
    renderWithBackend(<HardRoutedWorkspace />);
    await pickSample();
    fireEvent.click(radio("Salt & Pepper"));
    fireEvent.click(radio("High"));
    fireEvent.click(radio("Oracle"));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(Object.fromEntries(sentForm(fetchMock, "/api/restore/hard"))).toEqual({
      sample_id: "pets-abyssinian_12",
      corruption: "salt",
      level: "high",
      routing_mode: "oracle",
    });

    const bars = await screen.findByRole("group", { name: "Classifier probabilities" });
    expect(within(bars).getAllByRole("progressbar")).toHaveLength(4);
    expect(within(bars).getByRole("progressbar", { name: "Clean" })).toHaveAttribute("aria-valuetext", "69.1%");
    expect(screen.getByText(/Predicted:/)).toHaveTextContent("Predicted: Clean");
    expect(screen.getByText(/Expert used:/)).toHaveTextContent("Expert used: Identity bypass");
    expect(screen.getByText("Misrouted")).toBeInTheDocument(); // predicted clean, true salt
    expect(screen.getByText("classifier.onnx (no expert)")).toBeInTheDocument();
    expect(screen.getByText("Inference: 2.20 ms")).toBeInTheDocument();
  });
});

describe("Soft Mixture-of-Experts Restoration", () => {
  it("shows the four weights, the main contributor and the branch outputs", async () => {
    mockApi({ "/api/restore/moe": () => json(moeResult) });
    renderWithBackend(<MoEWorkspace />);
    upload(photo());
    fireEvent.click(radio("None"));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    const weights = await screen.findByRole("group", { name: "Expert weights" });
    expect(within(weights).getByRole("progressbar", { name: "Blur expert" })).toHaveAttribute("aria-valuetext", "48.4%");
    expect(within(weights).getByText("Main contributor").parentElement).toHaveTextContent("Blur expert");
    for (const name of ["Identity (clean)", "Salt & Pepper expert", "Blur expert", "Occlusion expert"]) {
      expect(screen.getByRole("img", { name: `Output of the ${name} branch` })).toHaveAttribute("src", PNG);
    }
    expect(screen.getByText("Softmax gate weights, sum = 100.0%")).toBeInTheDocument();
    // No reference for an upload used as is.
    expect(screen.getByText(/No original: the image was restored as uploaded/)).toBeInTheDocument();
  });
});

describe("Face-to-Sketch Generator", () => {
  afterEach(() => setCamera(false));

  it("generates a sketch in the chosen style and offers the download", async () => {
    const fetchMock = mockApi({ "/api/sketch": () => json(sketchResult) });
    renderWithBackend(<SketchWorkspace active />);
    await pickSample("face_01.jpg");
    fireEvent.click(radio("Style 3"));
    fireEvent.click(screen.getByRole("button", { name: "Generate sketch" }));
    expect(Object.fromEntries(sentForm(fetchMock, "/api/sketch"))).toEqual({ sample_id: "faces-face_01", style: "3" });
    const link = await screen.findByRole("link", { name: "Download sketch" });
    expect(link).toHaveAttribute("download", "faces-face_01_sketch-style3.png");
    expect(screen.getByText("Style 3 (index 2)")).toBeInTheDocument();
    expect(screen.getByText("1 channel (greyscale)")).toBeInTheDocument();
  });

  it("explains when the browser has no camera support", async () => {
    mockApi();
    setCamera(true);
    renderWithBackend(<SketchWorkspace active />);
    fireEvent.click(screen.getByRole("button", { name: "Use webcam" }));
    expect(screen.getByRole("alert")).toHaveTextContent("does not support camera access");
  });

  it("explains a denied camera permission", async () => {
    mockApi();
    const getUserMedia = vi.fn().mockRejectedValue(new DOMException("denied", "NotAllowedError"));
    setCamera(true, getUserMedia);
    renderWithBackend(<SketchWorkspace active />);
    fireEvent.click(screen.getByRole("button", { name: "Use webcam" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Camera permission was denied");
    expect(getUserMedia).toHaveBeenCalled();
  });

  it("explains that the camera needs a secure page", async () => {
    mockApi();
    setCamera(false);
    renderWithBackend(<SketchWorkspace active />);
    fireEvent.click(screen.getByRole("button", { name: "Use webcam" }));
    expect(screen.getByRole("alert")).toHaveTextContent("secure page");
  });
});

describe("App shell", () => {
  it("opens the workspace from the URL and switches tabs", async () => {
    mockApi();
    window.history.replaceState(null, "", "/");
    render(<App />);
    expect(window.location.pathname).toBe("/universal-restoration");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Universal Restoration");

    fireEvent.click(screen.getAllByRole("link", { name: "Soft Mixture-of-Experts Restoration" })[0]);
    expect(window.location.pathname).toBe("/soft-mixture-of-experts-restoration");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Soft Mixture-of-Experts Restoration");
    expect(await screen.findByText("Backend online")).toBeInTheDocument();
  });

  it("shows a degraded backend and the system information", async () => {
    mockApi({ "/api/health": () => json(health(["moe"])) });
    window.history.replaceState(null, "", "/moe");
    render(<App />);
    expect(await screen.findByText("Backend degraded")).toBeInTheDocument();
    expect(screen.getByText("Model not available")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "System information" }));
    const dialog = screen.getByRole("dialog", { name: "System information" });
    expect(within(dialog).getByText("1.30.0")).toBeInTheDocument();
    expect(within(dialog).getByText("not ready, missing: moe.onnx")).toBeInTheDocument();
    expect(within(dialog).getByText("udae.onnx")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("shows an offline backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<App />);
    // The status pill and the page-wide alert.
    expect(await screen.findAllByText("Backend offline")).toHaveLength(2);
    expect(screen.getByRole("alert")).toHaveTextContent(/Cannot reach the server/);
  });
});
