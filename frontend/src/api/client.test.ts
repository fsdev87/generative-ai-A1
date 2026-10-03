import { describe, expect, it, vi } from "vitest";
import { json, mockApi, options } from "../test/fixtures";
import {
  api,
  ApiError,
  buildRestoreForm,
  buildSketchForm,
  corruptionField,
  oracleAvailable,
  type CorruptionRequest,
} from "./client";

const file = new File(["fake image bytes"], "photo.jpg", { type: "image/jpeg" });
const sample = { kind: "sample" as const, id: "pets-abyssinian_12" };
const upload = { kind: "upload" as const, file };
const request = (overrides: Partial<CorruptionRequest> = {}): CorruptionRequest => ({
  choice: "blur",
  level: "medium",
  custom: null,
  seed: null,
  ...overrides,
});
const fields = (form: FormData) => [...form.keys()].sort();

describe("buildRestoreForm", () => {
  it("sends a sample with None as corruption=clean and nothing else", () => {
    const form = buildRestoreForm(sample, request({ choice: "none", seed: 5 }));
    expect(fields(form)).toEqual(["corruption", "sample_id"]);
    expect(form.get("sample_id")).toBe("pets-abyssinian_12");
    expect(form.get("corruption")).toBe("clean");
  });

  it("omits the corruption for an upload used as is", () => {
    const form = buildRestoreForm(upload, request({ choice: "none", seed: 5 }));
    expect(fields(form)).toEqual(["file"]);
    const sent = form.get("file") as File;
    expect(sent.name).toBe("photo.jpg");
  });

  it("sends the type, level and seed of a server-side corruption", () => {
    const form = buildRestoreForm(upload, request({ choice: "occlusion", level: "high", seed: 42 }));
    expect(fields(form)).toEqual(["corruption", "file", "level", "seed"]);
    expect(form.get("corruption")).toBe("occlusion");
    expect(form.get("level")).toBe("high");
    expect(form.get("seed")).toBe("42");
  });

  it("sends custom parameters instead of a level", () => {
    const form = buildRestoreForm(sample, request({ choice: "blur", custom: { k: 7, sigma: 1.2 } }));
    expect(fields(form)).toEqual(["corruption", "k", "sample_id", "sigma"]);
    expect(form.get("k")).toBe("7");
    expect(form.get("sigma")).toBe("1.2");
  });

  it("adds the routing mode for the hard-routed endpoint", () => {
    const form = buildRestoreForm(sample, request({ choice: "salt" }), "oracle");
    expect(form.get("routing_mode")).toBe("oracle");
    expect(buildRestoreForm(sample, request()).has("routing_mode")).toBe(false);
  });
});

describe("oracle routing rules", () => {
  it.each([
    ["sample", "none", true],
    ["sample", "salt", true],
    ["upload", "blur", true],
    ["upload", "none", false],
    [null, "none", false],
    [null, "occlusion", true],
  ] as const)("source %s with corruption %s -> %s", (source, choice, allowed) => {
    expect(oracleAvailable(source, choice)).toBe(allowed);
  });

  it("matches whether a corruption field is sent", () => {
    expect(corruptionField("sample", "none")).toBe("clean");
    expect(corruptionField("upload", "none")).toBeNull();
    expect(corruptionField("upload", "salt")).toBe("salt");
  });
});

describe("buildSketchForm", () => {
  it("sends the image and the style", () => {
    const form = buildSketchForm({ kind: "sample", id: "faces-face_01" }, 3);
    expect(fields(form)).toEqual(["sample_id", "style"]);
    expect(form.get("style")).toBe("3");
    expect(fields(buildSketchForm(upload, 1))).toEqual(["file", "style"]);
  });
});

describe("api requests and errors", () => {
  it("returns parsed JSON and builds the samples URL", async () => {
    const fetchMock = mockApi();
    await expect(api.options()).resolves.toEqual(options);
    const samples = await api.samples("pets");
    expect(samples[0].id).toBe("pets-abyssinian_12");
    expect(fetchMock).toHaveBeenCalledWith("/api/samples?category=pets", undefined);
  });

  it("posts the form data to the restore endpoint", async () => {
    const fetchMock = mockApi({ "/api/restore/moe": () => json({ ok: true }) });
    const form = buildRestoreForm(sample, request());
    await api.moe(form);
    const [, init] = fetchMock.mock.calls.at(-1)!;
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(form);
  });

  it.each([
    [503, "Model file udae.onnx is missing from /models. Add it and restart the backend."],
    [415, "The file is not a JPEG, PNG, WEBP or BMP image."],
    [400, "Send either an image file or a sample_id, not both."],
    [422, "Invalid request: level: Input should be 'low', 'medium' or 'high'"],
  ])("raises ApiError %i with the backend's detail", async (status, detail) => {
    mockApi({ "/api/restore/universal": () => json({ detail }, status) });
    const error = await api.universal(new FormData()).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status, detail });
    expect((error as ApiError).isModelUnavailable).toBe(status === 503);
  });

  it("explains a 413 from the proxy that has no JSON body", async () => {
    mockApi({ "/api/sketch": () => new Response("<html>413 Request Entity Too Large</html>", { status: 413 }) });
    await expect(api.sketch(new FormData())).rejects.toMatchObject({ status: 413, detail: expect.stringMatching(/10 MB/) });
  });

  it("explains a 502 while the backend is down", async () => {
    mockApi({ "/api/health": () => new Response("Bad gateway", { status: 502 }) });
    await expect(api.health()).rejects.toMatchObject({ status: 502, detail: expect.stringMatching(/not reachable/) });
  });

  it("turns a network failure into status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.health()).rejects.toMatchObject({ status: 0, detail: expect.stringMatching(/Cannot reach/) });
  });
});
