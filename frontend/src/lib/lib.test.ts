import { describe, expect, it } from "vitest";
import type { CorruptionSettings } from "../api/types";
import { parseSeed } from "../workspaces/restoration/useRestorationForm";
import { downloadFilename, isPngDataUrl } from "./download";
import { absoluteErrorMap } from "./errorMap";
import { argmax, describeCorruption, describeSeverity, entropy, formatDb, formatPercent, shortCorruption } from "./format";
import { workspaceForPath } from "./routes";

const blur: CorruptionSettings = {
  type: "blur",
  level: "medium",
  custom: true,
  seed: 7,
  params: { k: 5, sigma: 1.2 },
  rects: null,
  spec: {},
};

describe("download helpers", () => {
  it("builds a safe PNG file name from the source name", () => {
    expect(downloadFilename("Abyssinian_12.jpg", "restored-universal")).toBe("Abyssinian_12_restored-universal.png");
    expect(downloadFilename("my photo (1).jpeg", "sketch-style2")).toBe("my_photo_1_sketch-style2.png");
    expect(downloadFilename("C:\\fakepath\\..png", "x")).toBe("image_x.png");
  });

  it("accepts only PNG data URLs", () => {
    expect(isPngDataUrl("data:image/png;base64,iVBORw0KGgo=")).toBe(true);
    expect(isPngDataUrl("https://example.com/a.png")).toBe(false);
    expect(isPngDataUrl(null)).toBe(false);
  });
});

describe("format", () => {
  it("describes corruption settings from the API", () => {
    expect(describeCorruption(blur)).toBe("Gaussian blur · kernel 5 · σ 1.2");
    expect(shortCorruption(blur)).toBe("Blur · kernel 5 · σ 1.2");
    expect(describeSeverity(blur)).toBe("Custom (Medium range)");
    expect(describeCorruption(null)).toMatch(/as uploaded/);
    const occlusion = { ...blur, type: "occlusion" as const, custom: false, params: { n: 1, cover: 0.0998 } };
    expect(describeCorruption(occlusion)).toBe("Rectangular occlusion · 1 rectangle · 10.0% covered");
  });

  it("formats percentages, PSNR and probability statistics", () => {
    expect(formatPercent(0.891)).toBe("89.1%");
    expect(formatDb(27.414)).toBe("27.41 dB");
    expect(formatDb(100)).toMatch(/identical/);
    const p = { clean: 0.1, salt: 0.6, blur: 0.2, occlusion: 0.1 };
    expect(argmax(p)).toBe("salt");
    expect(entropy({ clean: 0.25, salt: 0.25, blur: 0.25, occlusion: 0.25 })).toBeCloseTo(Math.log(4));
    expect(entropy({ clean: 1, salt: 0, blur: 0, occlusion: 0 })).toBeCloseTo(0);
  });
});

describe("seed parsing", () => {
  it("accepts empty and whole numbers in range only", () => {
    expect(parseSeed("")).toEqual({ seed: null, error: null });
    expect(parseSeed(" 42 ")).toEqual({ seed: 42, error: null });
    expect(parseSeed("-1").error).toMatch(/whole number/);
    expect(parseSeed("1.5").error).toMatch(/whole number/);
    expect(parseSeed("2147483648").error).toMatch(/whole number/);
  });
});

describe("absoluteErrorMap", () => {
  it("computes the per-pixel error and the mean absolute error", () => {
    const out = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const ref = new Uint8ClampedArray([255, 255, 255, 255, 51, 51, 51, 255]);
    const map = absoluteErrorMap(out, ref, 2);
    expect([...map.pixels]).toEqual([0, 0, 0, 255, 102, 102, 102, 255]);
    expect(map.meanAbsError).toBeCloseTo(0.1); // (0 + 51) / 2 / 255
    expect(map.maxAbsError).toBeCloseTo(0.2);
  });

  it("rejects images of different sizes", () => {
    expect(() => absoluteErrorMap(new Uint8ClampedArray(4), new Uint8ClampedArray(8))).toThrow();
  });
});

describe("routes", () => {
  it("maps paths and aliases to the four workspaces", () => {
    expect(workspaceForPath("/hard-routed-restoration")).toMatchObject({ workspace: { key: "hard" }, canonical: true });
    expect(workspaceForPath("/moe/")).toMatchObject({ workspace: { key: "moe" }, canonical: false });
    expect(workspaceForPath("/")).toMatchObject({ workspace: { key: "universal" }, canonical: false });
    expect(workspaceForPath("/nope").workspace.title).toBe("Universal Restoration");
  });
});
