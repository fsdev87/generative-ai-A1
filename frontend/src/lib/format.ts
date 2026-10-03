// Display labels and number formatting. Everything formatted here comes from an API response.
import type { ClassScores, CorruptionSettings, CorruptionType, Expert, Level } from "../api/types";

export const CLASS_LABEL: Record<CorruptionType, string> = {
  clean: "Clean",
  salt: "Salt & Pepper",
  blur: "Blur",
  occlusion: "Occlusion",
};

export const CORRUPTION_NAME: Record<CorruptionType, string> = {
  clean: "Clean (no corruption)",
  salt: "Salt-and-pepper noise",
  blur: "Gaussian blur",
  occlusion: "Rectangular occlusion",
};

export const LEVEL_LABEL: Record<Level, string> = { low: "Low", medium: "Medium", high: "High" };

export const EXPERT_LABEL: Record<Expert, string> = {
  identity: "Identity bypass",
  specialist_salt: "Salt & Pepper specialist",
  specialist_blur: "Blur specialist",
  specialist_occlusion: "Occlusion specialist",
};

/** Names of the soft-MoE branches; clean is the identity branch. */
export const BRANCH_LABEL: Record<CorruptionType, string> = {
  clean: "Identity (clean)",
  salt: "Salt & Pepper expert",
  blur: "Blur expert",
  occlusion: "Occlusion expert",
};

/** A number without trailing zeros, e.g. 0.08 -> "0.08", 1.50 -> "1.5". */
export function trimNumber(value: number, maxDecimals = 3): string {
  return String(Number(value.toFixed(maxDecimals)));
}

export function formatMs(ms: number): string {
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms`;
}

export function formatPercent(fraction: number, decimals = 1): string {
  return `${(fraction * 100).toFixed(decimals)}%`;
}

/** PSNR as returned by the API, which caps identical images at 100 dB. */
export function formatDb(db: number): string {
  return db >= 100 ? "100 dB (identical)" : `${db.toFixed(2)} dB`;
}

/** SSIM with three decimals, as in the Stitch panels ("SSIM: 0.962"). */
export function formatSsim(value: number): string {
  return value.toFixed(3);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDuration(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h} h ${m} min`;
  if (m > 0) return `${m} min ${s % 60} s`;
  return `${s} s`;
}

/** Parameters of an applied corruption, e.g. "kernel 5 · σ 1.5". */
export function describeParams(c: CorruptionSettings): string {
  const p = c.params;
  switch (c.type) {
    case "salt":
      return `p ${trimNumber(p.p)}`;
    case "blur":
      return `kernel ${p.k} · σ ${trimNumber(p.sigma, 2)}`;
    case "occlusion":
      return `${p.n} rectangle${p.n === 1 ? "" : "s"} · ${formatPercent(p.cover)} covered`;
    default:
      return "";
  }
}

/** Full description for the details card, e.g. "Gaussian blur · kernel 5 · σ 1.5". */
export function describeCorruption(c: CorruptionSettings | null): string {
  if (c === null) return "None (image used as uploaded)";
  const params = describeParams(c);
  return params ? `${CORRUPTION_NAME[c.type]} · ${params}` : CORRUPTION_NAME[c.type];
}

/** Short label for the chip on the model-input image, e.g. "Blur · kernel 5 · σ 1.5". */
export function shortCorruption(c: CorruptionSettings): string {
  const params = describeParams(c);
  return params ? `${CLASS_LABEL[c.type]} · ${params}` : CLASS_LABEL[c.type];
}

/** "Medium" for a fixed level, "Custom (Medium range)" for custom parameters, "-" for clean. */
export function describeSeverity(c: CorruptionSettings): string {
  if (c.level === null) return "-";
  return c.custom ? `Custom (${LEVEL_LABEL[c.level]} range)` : LEVEL_LABEL[c.level];
}

/** The class with the largest score (ties: the first in class order). */
export function argmax(scores: ClassScores): CorruptionType {
  return (Object.keys(scores) as CorruptionType[]).reduce((best, k) => (scores[k] > scores[best] ? k : best));
}

/** Shannon entropy of a probability vector in nats (0 = certain, ln 4 = 1.386 = uniform). */
export function entropy(scores: ClassScores): number {
  return -Object.values(scores).reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0);
}

/** Difference between the largest and the second largest score. */
export function topMargin(scores: ClassScores): number {
  const sorted = Object.values(scores).sort((a, b) => b - a);
  return sorted[0] - (sorted[1] ?? 0);
}
