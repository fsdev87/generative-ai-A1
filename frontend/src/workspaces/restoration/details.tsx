// Label-value rows shared by the restoration details cards; all values come from the response.
import type { OptionsResponse, RestorationResult } from "../../api/types";
import type { DetailItem } from "../../components/DetailsCard";
import { describeCorruption, describeSeverity, formatMs } from "../../lib/format";

export function corruptionDetails(result: RestorationResult): DetailItem[] {
  const c = result.corruption;
  const items: DetailItem[] = [{ label: "Corruption", value: describeCorruption(c) }];
  if (c && c.type !== "clean") {
    items.push({ label: "Severity", value: describeSeverity(c) });
    items.push({ label: "Seed", value: c.seed });
  }
  return items;
}

export function timingDetails(timing: { inference_ms: number; total_ms: number }): DetailItem[] {
  return [
    { label: "Inference time", value: formatMs(timing.inference_ms), icon: "timer", accent: true },
    { label: "Server time", value: formatMs(timing.total_ms), icon: "schedule" },
  ];
}

export function sourceDetails(result: RestorationResult, options: OptionsResponse | null): DetailItem[] {
  const s = result.source;
  const items: DetailItem[] = [
    { label: "Source", value: `${s.kind === "sample" ? "Sample" : "Upload"} · ${s.name} (${s.format}, ${s.width} × ${s.height})` },
  ];
  if (options) items.push({ label: "Image size", value: `${options.image_size} × ${options.image_size}` });
  return items;
}
