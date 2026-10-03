// Horizontal bars for the four class scores: classifier probabilities (Hard-Routed, "grid"
// layout) and gate weights (Soft MoE, "list" layout). The highlighted class (prediction or main
// contributor) is drawn in the accent colour, as in the Stitch design.
import { CLASSES, type ClassScores, type CorruptionType } from "../api/types";
import { formatPercent } from "../lib/format";
import { Icon } from "./Icon";

export function ScoreBars({
  label,
  scores,
  labels,
  highlight,
  highlightTag,
  marker,
  variant,
}: {
  /** Accessible name of the group, e.g. "Classifier probabilities". */
  label: string;
  scores: ClassScores;
  labels: Record<CorruptionType, string>;
  highlight: CorruptionType;
  highlightTag: string;
  /** A second class to mark, e.g. the true class when it differs from the prediction. */
  marker?: { cls: CorruptionType; tag: string };
  variant: "grid" | "list";
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className={
        variant === "grid"
          ? "grid grid-cols-1 sm:grid-cols-2 gap-x-space-xl gap-y-space-md"
          : "flex flex-col gap-space-md"
      }
    >
      {CLASSES.map((cls) => {
        const value = scores[cls];
        const active = cls === highlight;
        const width = `${Math.min(1, Math.max(0, value)) * 100}%`;
        const percent = formatPercent(value);
        return (
          <div
            key={cls}
            className={`flex flex-col gap-1.5 ${active && variant === "grid" ? "p-space-xs -m-space-xs rounded-lg bg-surface-container-low" : ""}`}
          >
            <div className="flex items-center justify-between gap-space-sm font-label-md text-label-md">
              <div
                className={`flex items-center gap-1.5 min-w-0 ${active ? "text-primary font-semibold" : variant === "grid" ? "text-on-surface-variant" : "text-on-surface"}`}
              >
                {active && variant === "grid" && <Icon name="check_circle" className="text-[16px]" />}
                <span className="truncate">{labels[cls]}</span>
                {active && (
                  <span className="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-primary-fixed text-primary font-semibold whitespace-nowrap">
                    {highlightTag}
                  </span>
                )}
                {marker?.cls === cls && (
                  <span className="font-label-sm text-label-sm px-2 py-0.5 rounded-full border border-outline-variant text-on-surface-variant whitespace-nowrap">
                    {marker.tag}
                  </span>
                )}
              </div>
              <span
                className={`font-code-sm text-code-sm ${active ? "text-primary font-bold" : "text-on-surface-variant"}`}
              >
                {percent}
              </span>
            </div>
            <div
              role="progressbar"
              aria-label={labels[cls]}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Number((value * 100).toFixed(1))}
              aria-valuetext={percent}
              className={`w-full rounded-full overflow-hidden ${active && variant === "grid" ? "h-2.5" : "h-2"} ${
                variant === "grid" ? "bg-surface-container" : "bg-surface-container-low"
              }`}
            >
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  active ? "bg-primary" : variant === "grid" ? "bg-outline-variant" : "bg-surface-dim"
                }`}
                style={{ width }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
