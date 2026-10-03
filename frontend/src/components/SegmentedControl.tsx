// Segmented control (Stitch: Corruption / Severity / Sketch style selectors): a row of buttons in a
// tinted track; the selected one is a white "chip" (or primary-filled for the `primary` variant).
export interface Segment<T extends string | number> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

const SELECTED = {
  surface: "bg-surface-container-lowest text-primary font-semibold shadow-sm",
  primary: "bg-primary text-on-primary font-medium shadow-sm",
};

export function SegmentedControl<T extends string | number>({
  label,
  segments,
  value,
  onChange,
  disabled = false,
  variant = "surface",
}: {
  label: string;
  segments: Segment<T>[];
  value: T | null;
  onChange: (value: T) => void;
  disabled?: boolean;
  variant?: keyof typeof SELECTED;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="grid p-1 rounded-lg bg-surface-container-low gap-1"
      style={{ gridTemplateColumns: `repeat(${segments.length}, minmax(0, 1fr))` }}
    >
      {segments.map((segment) => {
        const selected = segment.value === value;
        const isDisabled = disabled || segment.disabled;
        return (
          <button
            key={segment.value}
            type="button"
            role="radio"
            aria-checked={selected}
            title={segment.title}
            disabled={isDisabled}
            onClick={() => onChange(segment.value)}
            className={`py-1.5 px-1 rounded-md font-label-md text-label-md text-center truncate transition-all ${
              selected ? SELECTED[variant] : "text-on-surface-variant hover:text-on-surface"
            } ${isDisabled ? "opacity-50 cursor-not-allowed hover:text-on-surface-variant" : ""}`}
          >
            {segment.label}
          </button>
        );
      })}
    </div>
  );
}

/** Label row above a control: name on the left, current value on the right. */
export function FieldLabel({ label, value, htmlFor }: { label: string; value?: string; htmlFor?: string }) {
  return (
    <div className="flex items-center justify-between gap-space-sm">
      <label htmlFor={htmlFor} className="font-label-md text-label-md text-on-surface">
        {label}
      </label>
      {value && <span className="font-code-sm text-code-sm text-primary font-semibold truncate">{value}</span>}
    </div>
  );
}
