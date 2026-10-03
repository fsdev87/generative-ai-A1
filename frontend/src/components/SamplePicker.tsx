// "Or pick a sample": thumbnails of the bundled clean images from GET /api/samples.
import { sampleImageUrl } from "../api/client";
import type { SampleInfo } from "../api/types";

export function SamplePicker({
  samples,
  error,
  selectedId,
  onSelect,
  disabled = false,
}: {
  samples: SampleInfo[] | null;
  error: string | null;
  selectedId: string | null;
  onSelect: (sample: SampleInfo) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-space-xs">
      <div className="flex items-center justify-between">
        <span className="font-label-md text-label-md text-on-surface">Or pick a sample</span>
        {samples && (
          <span className="font-code-sm text-code-sm text-on-surface-variant">
            {samples.length} sample{samples.length === 1 ? "" : "s"}
          </span>
        )}
      </div>
      {error && <p className="font-body-sm text-body-sm text-error">Samples could not be loaded: {error}</p>}
      {!error && samples === null && (
        <p className="font-body-sm text-body-sm text-on-surface-variant">Loading samples…</p>
      )}
      {samples?.length === 0 && (
        <p className="font-body-sm text-body-sm text-on-surface-variant">
          The server has no bundled samples; upload an image instead.
        </p>
      )}
      {samples && samples.length > 0 && (
        <div className="grid grid-cols-6 gap-space-xs pt-1">
          {samples.map((sample) => {
            const selected = sample.id === selectedId;
            return (
              <button
                key={sample.id}
                type="button"
                title={sample.name}
                aria-label={`Sample ${sample.name}`}
                aria-pressed={selected}
                disabled={disabled}
                onClick={() => onSelect(sample)}
                className={`relative aspect-square rounded-lg overflow-hidden p-0.5 transition-all ${
                  selected ? "bg-primary-container shadow-sm" : "bg-surface-container hover:opacity-90"
                }`}
              >
                <img
                  src={sampleImageUrl(sample)}
                  alt=""
                  loading="lazy"
                  className="w-full h-full object-cover rounded-md"
                />
                {selected && <span className="absolute inset-0 bg-primary/10 rounded-md pointer-events-none" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
