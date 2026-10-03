// The "Input" card of the three restoration workspaces (Stitch left column): upload or sample,
// corruption type and severity (or custom parameters), seed, workspace-specific controls (the
// routing toggle of the Hard-Routed workspace) and the Restore / Preview buttons.
import { useState, type ReactNode } from "react";
import type { CorruptionChoice } from "../../api/client";
import type { Level, ParamOption } from "../../api/types";
import { Card, Tag } from "../../components/Card";
import { Dropzone } from "../../components/Dropzone";
import { Icon } from "../../components/Icon";
import { SamplePicker } from "../../components/SamplePicker";
import { FieldLabel, SegmentedControl } from "../../components/SegmentedControl";
import { Spinner } from "../../components/Spinner";
import { useBackend, useSamples } from "../../hooks/backend";
import { CLASS_LABEL, LEVEL_LABEL, trimNumber } from "../../lib/format";
import { levelParams, type RestorationForm } from "./useRestorationForm";

const CHOICES: CorruptionChoice[] = ["none", "salt", "blur", "occlusion"];
const LEVELS: Level[] = ["low", "medium", "high"];

const PARAM_LABEL: Record<string, string> = {
  p: "Noise probability p",
  k: "Kernel size k",
  sigma: "Sigma σ",
  n: "Rectangles n",
  cover: "Covered fraction",
};

/** Slider step that gives round values inside a training range. */
const stepFor = (param: ParamOption) => (param.max <= 0.2 ? 0.005 : param.max <= 1 ? 0.01 : 0.1);

/** "kernel 5, σ 1.5" for the parameters of a level. */
function paramSummary(params: Record<string, number> | null): string {
  if (!params) return "";
  return Object.entries(params)
    .map(([name, value]) => `${name === "sigma" ? "σ" : name} ${trimNumber(value)}`)
    .join(", ");
}

function CorruptionHint({ choice, sourceKind }: { choice: CorruptionChoice; sourceKind: "upload" | "sample" | null }) {
  let text: string;
  if (choice !== "none") {
    text = "The server applies this corruption to the clean image, so the original is shown and PSNR / SSIM are measured.";
  } else if (sourceKind === "upload") {
    text = "None: the upload is restored as is (e.g. an already corrupted photo). Its true condition is unknown, so there is no original to compare with.";
  } else {
    text = "None: the clean sample is sent as clean, so PSNR / SSIM and oracle routing are available.";
  }
  return <p className="font-body-sm text-body-sm text-on-surface-variant">{text}</p>;
}

export function RestorationInputCard({
  form,
  runLabel = "Restore",
  onRun,
  onPreview,
  loading,
  children,
}: {
  form: RestorationForm;
  runLabel?: string;
  onRun: () => void;
  onPreview: () => void;
  loading: boolean;
  /** Workspace-specific controls shown above the buttons. */
  children?: ReactNode;
}) {
  const { options } = useBackend();
  const { samples, error: samplesError } = useSamples("pets");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, choice, level, customEnabled } = form;
  const selected = image.selected;
  const sourceKind = selected?.source.kind ?? null;
  const usesCorruption = choice !== "none";
  const paramOptions = options?.corruptions.find((c) => c.type === choice)?.params ?? [];
  const custom = form.customFor(options);
  const canRun = selected !== null && !form.seedError && !loading;

  return (
    <Card className="p-space-lg flex flex-col gap-space-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-space-xs">
          <Icon name="tune" className="text-[20px] text-primary" />
          <h2 className="font-headline-sm text-headline-sm text-on-surface">Input</h2>
        </div>
        <Tag>{sourceKind === "upload" ? "Upload" : sourceKind === "sample" ? "Sample" : "No image"}</Tag>
      </div>

      <div className="flex flex-col gap-space-xs">
        <Dropzone
          maxMb={options?.max_upload_mb}
          selectedName={sourceKind === "upload" ? selected?.name : undefined}
          onFile={(file) => {
            setUploadError(null);
            image.selectFile(file);
          }}
          onReject={setUploadError}
        />
        {uploadError && (
          <p role="alert" className="font-body-sm text-body-sm text-error">
            {uploadError}
          </p>
        )}
      </div>

      <SamplePicker
        samples={samples}
        error={samplesError}
        selectedId={selected?.source.kind === "sample" ? selected.source.id : null}
        onSelect={(sample) => {
          setUploadError(null);
          image.selectSample(sample);
        }}
      />

      <div className="flex flex-col gap-space-xs">
        <FieldLabel label="Corruption" value={choice === "none" ? "None" : CLASS_LABEL[choice]} />
        <SegmentedControl
          label="Corruption"
          value={choice}
          onChange={form.setChoice}
          segments={CHOICES.map((value) => ({
            value,
            label: value === "none" ? "None" : CLASS_LABEL[value],
            title: value === "none" ? "No artificial corruption (an upload is used as is)" : undefined,
          }))}
        />
        <CorruptionHint choice={choice} sourceKind={sourceKind} />
      </div>

      <div className="flex flex-col gap-space-xs">
        <FieldLabel
          label="Severity"
          value={usesCorruption && !customEnabled ? `${LEVEL_LABEL[level]} · ${paramSummary(levelParams(options, choice, level))}` : undefined}
        />
        <SegmentedControl
          label="Severity"
          value={usesCorruption && !customEnabled ? level : null}
          onChange={form.setLevel}
          disabled={!usesCorruption || customEnabled}
          segments={LEVELS.map((value) => ({ value, label: LEVEL_LABEL[value] }))}
        />
        <label className="flex items-center gap-space-sm pt-1 font-label-md text-label-md text-on-surface cursor-pointer w-fit">
          <input
            type="checkbox"
            className="accent-primary w-4 h-4"
            checked={customEnabled}
            disabled={!usesCorruption || !options}
            onChange={(event) => form.setCustomEnabled(event.target.checked)}
          />
          Custom parameters
        </label>
        {usesCorruption && customEnabled && custom && (
          <div className="flex flex-col gap-space-sm rounded-lg bg-surface-container-low p-space-md">
            {paramOptions.map((param) => (
              <div key={param.name} className="flex flex-col gap-1">
                <FieldLabel
                  label={PARAM_LABEL[param.name] ?? param.name}
                  htmlFor={param.choices ? undefined : `param-${param.name}`}
                  value={param.name === "cover" ? `${Math.round(custom[param.name] * 100)}%` : trimNumber(custom[param.name])}
                />
                {param.choices ? (
                  <SegmentedControl
                    label={PARAM_LABEL[param.name] ?? param.name}
                    value={custom[param.name]}
                    onChange={(value) => form.setCustomValue(options, param.name, value)}
                    segments={param.choices.map((value) => ({ value, label: String(value) }))}
                  />
                ) : (
                  <input
                    id={`param-${param.name}`}
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={stepFor(param)}
                    value={custom[param.name]}
                    onChange={(event) => form.setCustomValue(options, param.name, Number(event.target.value))}
                    className="w-full accent-primary"
                  />
                )}
                <span className="font-code-sm text-code-sm text-on-surface-variant">
                  Training range {trimNumber(param.min)} to {trimNumber(param.max)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-col gap-space-xs">
        <FieldLabel label="Seed" htmlFor="seed-input" />
        <input
          id="seed-input"
          inputMode="numeric"
          placeholder="Random"
          value={form.seedText}
          disabled={!usesCorruption}
          onChange={(event) => form.setSeedText(event.target.value)}
          aria-invalid={form.seedError !== null}
          className="w-full rounded-lg bg-surface-container-low px-space-md py-1.5 font-code-sm text-code-sm text-on-surface placeholder:text-on-surface-variant border border-transparent focus:border-primary focus:outline-none disabled:opacity-50"
        />
        {form.seedError ? (
          <p role="alert" className="font-body-sm text-body-sm text-error">{form.seedError}</p>
        ) : (
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            Empty for a random seed; the seed used is shown with the result. The same seed reproduces the corruption.
          </p>
        )}
      </div>

      {children}

      <div className="flex flex-col gap-space-sm pt-2">
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="w-full py-2.5 px-space-md rounded-lg bg-primary-container hover:bg-primary text-on-primary font-label-md text-label-md flex items-center justify-center gap-space-xs transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-primary-container"
        >
          {loading ? <Spinner className="w-4 h-4" light /> : <Icon name="bolt" className="text-[18px]" />}
          <span>{loading ? "Running…" : runLabel}</span>
        </button>
        <button
          type="button"
          onClick={onPreview}
          disabled={!canRun || !usesCorruption}
          title={usesCorruption ? "Apply the corruption on the server without restoring (POST /api/corrupt)" : "Choose a corruption to preview it"}
          className="w-full py-2 px-space-md rounded-lg bg-surface-container-lowest hover:bg-surface-container-low text-on-surface border border-surface-container-highest font-label-md text-label-md flex items-center justify-center gap-space-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Icon name="visibility" className="text-[18px]" />
          <span>Preview corruption</span>
        </button>
        {!selected && (
          <p className="font-body-sm text-body-sm text-on-surface-variant text-center">Upload an image or pick a sample first.</p>
        )}
      </div>

      <div className="p-space-sm bg-surface-container-low rounded-lg flex items-start gap-space-sm">
        <Icon name="info" className="text-[18px] text-on-surface-variant mt-0.5" />
        <div className="flex flex-col">
          <span className="font-label-sm text-label-sm text-on-surface">Preprocessing as in training</span>
          <span className="font-body-sm text-body-sm text-on-surface-variant text-[11px] leading-tight">
            The server converts the image to RGB and resizes it to {options ? `${options.image_size} × ${options.image_size}` : "the model size"} (bicubic,
            aspect ratio not kept) before the corruption and the model.
          </span>
        </div>
      </div>
    </Card>
  );
}
