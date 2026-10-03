// The three square image panels of the restoration workspaces: "Original" (the clean reference,
// only when the server applied the corruption), "Model input" and "Restored output".
import type { CorruptionSettings, Metrics, RestorationResult } from "../../api/types";
import { Icon } from "../../components/Icon";
import { ImagePanel } from "../../components/ImagePanel";
import { useBackend } from "../../hooks/backend";
import type { SelectedImage } from "../../hooks/useSelectedImage";
import { formatDb, formatSsim, shortCorruption } from "../../lib/format";
import type { RestorationView } from "./useRestorationRun";

function inputBadge(corruption: CorruptionSettings | null): { text: string; tone: "error" | "neutral" } {
  if (corruption === null) return { text: "As uploaded", tone: "neutral" };
  if (corruption.type === "clean") return { text: "Clean", tone: "neutral" };
  return { text: "Corrupted", tone: "error" };
}

function gain(metrics: Metrics): string {
  const delta = metrics.psnr_output_db - metrics.psnr_input_db;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} dB`;
}

export function RestorationPanels<T extends RestorationResult>({
  view,
  selected,
  loading,
  outputBadge,
}: {
  view: RestorationView<T> | null;
  selected: SelectedImage | null;
  loading: boolean;
  outputBadge: string;
}) {
  const { options } = useBackend();
  const size = options ? `${options.image_size} × ${options.image_size}` : null;
  const result = view?.kind === "result" ? view.data : null;
  const preview = view?.kind === "preview" ? view.data : null;

  const original = result ? result.reference : (preview?.clean ?? null);
  const input = result?.input ?? preview?.image ?? null;
  const corruption = result ? result.corruption : (preview?.corruption ?? null);
  const metrics = result?.metrics ?? null;
  const badge = inputBadge(corruption);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-gutter">
      <ImagePanel
        title="Original"
        dotClass="bg-secondary"
        badge="Ground truth"
        src={original}
        alt="Clean original image before the corruption"
        chip={size}
        pixelated
        placeholder={
          result
            ? "No original: the image was restored as uploaded, so its clean version is unknown."
            : "The clean image appears here when the server applies the corruption."
        }
        footer={original ? <><span>Clean reference</span>{size && <span>{size} px</span>}</> : undefined}
      />
      {input ? (
        <ImagePanel
          title="Model input"
          dotClass="bg-error"
          badge={badge.text}
          badgeTone={badge.tone}
          src={input}
          alt="Image given to the model"
          chip={corruption && corruption.type !== "clean" ? shortCorruption(corruption) : size}
          pixelated
          placeholder=""
          footer={
            metrics ? (
              <>
                <span>PSNR: {formatDb(metrics.psnr_input_db)}</span>
                <span>SSIM: {formatSsim(metrics.ssim_input)}</span>
              </>
            ) : (
              <span>{result ? "PSNR / SSIM need the original" : "Preview: PSNR / SSIM after Restore"}</span>
            )
          }
        />
      ) : (
        <ImagePanel
          title="Model input"
          dotClass="bg-error"
          badge={selected ? "Selected" : undefined}
          src={selected?.previewUrl}
          alt={selected ? `Selected image ${selected.name}` : ""}
          fit="contain"
          placeholder="Upload an image or pick a sample."
          footer={selected ? <span className="truncate">{selected.name}</span> : undefined}
        />
      )}
      <ImagePanel
        title="Restored output"
        dotClass="bg-primary-container"
        badge={result ? outputBadge : undefined}
        badgeTone="primary"
        src={result?.output}
        alt="Restored output of the model"
        pixelated
        chip={
          result &&
          (metrics ? (
            gain(metrics)
          ) : (
            <span className="flex items-center gap-1">
              <Icon name="check_circle" className="text-[13px]" />
              Restored
            </span>
          ))
        }
        chipPrimary
        loading={loading}
        placeholder={preview ? "Press Restore to run the model on this input." : "Run Restore to see the result."}
        footer={
          result ? (
            <>
              <span className="text-primary font-semibold">
                {metrics ? `PSNR: ${formatDb(metrics.psnr_output_db)}` : "PSNR / SSIM need the original"}
              </span>
              {metrics && <span className="text-primary font-semibold">SSIM: {formatSsim(metrics.ssim_output)}</span>}
            </>
          ) : undefined
        }
      />
    </div>
  );
}
