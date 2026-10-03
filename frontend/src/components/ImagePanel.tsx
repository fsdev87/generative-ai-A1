// Square image panel (Stitch: "Original", "Model input", "Restored output", "Photo",
// "Generated sketch"): title with a coloured dot and a badge, the image with a caption chip, and
// a footer row for metrics. Shows a placeholder text when there is no image yet.
import type { ReactNode } from "react";
import { Icon } from "./Icon";
import { Spinner } from "./Spinner";

const BADGE_TONES = {
  neutral: "bg-surface-container text-on-surface-variant",
  error: "bg-error-container text-on-error-container",
  primary: "bg-primary-fixed text-on-primary-fixed",
  solid: "bg-primary text-on-primary",
};

export type BadgeTone = keyof typeof BADGE_TONES;

export function ImagePanel({
  title,
  icon,
  dotClass,
  badge,
  badgeTone = "neutral",
  src,
  alt,
  chip,
  chipPrimary = false,
  footer,
  placeholder,
  loading = false,
  loadingText = "Running the model…",
  fit = "cover",
  pixelated = false,
  highlight = false,
}: {
  title: string;
  icon?: string;
  dotClass?: string;
  badge?: string;
  badgeTone?: BadgeTone;
  src?: string | null;
  alt: string;
  chip?: ReactNode;
  chipPrimary?: boolean;
  footer?: ReactNode;
  placeholder: string;
  loading?: boolean;
  loadingText?: string;
  /** "contain" for a preview of an arbitrary photo, "cover" for the square model images. */
  fit?: "cover" | "contain";
  /** Show the 128x128 model images with hard pixel edges, so noise is not smoothed away. */
  pixelated?: boolean;
  highlight?: boolean;
}) {
  return (
    <section
      aria-label={title}
      className="bg-surface-container-lowest rounded-xl p-space-md shadow-sm flex flex-col min-w-0"
    >
      <div className="flex items-center justify-between gap-space-xs mb-space-sm">
        <div className="flex items-center gap-space-xs min-w-0">
          {icon ? (
            <Icon name={icon} className={`text-[18px] ${highlight ? "text-primary" : "text-on-surface-variant"}`} />
          ) : (
            <span className={`w-2 h-2 rounded-full shrink-0 ${dotClass ?? "bg-outline"}`} />
          )}
          <h3
            className={`font-headline-sm text-headline-sm truncate ${highlight ? "text-primary" : "text-on-surface"}`}
          >
            {title}
          </h3>
        </div>
        {badge && (
          <span
            className={`font-label-sm text-label-sm px-space-xs py-0.5 rounded whitespace-nowrap ${BADGE_TONES[badgeTone]}`}
          >
            {badge}
          </span>
        )}
      </div>
      <div className="w-full aspect-square rounded-lg overflow-hidden bg-surface-container-low relative">
        {src ? (
          <img
            src={src}
            alt={alt}
            className={`w-full h-full ${fit === "cover" ? "object-cover" : "object-contain"} ${
              pixelated ? "[image-rendering:pixelated]" : ""
            }`}
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-space-xs p-space-md text-center text-on-surface-variant">
            <Icon name="image" className="text-[28px] opacity-60" />
            <span className="font-body-sm text-body-sm">{placeholder}</span>
          </div>
        )}
        {chip && src && (
          <div
            className={`absolute bottom-2 left-2 max-w-[calc(100%-1rem)] truncate px-space-xs py-0.5 rounded font-code-sm text-code-sm ${
              chipPrimary
                ? "bg-primary-container text-on-primary shadow-sm"
                : "bg-inverse-surface/80 backdrop-blur-sm text-inverse-on-surface"
            }`}
          >
            {chip}
          </div>
        )}
        {loading && (
          <div className="absolute inset-0 bg-surface-container-lowest/80 backdrop-blur-sm flex flex-col items-center justify-center gap-space-sm">
            <Spinner />
            <span className="font-label-md text-label-md text-on-surface">{loadingText}</span>
          </div>
        )}
      </div>
      {footer && (
        <div className="mt-space-sm flex items-center justify-between gap-space-xs text-on-surface-variant font-label-sm text-label-sm">
          {footer}
        </div>
      )}
    </section>
  );
}
