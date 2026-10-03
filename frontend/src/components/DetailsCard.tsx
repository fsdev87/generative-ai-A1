// Compact details card (Stitch: label-value grid with the "Download result" button on the right)
// and the download button itself.
import type { ReactNode } from "react";
import { isPngDataUrl } from "../lib/download";
import { Card } from "./Card";
import { Icon } from "./Icon";

export interface DetailItem {
  label: string;
  value: ReactNode;
  icon?: string;
  /** Accent colour for the headline numbers (inference time). */
  accent?: boolean;
  /** Span the whole row (e.g. the hard-routing time breakdown). */
  wide?: boolean;
}

export function DetailsCard({
  title = "Run details",
  items,
  actions,
  children,
}: {
  title?: string;
  items: DetailItem[];
  actions?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <Card className="p-space-lg flex flex-col md:flex-row md:items-center justify-between gap-space-lg">
      <div className="flex flex-col gap-space-sm w-full min-w-0">
        <div className="flex items-center gap-space-xs">
          <Icon name="analytics" className="text-[18px] text-primary" />
          <h3 className="font-headline-sm text-headline-sm text-on-surface">{title}</h3>
        </div>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-space-xl gap-y-space-sm pt-space-xs">
          {items.map((item) => (
            <div key={item.label} className={`flex flex-col min-w-0 ${item.wide ? "col-span-2 sm:col-span-4" : ""}`}>
              <dt className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">
                {item.label}
              </dt>
              <dd
                className={`flex items-center gap-1 mt-0.5 font-code-sm text-code-sm font-semibold break-words ${
                  item.accent ? "text-primary" : "text-on-surface"
                }`}
              >
                {item.icon && <Icon name={item.icon} className="text-[14px] text-on-surface-variant" />}
                <span className="min-w-0">{item.value}</span>
              </dd>
            </div>
          ))}
        </dl>
        {children}
      </div>
      {actions && <div className="flex items-center gap-space-sm shrink-0">{actions}</div>}
    </Card>
  );
}

const BUTTON_STYLES = {
  secondary:
    "bg-surface-container-lowest hover:bg-surface-container-low text-on-surface border border-surface-container-highest",
  primary: "bg-primary hover:bg-primary-container text-on-primary",
};

/** Saves a PNG data URL from the API under `filename`; disabled until there is a result. */
export function DownloadButton({
  href,
  filename,
  label,
  variant = "secondary",
}: {
  href: string | null | undefined;
  filename: string;
  label: string;
  variant?: keyof typeof BUTTON_STYLES;
}) {
  const className = `py-2 px-space-md rounded-lg font-label-md text-label-md flex items-center justify-center gap-space-xs transition-colors shadow-sm whitespace-nowrap ${BUTTON_STYLES[variant]}`;
  if (!isPngDataUrl(href)) {
    return (
      <button type="button" disabled className={`${className} opacity-50 cursor-not-allowed`}>
        <Icon name="download" className="text-[18px]" />
        <span>{label}</span>
      </button>
    );
  }
  return (
    <a href={href} download={filename} className={className}>
      <Icon name="download" className="text-[18px]" />
      <span>{label}</span>
    </a>
  );
}
