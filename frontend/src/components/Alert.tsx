// Inline message box for errors (API detail strings), warnings (missing models) and hints.
import type { ReactNode } from "react";
import { Icon } from "./Icon";

const TONES = {
  error: { box: "bg-error-container text-on-error-container", icon: "error" },
  warning: { box: "bg-tertiary-fixed text-on-tertiary-fixed", icon: "warning" },
  info: { box: "bg-surface-container-low text-on-surface-variant", icon: "info" },
} as const;

export function Alert({
  tone,
  title,
  children,
}: {
  tone: keyof typeof TONES;
  title?: string;
  children: ReactNode;
}) {
  const style = TONES[tone];
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`flex items-start gap-space-sm rounded-lg p-space-md font-body-sm text-body-sm ${style.box}`}
    >
      <Icon name={style.icon} className="text-[18px] mt-px shrink-0" />
      <div className="min-w-0 break-words">
        {title && <p className="font-label-md text-label-md font-semibold mb-0.5">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  );
}
