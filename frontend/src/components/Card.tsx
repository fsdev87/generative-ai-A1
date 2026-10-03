// White card on the light surface (Stitch: bg-surface-container-lowest rounded-xl shadow-sm).
import type { ReactNode } from "react";
import { Icon } from "./Icon";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`bg-surface-container-lowest rounded-xl shadow-sm ${className}`}>{children}</div>;
}

/** Card title row: icon + heading on the left, an optional tag or badge on the right. */
export function CardTitle({
  icon,
  title,
  subtitle,
  aside,
  as: Heading = "h2",
}: {
  icon: string;
  title: string;
  subtitle?: ReactNode;
  aside?: ReactNode;
  as?: "h2" | "h3";
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm">
      <div className="flex items-center gap-space-sm min-w-0">
        <Icon name={icon} className="text-[20px] text-primary" />
        <div className="min-w-0">
          <Heading className="font-headline-sm text-headline-sm text-on-surface">{title}</Heading>
          {subtitle && <p className="font-body-sm text-body-sm text-on-surface-variant">{subtitle}</p>}
        </div>
      </div>
      {aside}
    </div>
  );
}

/** Small uppercase tag, e.g. "Source · Param" in the input card header. */
export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant bg-surface-container-low px-space-xs py-0.5 rounded whitespace-nowrap">
      {children}
    </span>
  );
}
