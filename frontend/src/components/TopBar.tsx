// Top app bar (Stitch header): logo, the four workspace tabs, backend status pill and a button
// for the system information dialog (replaces the design's decorative avatar).
import type { MouseEvent } from "react";
import type { BackendStatus } from "../hooks/backend";
import { WORKSPACES, type Workspace } from "../lib/routes";
import { Icon } from "./Icon";

const STATUS = {
  checking: { label: "Checking backend…", dot: "bg-outline", ping: false },
  online: { label: "Backend online", dot: "bg-emerald-500", ping: true },
  degraded: { label: "Backend degraded", dot: "bg-amber-500", ping: false },
  offline: { label: "Backend offline", dot: "bg-error", ping: false },
} as const;

export function StatusPill({ status, onClick }: { status: BackendStatus; onClick: () => void }) {
  const style = STATUS[status];
  const hint = {
    checking: "Waiting for the first health check",
    online: "All models are loaded",
    degraded: "Some models are missing or failed to load; click for details",
    offline: "The backend does not answer; click for details",
  }[status];
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint}
      className="flex items-center gap-space-xs px-space-md py-1 rounded-full bg-surface-container border border-outline-variant/30 hover:bg-surface-container-high transition-colors"
    >
      <span className="relative flex h-2 w-2">
        {style.ping && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${style.dot}`} />
      </span>
      <span role="status" className="font-label-sm text-label-sm text-on-surface-variant whitespace-nowrap">
        {style.label}
      </span>
    </button>
  );
}

function Tabs({
  active,
  onNavigate,
  compact,
}: {
  active: Workspace;
  onNavigate: (w: Workspace) => void;
  compact: boolean;
}) {
  const go = (event: MouseEvent, workspace: Workspace) => {
    // Let the browser handle ctrl/cmd-click (new tab); otherwise switch without reloading.
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    onNavigate(workspace);
  };
  return WORKSPACES.map((workspace) => {
    const current = workspace.key === active.key;
    const base = compact ? "whitespace-nowrap px-space-md py-1 rounded-lg" : "px-space-md py-1.5 rounded-lg transition-all";
    const state = current
      ? `bg-surface-container text-primary font-semibold ${compact ? "" : "shadow-[0_1px_2px_rgba(15,23,42,0.04)]"}`
      : "text-on-surface-variant font-label-md text-label-md hover:text-on-surface hover:bg-surface-container-high";
    return (
      <a
        key={workspace.key}
        href={workspace.path}
        aria-current={current ? "page" : undefined}
        onClick={(event) => go(event, workspace)}
        className={`${base} ${state}`}
      >
        {workspace.title}
      </a>
    );
  });
}

export function TopBar({
  active,
  onNavigate,
  status,
  onOpenInfo,
}: {
  active: Workspace;
  onNavigate: (w: Workspace) => void;
  status: BackendStatus;
  onOpenInfo: () => void;
}) {
  return (
    // Sticky instead of the design's fixed header, so the second tab row on smaller screens
    // never covers the page content.
    <header className="sticky top-0 z-40 bg-surface/90 backdrop-blur-md shadow-[0_1px_8px_rgba(0,0,0,0.04)]">
      <div className="h-14 w-full px-gutter flex items-center justify-between gap-space-lg">
        <div className="flex items-center gap-space-xl min-w-0">
          <div className="flex items-center gap-space-sm select-none shrink-0">
            <div className="w-7 h-7 rounded-lg bg-primary-container flex items-center justify-center text-on-primary shadow-sm">
              <Icon name="view_in_ar" className="text-[18px]" />
            </div>
            <span className="font-headline-sm text-headline-sm text-on-surface tracking-tight font-semibold">
              GenAI Studio
            </span>
          </div>
          <nav
            aria-label="Workspaces"
            className="hidden xl:flex items-center gap-space-xs p-space-xs bg-surface-container-low rounded-lg"
          >
            <Tabs active={active} onNavigate={onNavigate} compact={false} />
          </nav>
        </div>
        <div className="flex items-center gap-space-md">
          <StatusPill status={status} onClick={onOpenInfo} />
          <button
            type="button"
            onClick={onOpenInfo}
            aria-label="System information"
            title="System information"
            className="w-8 h-8 rounded-full bg-primary hover:bg-primary-container flex items-center justify-center text-on-primary transition-colors"
          >
            <Icon name="info" className="text-[18px]" />
          </button>
        </div>
      </div>
      <nav
        aria-label="Workspaces (compact)"
        className="xl:hidden flex items-center gap-space-xs overflow-x-auto px-gutter py-space-xs bg-surface-container-low"
      >
        <Tabs active={active} onNavigate={onNavigate} compact />
      </nav>
    </header>
  );
}
