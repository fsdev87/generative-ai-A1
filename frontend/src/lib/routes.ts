// The four workspaces and a minimal path-based router (History API), so every workspace has a
// deep-linkable URL such as /hard-routed-restoration. nginx serves index.html for these paths.
import { useCallback, useEffect, useState } from "react";
import type { WorkspaceKey } from "../api/types";

export interface Workspace {
  key: WorkspaceKey;
  /** Exact workspace name required by the assignment. */
  title: string;
  path: string;
  /** Short paths that also open the workspace. */
  aliases: string[];
  /** Material Symbols icon name. */
  icon: string;
}

export const WORKSPACES: Workspace[] = [
  {
    key: "universal",
    title: "Universal Restoration",
    path: "/universal-restoration",
    aliases: ["/universal"],
    icon: "auto_fix_high",
  },
  {
    key: "hard",
    title: "Hard-Routed Restoration",
    path: "/hard-routed-restoration",
    aliases: ["/hard"],
    icon: "account_tree",
  },
  {
    key: "moe",
    title: "Soft Mixture-of-Experts Restoration",
    path: "/soft-mixture-of-experts-restoration",
    aliases: ["/moe"],
    icon: "donut_large",
  },
  {
    key: "sketch",
    title: "Face-to-Sketch Generator",
    path: "/face-to-sketch-generator",
    aliases: ["/sketch"],
    icon: "draw",
  },
];

/** The workspace for a URL path; unknown paths (and "/") fall back to the first workspace. */
export function workspaceForPath(pathname: string): { workspace: Workspace; canonical: boolean } {
  const path = pathname.replace(/\/+$/, "").toLowerCase() || "/";
  const exact = WORKSPACES.find((w) => w.path === path);
  if (exact) return { workspace: exact, canonical: true };
  const alias = WORKSPACES.find((w) => w.aliases.includes(path));
  return { workspace: alias ?? WORKSPACES[0], canonical: false };
}

/** The active workspace, kept in sync with the address bar and the back/forward buttons. */
export function useWorkspaceRoute(): [Workspace, (w: Workspace) => void] {
  const [workspace, setWorkspace] = useState(() => workspaceForPath(window.location.pathname).workspace);

  useEffect(() => {
    // Normalise "/" and aliases to the canonical URL without adding a history entry.
    const { workspace: current, canonical } = workspaceForPath(window.location.pathname);
    if (!canonical) window.history.replaceState(null, "", current.path + window.location.search);
    const onPopState = () => setWorkspace(workspaceForPath(window.location.pathname).workspace);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((next: Workspace) => {
    if (window.location.pathname !== next.path) window.history.pushState(null, "", next.path);
    setWorkspace(next);
  }, []);

  return [workspace, navigate];
}
