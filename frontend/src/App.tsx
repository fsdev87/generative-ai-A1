// Application shell: top bar with the four workspace tabs, the active workspace, footer and the
// system information dialog. All four workspaces stay mounted (hidden when inactive), so inputs
// and results survive switching tabs.
import { useCallback, useEffect, useState } from "react";
import { Alert } from "./components/Alert";
import { Footer } from "./components/Footer";
import { SystemInfoDialog } from "./components/SystemInfoDialog";
import { TopBar } from "./components/TopBar";
import { BackendProvider, useBackend } from "./hooks/backend";
import { useWorkspaceRoute } from "./lib/routes";
import { HardRoutedWorkspace } from "./workspaces/HardRoutedWorkspace";
import { MoEWorkspace } from "./workspaces/MoEWorkspace";
import { SketchWorkspace } from "./workspaces/SketchWorkspace";
import { UniversalWorkspace } from "./workspaces/UniversalWorkspace";

function Shell() {
  const [workspace, navigate] = useWorkspaceRoute();
  const { status, health, healthError } = useBackend();
  const [infoOpen, setInfoOpen] = useState(false);
  const closeInfo = useCallback(() => setInfoOpen(false), []);

  useEffect(() => {
    document.title = `${workspace.title} · GenAI Studio`;
  }, [workspace]);

  return (
    <div className="min-h-screen flex flex-col bg-surface">
      <TopBar active={workspace} onNavigate={navigate} status={status} onOpenInfo={() => setInfoOpen(true)} />
      <main className="flex-1 w-full">
        {status === "offline" && (
          <div className="max-w-[1600px] mx-auto px-gutter pt-space-lg">
            <Alert tone="error" title="Backend offline">
              {healthError} The page checks again every 10 seconds.
            </Alert>
          </div>
        )}
        <div hidden={workspace.key !== "universal"}>
          <UniversalWorkspace />
        </div>
        <div hidden={workspace.key !== "hard"}>
          <HardRoutedWorkspace />
        </div>
        <div hidden={workspace.key !== "moe"}>
          <MoEWorkspace />
        </div>
        <div hidden={workspace.key !== "sketch"}>
          <SketchWorkspace active={workspace.key === "sketch"} />
        </div>
      </main>
      <Footer health={health} />
      {infoOpen && <SystemInfoDialog onClose={closeInfo} />}
    </div>
  );
}

export function App() {
  return (
    <BackendProvider>
      <Shell />
    </BackendProvider>
  );
}
