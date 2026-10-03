// Page frame of a workspace (Stitch: max-w-[1600px] container, 12-column grid with the input card
// in 4 columns and the results in 8; the columns stack below the lg breakpoint).
import type { ReactNode } from "react";

export function WorkspaceLayout({ header, input, results }: { header: ReactNode; input: ReactNode; results: ReactNode }) {
  return (
    <div className="w-full px-gutter py-space-lg max-w-[1600px] mx-auto">
      {header}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter items-start">
        <div className="lg:col-span-4 w-full min-w-0 flex flex-col gap-space-md">{input}</div>
        <div className="lg:col-span-8 w-full min-w-0 flex flex-col gap-space-lg">{results}</div>
      </div>
    </div>
  );
}
