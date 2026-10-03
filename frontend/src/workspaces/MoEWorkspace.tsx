// Task 3 workspace, "Soft Mixture-of-Experts Restoration" (design/stitch/moe.html): the gate of
// moe.onnx gives every branch (identity + three experts) a weight; the output is the weighted sum
// of the branch outputs.
import { api } from "../api/client";
import { CLASSES, type MoEResponse } from "../api/types";
import { Card, CardTitle } from "../components/Card";
import { DetailsCard, DownloadButton } from "../components/DetailsCard";
import { ErrorMapCard } from "../components/ErrorMapCard";
import { ModelStatusNotice } from "../components/ModelStatusNotice";
import { RequestError } from "../components/RequestError";
import { ScoreBars } from "../components/ScoreBars";
import { WorkspaceHeader } from "../components/WorkspaceHeader";
import { WorkspaceLayout } from "../components/WorkspaceLayout";
import { useBackend } from "../hooks/backend";
import { downloadFilename } from "../lib/download";
import { BRANCH_LABEL, CLASS_LABEL, formatPercent } from "../lib/format";
import { corruptionDetails, sourceDetails, timingDetails } from "./restoration/details";
import { RestorationInputCard } from "./restoration/RestorationInputCard";
import { RestorationPanels } from "./restoration/RestorationPanels";
import { useRestorationForm } from "./restoration/useRestorationForm";
import { useRestorationRun } from "./restoration/useRestorationRun";

const THUMB_LABEL = { ...CLASS_LABEL, clean: "Identity" };

export function ExpertWeightsCard({ result }: { result: MoEResponse }) {
  const total = CLASSES.reduce((sum, cls) => sum + result.weights[cls], 0);
  return (
    <Card className="p-space-lg flex flex-col gap-space-lg">
      <CardTitle
        as="h3"
        icon="donut_large"
        title="Expert weights"
        aside={
          <p className="font-label-sm text-label-sm text-on-surface-variant">
            Softmax gate weights, sum = {formatPercent(total)}
          </p>
        }
      />
      <ScoreBars
        label="Expert weights"
        variant="list"
        scores={result.weights}
        labels={BRANCH_LABEL}
        highlight={result.dominant_branch}
        highlightTag="Main contributor"
      />
      <div className="pt-space-xs flex flex-col gap-space-sm">
        <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
          Output of every branch
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-space-md">
          {CLASSES.map((cls) => {
            const main = cls === result.dominant_branch;
            return (
              <figure
                key={cls}
                className={`bg-surface-container-low rounded-lg p-space-sm flex flex-col items-center gap-space-xs text-center ${
                  main ? "ring-2 ring-primary" : ""
                }`}
              >
                <div className="w-full aspect-square rounded-md overflow-hidden bg-surface-container">
                  <img
                    src={result.branch_outputs[cls]}
                    alt={`Output of the ${BRANCH_LABEL[cls]} branch`}
                    className="w-full h-full object-cover [image-rendering:pixelated]"
                  />
                </div>
                <figcaption className="flex flex-col">
                  <span className={`font-label-md text-label-md ${main ? "text-primary font-semibold" : "text-on-surface font-medium"}`}>
                    {THUMB_LABEL[cls]}
                  </span>
                  <span className={`font-code-sm text-code-sm ${main ? "text-primary font-semibold" : "text-on-surface-variant"}`}>
                    {formatPercent(result.weights[cls])}
                  </span>
                </figcaption>
              </figure>
            );
          })}
        </div>
      </div>
    </Card>
  );
}

export function MoEWorkspace() {
  const { options } = useBackend();
  const form = useRestorationForm();
  const { view, result, loading, error, run, preview } = useRestorationRun(form, api.moe);

  return (
    <WorkspaceLayout
      header={
        <WorkspaceHeader
          workspace="moe"
          icon="donut_large"
          title="Soft Mixture-of-Experts Restoration"
          tag="Task 3"
          subtitle="A gating network weights the identity branch and the three experts; the output is their weighted sum."
        />
      }
      input={<RestorationInputCard form={form} onRun={() => run()} onPreview={preview} loading={loading} />}
      results={
        <>
          <ModelStatusNotice workspace="moe" />
          <RequestError error={error} />
          <RestorationPanels view={view} selected={form.image.selected} loading={loading} outputBadge="MoE combined" />
          {result && <ExpertWeightsCard result={result} />}
          {result && (
            <DetailsCard
              items={[
                ...corruptionDetails(result),
                { label: "Model", value: result.model, icon: "dataset" },
                { label: "Main contributor", value: `${BRANCH_LABEL[result.dominant_branch]} (${formatPercent(result.weights[result.dominant_branch])})` },
                ...timingDetails(result.timing),
                ...sourceDetails(result, options),
              ]}
              actions={
                <DownloadButton
                  href={result.output}
                  filename={downloadFilename(result.source.name, "restored-moe")}
                  label="Download result"
                />
              }
            />
          )}
          {result?.reference && <ErrorMapCard output={result.output} reference={result.reference} />}
        </>
      }
    />
  );
}
