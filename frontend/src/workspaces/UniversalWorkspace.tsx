// Task 1 workspace, "Universal Restoration" (design/stitch/universal.html): one denoising
// autoencoder (udae.onnx) restores any of the corruptions without being told which one.
import { api } from "../api/client";
import { DetailsCard, DownloadButton } from "../components/DetailsCard";
import { ErrorMapCard } from "../components/ErrorMapCard";
import { ModelStatusNotice } from "../components/ModelStatusNotice";
import { RequestError } from "../components/RequestError";
import { WorkspaceHeader } from "../components/WorkspaceHeader";
import { WorkspaceLayout } from "../components/WorkspaceLayout";
import { useBackend } from "../hooks/backend";
import { downloadFilename } from "../lib/download";
import { corruptionDetails, sourceDetails, timingDetails } from "./restoration/details";
import { RestorationInputCard } from "./restoration/RestorationInputCard";
import { RestorationPanels } from "./restoration/RestorationPanels";
import { useRestorationForm } from "./restoration/useRestorationForm";
import { useRestorationRun } from "./restoration/useRestorationRun";

export function UniversalWorkspace() {
  const { options } = useBackend();
  const form = useRestorationForm();
  const { view, result, loading, error, run, preview } = useRestorationRun(form, api.universal);

  return (
    <WorkspaceLayout
      header={
        <WorkspaceHeader
          workspace="universal"
          icon="auto_fix_high"
          title="Universal Restoration"
          tag="Task 1"
          subtitle="One convolutional denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images without being told the corruption."
        />
      }
      input={<RestorationInputCard form={form} onRun={() => run()} onPreview={preview} loading={loading} />}
      results={
        <>
          <ModelStatusNotice workspace="universal" />
          <RequestError error={error} />
          <RestorationPanels view={view} selected={form.image.selected} loading={loading} outputBadge="Restored" />
          {result && (
            <DetailsCard
              items={[
                ...corruptionDetails(result),
                { label: "Model", value: result.model, icon: "dataset" },
                ...timingDetails(result.timing),
                ...sourceDetails(result, options),
              ]}
              actions={
                <DownloadButton
                  href={result.output}
                  filename={downloadFilename(result.source.name, "restored-universal")}
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
