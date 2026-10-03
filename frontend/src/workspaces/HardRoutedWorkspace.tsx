// Task 2 workspace, "Hard-Routed Restoration" (design/stitch/hard_routed.html): classifier.onnx
// predicts the corruption, then exactly one specialist restores the image (clean = identity
// bypass). Oracle routing uses the true corruption instead of the prediction.
import { useState, type ReactNode } from "react";
import { api, oracleAvailable } from "../api/client";
import type { HardResponse, RoutingMode } from "../api/types";
import { Alert } from "../components/Alert";
import { Card, CardTitle } from "../components/Card";
import { DetailsCard, DownloadButton } from "../components/DetailsCard";
import { ErrorMapCard } from "../components/ErrorMapCard";
import { Icon } from "../components/Icon";
import { ModelStatusNotice } from "../components/ModelStatusNotice";
import { RequestError } from "../components/RequestError";
import { ScoreBars } from "../components/ScoreBars";
import { WorkspaceHeader } from "../components/WorkspaceHeader";
import { WorkspaceLayout } from "../components/WorkspaceLayout";
import { useBackend } from "../hooks/backend";
import { downloadFilename } from "../lib/download";
import { CLASS_LABEL, EXPERT_LABEL, entropy, formatMs, formatPercent, topMargin } from "../lib/format";
import { corruptionDetails, sourceDetails } from "./restoration/details";
import { RestorationInputCard } from "./restoration/RestorationInputCard";
import { RestorationPanels } from "./restoration/RestorationPanels";
import { useRestorationForm } from "./restoration/useRestorationForm";
import { useRestorationRun } from "./restoration/useRestorationRun";

export const ORACLE_UNAVAILABLE =
  "Oracle routing needs the true corruption, which is known only when the server applies it: choose a corruption, or pick a sample with None (sent as clean).";

export function RoutingToggle({
  mode,
  onChange,
  oracleAllowed,
}: {
  mode: RoutingMode;
  onChange: (mode: RoutingMode) => void;
  oracleAllowed: boolean;
}) {
  const button = (value: RoutingMode, label: string, disabled = false) => (
    <button
      type="button"
      role="radio"
      aria-checked={mode === value}
      disabled={disabled}
      title={disabled ? ORACLE_UNAVAILABLE : undefined}
      onClick={() => onChange(value)}
      className={`px-space-md py-1 rounded-full font-label-sm text-label-sm transition-all ${
        mode === value ? "bg-primary text-on-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface"
      } ${disabled ? "opacity-50 cursor-not-allowed hover:text-on-surface-variant" : ""}`}
    >
      {label}
    </button>
  );
  return (
    <div className="flex flex-col gap-space-xs p-space-md rounded-lg bg-surface-container-low">
      <div className="flex items-center justify-between gap-space-sm">
        <div className="flex flex-col">
          <span className="font-label-md text-label-md text-on-surface">Routing</span>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            {mode === "oracle" ? "The true corruption chooses the expert" : "The classifier's prediction chooses the expert"}
          </span>
        </div>
        <div role="radiogroup" aria-label="Routing mode" className="inline-flex p-0.5 bg-surface-container rounded-full shrink-0">
          {button("predicted", "Predicted")}
          {button("oracle", "Oracle", !oracleAllowed)}
        </div>
      </div>
      {!oracleAllowed && <p className="font-body-sm text-body-sm text-on-surface-variant">{ORACLE_UNAVAILABLE}</p>}
    </div>
  );
}

function Badge({ icon, children }: { icon: string; children: ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 px-space-md py-1.5 rounded-lg bg-surface-container-high text-primary font-label-md text-label-md">
      <Icon name={icon} className="text-[16px]" />
      <span>{children}</span>
    </div>
  );
}

/** Explains how the expert was chosen, including classifier mistakes. */
function RoutingNote({ result }: { result: HardResponse }) {
  const { predicted_class: predicted, true_class: truth, routing_mode: mode, routed_class: routed } = result;
  if (truth && predicted !== truth) {
    return mode === "oracle" ? (
      <Alert tone="info" title="Oracle routing">
        The true class {CLASS_LABEL[truth]} chose the expert; the classifier predicted {CLASS_LABEL[predicted]}.
      </Alert>
    ) : (
      <Alert tone="warning" title="Misrouted">
        The classifier predicted {CLASS_LABEL[predicted]}, but the true corruption is {CLASS_LABEL[truth]}, so the{" "}
        {EXPERT_LABEL[result.selected_expert]} was used. Try Oracle routing to compare.
      </Alert>
    );
  }
  if (routed === "clean") {
    return (
      <Alert tone="info" title="Identity bypass">
        The input was routed as clean, so no expert ran and the output equals the input.
      </Alert>
    );
  }
  return null;
}

function ClassifierCard({ result }: { result: HardResponse }) {
  const p = result.probabilities;
  const truthDiffers = result.true_class !== null && result.true_class !== result.predicted_class;
  return (
    <Card className="p-space-lg flex flex-col gap-space-lg">
      <CardTitle
        as="h3"
        icon="account_tree"
        title="Classifier"
        subtitle="Softmax probabilities of classifier.onnx; the highest one is the prediction"
        aside={
          <div className="flex flex-wrap items-center gap-space-xs">
            <Badge icon="radar">
              Predicted: <strong className="font-semibold">{CLASS_LABEL[result.predicted_class]}</strong>
            </Badge>
            <Badge icon="psychology">
              Expert used: <strong className="font-semibold">{EXPERT_LABEL[result.selected_expert]}</strong>
            </Badge>
            {result.true_class && (
              <Badge icon="verified">
                True: <strong className="font-semibold">{CLASS_LABEL[result.true_class]}</strong>
              </Badge>
            )}
          </div>
        }
      />
      <ScoreBars
        label="Classifier probabilities"
        variant="grid"
        scores={p}
        labels={CLASS_LABEL}
        highlight={result.predicted_class}
        highlightTag="Predicted"
        marker={truthDiffers && result.true_class ? { cls: result.true_class, tag: "True class" } : undefined}
      />
      <div className="flex flex-wrap items-center justify-between gap-space-sm font-code-sm text-code-sm text-on-surface-variant">
        <span>
          Confidence: <strong className="text-on-surface">{formatPercent(p[result.predicted_class])}</strong> (margin{" "}
          {formatPercent(topMargin(p))} over the runner-up)
        </span>
        <span>
          Entropy: <strong className="text-on-surface">{entropy(p).toFixed(3)} nats</strong> (uniform = 1.386)
        </span>
      </div>
      <RoutingNote result={result} />
    </Card>
  );
}

function TimingBreakdown({ timing }: { timing: HardResponse["timing"] }) {
  const chip = "px-space-xs py-0.5 bg-surface-container rounded text-on-surface";
  return (
    <span className="flex flex-wrap items-center gap-space-sm">
      <span className={chip}>
        Classifier: <span className="font-semibold text-primary">{formatMs(timing.classifier_ms)}</span>
      </span>
      <span className="text-on-surface-variant">+</span>
      <span className={chip}>
        Expert: <span className="font-semibold text-primary">{formatMs(timing.expert_ms)}</span>
      </span>
      <span className="text-on-surface-variant">=</span>
      <span className="px-space-xs py-0.5 bg-surface-container-highest rounded text-on-surface font-bold">
        Inference: {formatMs(timing.inference_ms)}
      </span>
      <span className="text-on-surface-variant">· server total {formatMs(timing.total_ms)}</span>
    </span>
  );
}

export function HardRoutedWorkspace() {
  const { options } = useBackend();
  const form = useRestorationForm();
  const { view, result, loading, error, run, preview } = useRestorationRun(form, api.hard);
  const [routingMode, setRoutingMode] = useState<RoutingMode>("predicted");
  const oracleAllowed = oracleAvailable(form.image.selected?.source.kind ?? null, form.choice);
  const effectiveMode: RoutingMode = oracleAllowed ? routingMode : "predicted";

  const expertFile = result && result.selected_expert !== "identity" ? `${result.selected_expert}.onnx` : null;

  return (
    <WorkspaceLayout
      header={
        <WorkspaceHeader
          workspace="hard"
          icon="account_tree"
          title="Hard-Routed Restoration"
          tag="Task 2"
          subtitle="A corruption classifier picks one specialist autoencoder per image; clean images bypass restoration."
        />
      }
      input={
        <RestorationInputCard form={form} onRun={() => run(effectiveMode)} onPreview={preview} loading={loading}>
          <RoutingToggle mode={effectiveMode} onChange={setRoutingMode} oracleAllowed={oracleAllowed} />
        </RestorationInputCard>
      }
      results={
        <>
          <ModelStatusNotice workspace="hard" />
          <RequestError error={error} />
          <RestorationPanels
            view={view}
            selected={form.image.selected}
            loading={loading}
            outputBadge={result?.selected_expert === "identity" ? "Identity bypass" : "Expert output"}
          />
          {result && <ClassifierCard result={result} />}
          {result && (
            <DetailsCard
              items={[
                ...corruptionDetails(result),
                { label: "Routing", value: result.routing_mode === "oracle" ? "Oracle (true class)" : "Predicted (classifier)" },
                {
                  label: "Models",
                  value: expertFile ? `classifier.onnx + ${expertFile}` : "classifier.onnx (no expert)",
                  icon: "dataset",
                },
                { label: "Inference time breakdown", value: <TimingBreakdown timing={result.timing} />, wide: true },
                ...sourceDetails(result, options),
              ]}
              actions={
                <DownloadButton
                  href={result.output}
                  filename={downloadFilename(result.source.name, `restored-hard-${result.routing_mode}`)}
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
