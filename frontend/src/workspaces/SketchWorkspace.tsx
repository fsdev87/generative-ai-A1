// Task 4 workspace, "Face-to-Sketch Generator" (design/stitch/sketch.html): the conditional GAN
// generator (generator.onnx) turns a face photo into a sketch in FS2K style 1, 2 or 3.
import { useEffect, useState } from "react";
import { api, buildSketchForm } from "../api/client";
import type { SketchResponse } from "../api/types";
import { Card, Tag } from "../components/Card";
import { DetailsCard, DownloadButton } from "../components/DetailsCard";
import { Dropzone } from "../components/Dropzone";
import { Icon } from "../components/Icon";
import { ImagePanel } from "../components/ImagePanel";
import { ModelStatusNotice } from "../components/ModelStatusNotice";
import { RequestError } from "../components/RequestError";
import { SamplePicker } from "../components/SamplePicker";
import { SegmentedControl } from "../components/SegmentedControl";
import { Spinner } from "../components/Spinner";
import { WebcamCapture } from "../components/WebcamCapture";
import { WorkspaceHeader } from "../components/WorkspaceHeader";
import { WorkspaceLayout } from "../components/WorkspaceLayout";
import { useBackend, useSamples } from "../hooks/backend";
import { useRequest } from "../hooks/useRequest";
import { useSelectedImage } from "../hooks/useSelectedImage";
import { downloadFilename } from "../lib/download";
import { formatMs } from "../lib/format";

/** What the FS2K training sketches of each style look like (docs/fs2k_notes.md, section 5). */
const STYLE_NOTES: Record<number, string> = {
  1: "In FS2K, style 1 sketches are light contour drawings.",
  2: "In FS2K, style 2 sketches have dense, dark strokes.",
  3: "In FS2K, style 3 sketches use heavy hatching.",
};

export function SketchWorkspace({ active }: { active: boolean }) {
  const { options } = useBackend();
  const { samples, error: samplesError } = useSamples("faces");
  const image = useSelectedImage();
  const request = useRequest<SketchResponse>();
  const { reset } = request;
  const [style, setStyle] = useState(2); // the design opens with Style 2
  const [webcamOpen, setWebcamOpen] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const selected = image.selected;
  const result = request.data;
  const styles = options?.styles ?? [1, 2, 3];

  useEffect(() => reset(), [selected, reset]);
  // Release the camera when the user switches to another workspace.
  useEffect(() => {
    if (!active) setWebcamOpen(false);
  }, [active]);

  const generate = () => {
    if (!selected) return;
    const body = buildSketchForm(selected.source, style);
    void request.run((signal) => api.sketch(body, signal));
  };

  const resetAll = () => {
    image.clear();
    setWebcamOpen(false);
    setUploadError(null);
    setStyle(2);
  };

  const status = request.loading ? "Generating…" : request.error ? "Failed" : result ? "Completed" : "Ready";

  return (
    <WorkspaceLayout
      header={
        <WorkspaceHeader
          workspace="sketch"
          icon="draw"
          title="Face-to-Sketch Generator"
          tag="Task 4"
          subtitle="A style-conditioned conditional GAN (U-Net generator) turns a face photo into a sketch."
          actions={
            <button
              type="button"
              onClick={resetAll}
              className="flex items-center gap-space-xs px-space-md py-1.5 rounded-lg bg-surface-container-lowest text-on-surface-variant font-label-md text-label-md hover:bg-surface-container transition-colors shadow-sm"
            >
              <Icon name="restart_alt" className="text-[18px]" />
              Reset
            </button>
          }
        />
      }
      input={
        <>
          <Card className="p-space-lg flex flex-col gap-space-md">
            <div className="flex items-center justify-between">
              <h2 className="font-headline-sm text-headline-sm text-on-surface font-semibold">Photo</h2>
              <Tag>{selected ? (selected.source.kind === "sample" ? "Sample" : "Upload") : "Source input"}</Tag>
            </div>
            {webcamOpen ? (
              <WebcamCapture
                onClose={() => setWebcamOpen(false)}
                onCapture={(file) => {
                  image.selectFile(file);
                  setWebcamOpen(false);
                }}
              />
            ) : (
              <>
                <Dropzone
                  maxMb={options?.max_upload_mb}
                  selectedName={selected?.source.kind === "upload" ? selected.name : undefined}
                  onFile={(file) => {
                    setUploadError(null);
                    image.selectFile(file);
                  }}
                  onReject={setUploadError}
                />
                {uploadError && (
                  <p role="alert" className="font-body-sm text-body-sm text-error">
                    {uploadError}
                  </p>
                )}
                <button
                  type="button"
                  onClick={() => setWebcamOpen(true)}
                  className="w-full flex items-center justify-center gap-space-sm py-2 px-space-md rounded-lg bg-surface-container-low hover:bg-surface-container text-on-surface font-label-md text-label-md transition-colors shadow-sm"
                >
                  <Icon name="photo_camera" className="text-[18px] text-on-surface-variant" />
                  Use webcam
                </button>
              </>
            )}
            {samples?.length !== 0 && (
              <SamplePicker
                samples={samples}
                error={samplesError}
                selectedId={selected?.source.kind === "sample" ? selected.source.id : null}
                onSelect={(sample) => {
                  setUploadError(null);
                  image.selectSample(sample);
                }}
              />
            )}
            <div className="pt-space-xs flex flex-col gap-space-xs">
              <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant font-semibold">
                Sketch style
              </span>
              <SegmentedControl
                label="Sketch style"
                variant="primary"
                value={style}
                onChange={setStyle}
                segments={styles.map((value) => ({ value, label: `Style ${value}` }))}
              />
              {STYLE_NOTES[style] && (
                <span className="font-body-sm text-body-sm text-on-surface-variant mt-1">{STYLE_NOTES[style]}</span>
              )}
            </div>
            <button
              type="button"
              onClick={generate}
              disabled={!selected || request.loading}
              className="w-full flex items-center justify-center gap-space-sm py-2.5 px-space-lg rounded-lg bg-primary hover:bg-primary-container text-on-primary font-headline-sm text-headline-sm transition-all shadow-sm disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-primary"
            >
              {request.loading ? <Spinner className="w-5 h-5" light /> : <Icon name="auto_fix_high" className="text-[20px]" />}
              <span>{request.loading ? "Generating…" : "Generate sketch"}</span>
            </button>
            {!selected && (
              <p className="font-body-sm text-body-sm text-on-surface-variant text-center">
                Upload a face photo, use the webcam or pick a sample first.
              </p>
            )}
          </Card>
          <Card className="p-space-md flex items-center gap-space-md">
            <div className="w-9 h-9 rounded-lg bg-surface-container-high flex items-center justify-center text-primary shrink-0">
              <Icon name="layers" className="text-[20px]" />
            </div>
            <div className="min-w-0">
              <p className="font-label-md text-label-md font-medium text-on-surface">Preprocessing as in training</p>
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                The server crops the photo to a centred square and resizes it to{" "}
                {options ? `${options.image_size} × ${options.image_size}` : "the model size"}; the style is passed to
                the generator as a learned embedding.
              </p>
            </div>
          </Card>
        </>
      }
      results={
        <>
          <ModelStatusNotice workspace="sketch" />
          <RequestError error={request.error} />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-space-lg">
            <ImagePanel
              title="Photo"
              icon="person"
              badge={result ? "Model input" : selected ? "Selected" : "Input frame"}
              src={result?.photo ?? selected?.previewUrl}
              alt={result ? "Cropped and resized photo given to the generator" : "Selected photo"}
              chip={result && options ? `${options.image_size} × ${options.image_size}` : undefined}
              placeholder="Upload a face photo, use the webcam or pick a sample."
              footer={selected ? <span className="truncate">{selected.name}</span> : undefined}
            />
            <ImagePanel
              title="Generated sketch"
              icon="draw"
              highlight
              badge={status}
              badgeTone={request.error ? "error" : result ? "primary" : "neutral"}
              src={result?.sketch}
              alt={result ? `Generated sketch in style ${result.style}` : ""}
              chip={result && `${result.sketch_channels === 1 ? "Greyscale" : "RGB"} · Style ${result.style}`}
              loading={request.loading}
              loadingText="Generating the sketch…"
              placeholder="Choose a style and press Generate sketch."
            />
          </div>
          {result && (
            <DetailsCard
              items={[
                { label: "Style", value: `Style ${result.style} (index ${result.style_index})` },
                { label: "Model", value: result.model, icon: "dataset" },
                { label: "Inference time", value: formatMs(result.timing.inference_ms), icon: "timer", accent: true },
                { label: "Server time", value: formatMs(result.timing.total_ms), icon: "schedule" },
                {
                  label: "Source",
                  value: `${result.source.kind === "sample" ? "Sample" : "Upload"} · ${result.source.name} (${result.source.format}, ${result.source.width} × ${result.source.height})`,
                },
                ...(options ? [{ label: "Image size", value: `${options.image_size} × ${options.image_size}` }] : []),
                { label: "Output", value: result.sketch_channels === 1 ? "1 channel (greyscale)" : `${result.sketch_channels} channels (RGB)` },
              ]}
              actions={
                <DownloadButton
                  variant="primary"
                  href={result.sketch}
                  filename={downloadFilename(result.source.name, `sketch-style${result.style}`)}
                  label="Download sketch"
                />
              }
            />
          )}
        </>
      }
    />
  );
}
