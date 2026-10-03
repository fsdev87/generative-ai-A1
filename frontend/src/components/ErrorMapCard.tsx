// "Reconstruction delta map" card (Stitch universal screen). The design's numbers (L1 residual,
// LPIPS, "0 ALLOC DROPS") were placeholders; here the absolute error map |output - original| and
// its mean / max are computed in the browser from the two PNGs the API returned.
import { useEffect, useState } from "react";
import { absoluteErrorMap } from "../lib/errorMap";
import { Icon } from "./Icon";

const GAIN = 4;

function loadPixels(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) return reject(new Error("Canvas is not available"));
      context.drawImage(image, 0, 0);
      resolve(context.getImageData(0, 0, canvas.width, canvas.height));
    };
    image.onerror = () => reject(new Error("The image could not be decoded"));
    image.src = url;
  });
}

interface MapResult {
  url: string;
  meanAbsError: number;
  maxAbsError: number;
}

export function ErrorMapCard({ output, reference }: { output: string; reference: string }) {
  const [result, setResult] = useState<MapResult | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setResult(null);
    setFailed(false);
    Promise.all([loadPixels(output), loadPixels(reference)])
      .then(([out, ref]) => {
        if (out.width !== ref.width || out.height !== ref.height) throw new Error("Different image sizes");
        const map = absoluteErrorMap(out.data, ref.data, GAIN);
        const canvas = document.createElement("canvas");
        canvas.width = out.width;
        canvas.height = out.height;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Canvas is not available");
        context.putImageData(new ImageData(map.pixels, out.width, out.height), 0, 0);
        if (active) setResult({ url: canvas.toDataURL("image/png"), meanAbsError: map.meanAbsError, maxAbsError: map.maxAbsError });
      })
      .catch(() => active && setFailed(true));
    return () => {
      active = false;
    };
  }, [output, reference]);

  return (
    <div className="bg-surface-container-low rounded-xl p-space-md flex flex-col sm:flex-row items-center justify-between gap-space-md">
      <div className="flex items-center gap-space-md">
        <div className="w-10 h-10 rounded-lg bg-surface-container-lowest flex items-center justify-center text-primary shadow-sm shrink-0">
          <Icon name="ssid_chart" className="text-[20px]" />
        </div>
        <div>
          <span className="font-label-md text-label-md text-on-surface block">Reconstruction error map</span>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            |restored output − original| averaged over RGB, brightness ×{GAIN} (computed in the browser).
          </span>
          {result && (
            <span className="block font-code-sm text-code-sm text-on-surface mt-0.5">
              Mean absolute error (L1): {result.meanAbsError.toFixed(4)} · max: {result.maxAbsError.toFixed(3)}
            </span>
          )}
          {failed && (
            <span className="block font-body-sm text-body-sm text-error">The error map could not be computed in this browser.</span>
          )}
        </div>
      </div>
      {result && (
        <img
          src={result.url}
          alt="Absolute error map between the restored output and the original"
          className="w-28 h-28 rounded-lg shrink-0 bg-inverse-surface [image-rendering:pixelated]"
        />
      )}
    </div>
  );
}
