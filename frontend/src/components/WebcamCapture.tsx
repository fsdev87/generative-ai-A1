// Live webcam preview with a "Capture photo" button. The captured frame becomes a PNG File and
// is sent like an upload. The camera is released when the component closes.
import { useEffect, useRef, useState } from "react";
import { cameraErrorMessage, cameraUnavailableReason } from "../lib/camera";
import { Alert } from "./Alert";
import { Icon } from "./Icon";

export function WebcamCapture({ onCapture, onClose }: { onCapture: (file: File) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(() => cameraUnavailableReason());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (cameraUnavailableReason()) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        if (video.current) {
          video.current.srcObject = s;
          video.current.play().catch(() => undefined);
        }
      })
      .catch((e: unknown) => !cancelled && setError(cameraErrorMessage(e)));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return setError("The frame could not be captured in this browser. Upload a photo instead.");
    // The preview is mirrored like a selfie camera; capture exactly what the user sees.
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(v, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return setError("The frame could not be captured. Try again or upload a photo.");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      onCapture(new File([blob], `webcam-${stamp}.png`, { type: "image/png" }));
    }, "image/png");
  };

  return (
    <div className="flex flex-col gap-space-sm">
      {error ? (
        <Alert tone="error" title="Camera not available">
          {error}
        </Alert>
      ) : (
        <div className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-inverse-surface">
          <video
            ref={video}
            muted
            playsInline
            onLoadedData={() => setReady(true)}
            aria-label="Webcam preview"
            className="w-full h-full object-cover -scale-x-100"
          />
          {!ready && (
            <span className="absolute inset-0 flex items-center justify-center text-inverse-on-surface font-body-sm text-body-sm">
              Starting the camera… allow access if the browser asks.
            </span>
          )}
        </div>
      )}
      <div className="grid grid-cols-2 gap-space-sm">
        <button
          type="button"
          onClick={capture}
          disabled={!ready || error !== null}
          className="flex items-center justify-center gap-space-xs py-2 px-space-md rounded-lg bg-primary hover:bg-primary-container text-on-primary font-label-md text-label-md transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Icon name="photo_camera" className="text-[18px]" />
          Capture photo
        </button>
        <button
          type="button"
          onClick={onClose}
          className="flex items-center justify-center gap-space-xs py-2 px-space-md rounded-lg bg-surface-container-low hover:bg-surface-container text-on-surface font-label-md text-label-md transition-colors"
        >
          <Icon name="close" className="text-[18px]" />
          Close camera
        </button>
      </div>
    </div>
  );
}
