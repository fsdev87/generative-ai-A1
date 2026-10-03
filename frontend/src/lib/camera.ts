// Webcam availability checks and readable messages for getUserMedia failures.

/** Why the camera cannot be used in this page, or null when getUserMedia can be tried. */
export function cameraUnavailableReason(): string | null {
  if (!window.isSecureContext) {
    return "The camera only works on a secure page: open the app at http://localhost:5173 (or over https). You can still upload a photo.";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser does not support camera access. Upload a photo instead.";
  }
  return null;
}

/** Message for an error thrown by navigator.mediaDevices.getUserMedia. */
export function cameraErrorMessage(error: unknown): string {
  const name = error instanceof DOMException || error instanceof Error ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access in the browser's address bar, or upload a photo instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect a camera or upload a photo instead.";
    case "NotReadableError":
    case "AbortError":
      return "The camera is in use by another application or could not be started. Close the other application or upload a photo.";
    default:
      return "The camera could not be started. Upload a photo instead.";
  }
}
