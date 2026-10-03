// File names for the "Download" buttons. The API returns PNG data URLs, which an <a download>
// link saves directly.

/** True for the PNG data URLs the backend returns. */
export function isPngDataUrl(value: string | null | undefined): value is string {
  return typeof value === "string" && value.startsWith("data:image/png;base64,");
}

/**
 * A safe download name from the source name and what the image is, e.g.
 * ("Abyssinian_12.jpg", "restored-universal") -> "Abyssinian_12_restored-universal.png".
 */
export function downloadFilename(sourceName: string, suffix: string): string {
  const stem = sourceName
    .replace(/^.*[\\/]/, "") // no directories
    .replace(/\.[^.]*$/, "") // no extension
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `${stem || "image"}_${suffix}.png`;
}
