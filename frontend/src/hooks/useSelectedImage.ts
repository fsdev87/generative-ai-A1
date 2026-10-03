// The image chosen in a workspace: an uploaded file (or webcam frame) or a bundled sample,
// with a URL to preview it before the request is sent.
import { useCallback, useEffect, useRef, useState } from "react";
import { sampleImageUrl, type ImageSource } from "../api/client";
import type { SampleInfo } from "../api/types";

export interface SelectedImage {
  source: ImageSource;
  /** Object URL of the file, or the sample's image URL. */
  previewUrl: string;
  name: string;
}

export function useSelectedImage() {
  const [selected, setSelected] = useState<SelectedImage | null>(null);
  const objectUrl = useRef<string | null>(null);

  // Replace the selection and release the object URL of the previous upload.
  const replace = useCallback((next: SelectedImage | null) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = next?.source.kind === "upload" ? next.previewUrl : null;
    setSelected(next);
  }, []);

  useEffect(() => () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
  }, []);

  const selectFile = useCallback(
    (file: File) =>
      replace({ source: { kind: "upload", file }, previewUrl: URL.createObjectURL(file), name: file.name }),
    [replace],
  );

  const selectSample = useCallback(
    (sample: SampleInfo) =>
      replace({ source: { kind: "sample", id: sample.id }, previewUrl: sampleImageUrl(sample), name: sample.name }),
    [replace],
  );

  const clear = useCallback(() => replace(null), [replace]);

  return { selected, selectFile, selectSample, clear };
}
