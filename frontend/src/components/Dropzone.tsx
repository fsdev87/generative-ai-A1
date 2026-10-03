// Upload dropzone: drag and drop or click to choose an image file. The file type is checked by the
// backend (it decodes the image and answers 415 otherwise); only the size is checked here, so a
// too large file is not uploaded at all.
import { useRef, useState, type DragEvent } from "react";
import { Icon } from "./Icon";

export const ACCEPTED_TYPES = "image/png,image/jpeg,image/webp,image/bmp";

export function Dropzone({
  onFile,
  onReject,
  maxMb,
  selectedName,
  icon = "cloud_upload",
  disabled = false,
}: {
  onFile: (file: File) => void;
  onReject: (message: string) => void;
  /** Upload limit from GET /api/options; checked before uploading. */
  maxMb?: number;
  selectedName?: string;
  icon?: string;
  disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (file: File | undefined) => {
    if (!file) return;
    if (maxMb !== undefined && file.size > maxMb * 1024 * 1024) {
      const size = (file.size / (1024 * 1024)).toFixed(1);
      onReject(`${file.name} is ${size} MB; images may be at most ${maxMb} MB.`);
      return;
    }
    onFile(file);
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) accept(event.dataTransfer.files[0]);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`w-full rounded-lg transition-colors ${
        dragging ? "bg-surface-container ring-2 ring-primary" : "bg-surface-container-low hover:bg-surface-container"
      }`}
    >
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        className="group w-full flex flex-col items-center justify-center p-space-lg text-center cursor-pointer disabled:cursor-not-allowed"
      >
        <span className="w-10 h-10 rounded-full bg-surface-container-lowest flex items-center justify-center text-primary group-hover:scale-105 transition-transform mb-space-sm shadow-sm">
          <Icon name={icon} className="text-[20px]" />
        </span>
        <span className="font-label-md text-label-md text-on-surface mb-0.5">
          {dragging ? "Drop the image here" : "Drop an image or click to upload"}
        </span>
        <span className="font-body-sm text-body-sm text-on-surface-variant">
          PNG, JPG, WEBP or BMP{maxMb !== undefined ? ` up to ${maxMb} MB` : ""}
        </span>
        {selectedName && (
          <span className="mt-space-xs font-code-sm text-code-sm text-primary truncate max-w-full">
            Selected: {selectedName}
          </span>
        )}
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPTED_TYPES}
        aria-label="Upload image"
        className="sr-only"
        tabIndex={-1}
        disabled={disabled}
        onChange={(event) => {
          accept(event.target.files?.[0]);
          event.target.value = ""; // allow choosing the same file again
        }}
      />
    </div>
  );
}
