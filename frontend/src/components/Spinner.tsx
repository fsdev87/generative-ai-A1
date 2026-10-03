// Loading spinner (Stitch sketch screen: border-2 border-primary border-t-transparent animate-spin).
export function Spinner({ className = "w-8 h-8", light = false }: { className?: string; light?: boolean }) {
  return (
    <span
      role="presentation"
      className={`inline-block rounded-full border-2 border-t-transparent animate-spin ${
        light ? "border-on-primary" : "border-primary"
      } ${className}`}
    />
  );
}
