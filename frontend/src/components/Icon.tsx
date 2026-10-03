// Material Symbols Outlined icon, self-hosted through the `material-symbols` npm package
// (same icon font and names as the Stitch design).
export function Icon({ name, className = "" }: { name: string; className?: string }) {
  return (
    <span aria-hidden="true" className={`material-symbols-outlined select-none ${className}`}>
      {name}
    </span>
  );
}
