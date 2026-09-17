import { cn } from "@/lib/cn";

// Shared micro-label: the numbered tag above each section ("02 · ROUTER").
export function SectionTag({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-center gap-3 font-mono text-[11px] tracking-[0.28em] uppercase", className)}>
      <span className="text-ember" aria-hidden>●</span>
      <span>{children}</span>
      <span className="h-px flex-1 bg-line" aria-hidden />
    </div>
  );
}

// Editorial display heading with a mono label and optional serif-italic drift.
export function Display({
  children,
  as = "h1",
  className,
}: {
  children: React.ReactNode;
  as?: "h1" | "h2" | "h3";
  className?: string;
}) {
  const Tag = as;
  return (
    <Tag className={cn("font-display font-semibold display-clamp", className)}>
      {children}
    </Tag>
  );
}

export function Emph({ children }: { children: React.ReactNode }) {
  return <em className="font-display italic font-normal text-ember">{children}</em>;
}