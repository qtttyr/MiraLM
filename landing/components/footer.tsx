import { GITHUB, WANDB } from "@/lib/content";

const LINKS = [
  { label: "CODE", href: GITHUB, ext: true, note: "github.com/qtttyr" },
  { label: "RUNS", href: WANDB, ext: true, note: "wandb.ai · public" },
  { label: "BENCH", href: "#bench", ext: false, note: "results / eval_mira.json" },
  { label: "BUILD", href: "#build", ext: false, note: "timeline" },
];

export function Footer() {
  return (
    <footer className="relative mt-0 bg-slab text-cream">
      {/* hairline top */}
      <div className="h-px w-full bg-slabline" aria-hidden />

      <div className="mx-auto max-w-[1200px] px-6 pt-16 pb-10">
        <div className="grid gap-12 md:grid-cols-[1.4fr_1fr]">
          <div>
            <p className="font-mono text-[11px] tracking-[0.3em] text-creamdim uppercase">MIRALM · 47,640,968 PARAMETERS</p>
            <h2 className="mt-6 max-w-[18ch] font-display text-5xl leading-[0.94] font-semibold sm:text-7xl">
              Built by hand.
              <span className="block font-display italic font-normal text-emberglow">Zero shortcuts.</span>
            </h2>
            <p className="mt-6 max-w-[44ch] text-sm leading-6 text-creamdim">
              A single-machine experiment in what a small, well-routed model can do.
              No pretrained weights, no distillation — one notebook GPU, a twinkling
              budget, and the patience to watch it learn.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-x-8 gap-y-2 content-start">
            {LINKS.map((l) => (
              <a
                key={l.label}
                href={l.href}
                {...(l.ext ? { target: "_blank", rel: "noreferrer" } : {})}
                className="group border-b border-slabline py-3 font-mono text-sm tracking-[0.16em] text-cream transition-colors hover:text-emberglow"
              >
                {l.label} {l.ext ? "↗" : "↓"}
                <span className="block font-sans text-[11px] tracking-normal text-creamdim group-hover:text-cream">
                  {l.note}
                </span>
              </a>
            ))}
          </div>
        </div>

        <div className="mt-16 flex flex-col gap-3 border-t border-slabline pt-6 font-mono text-[10px] tracking-[0.2em] text-creamdim uppercase sm:flex-row sm:items-center sm:justify-between">
          <span>GIBC V2 · TRACK 01 — FOUNDATIONAL MODEL</span>
          <span>BUILT WITH AI-ASSISTED TOOLING · ALL WEIGHTS LEARNED FROM SCRATCH</span>
        </div>

        {/* giant hollow wordmark */}
        <div aria-hidden className="pointer-events-none mt-8 select-none overflow-hidden">
          <div className="text-hollow font-mono text-[18vw] leading-[0.8] tracking-tight text-center">
            MIRALM
          </div>
        </div>
      </div>
    </footer>
  );
}