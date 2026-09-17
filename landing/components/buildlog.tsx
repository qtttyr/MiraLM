import { ROADMAP } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/cn";

const STATUS: Record<string, { dot: string; label: string; text: string }> = {
  done: { dot: "bg-ember", label: "DONE", text: "text-ink" },
  next: { dot: "bg-petrol", label: "IN TRAINING", text: "text-petrol" },
  open: { dot: "bg-linestrong", label: "OPEN", text: "text-ink3" },
};

export function BuildLog() {
  return (
    <section id="build" className="relative bg-paper2 py-24 sm:py-32">
      <div className="mx-auto max-w-[1200px] px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <SectionTag>05 · BUILD</SectionTag>
            <h2 className="mt-4 max-w-[15ch] font-display text-5xl font-semibold leading-[0.92] tracking-tight sm:text-6xl">
              Six weeks, <Emph>one notebook GPU.</Emph>
            </h2>
          </div>
          <p className="max-w-[38ch] text-sm leading-6 text-ink2">
            The build was serial on purpose: architecture, then data, then training.
            Each stage shipped with its own test suite before the next began.
          </p>
        </div>

        <div className="mt-14 grid gap-6 lg:grid-cols-[1fr_1.4fr]">
          {/* week tracker card */}
          <aside className="lg:sticky lg:top-28 lg:self-start">
            <Reveal frame className="rounded-[16px] border border-line bg-paper p-6">
              <div className="font-mono text-[10px] tracking-[0.28em] text-ink3 uppercase">RUN-STATE</div>
              <div className="mt-5 flex items-center gap-4">
                <span className="pulse-dot h-3 w-3 rounded-full bg-ember" aria-hidden />
                <div>
                  <div className="font-mono text-sm font-medium tracking-[0.16em] uppercase">Epoch of boxes</div>
                  <div className="font-mono text-[11px] text-ink3">5 tests × 100 passing · step 12,842</div>
                </div>
              </div>
              <div className="mt-5 h-1.5 w-full rounded-full bg-line">
                <div className="h-full w-[68%] rounded-full bg-ember" />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[9px] tracking-[0.2em] text-ink3 uppercase">
                <span>W1</span>
                <span className="text-ember">W4 · NOW</span>
                <span>W6</span>
              </div>
            </Reveal>
          </aside>

          {/* timeline */}
          <ol className="relative space-y-4">
            {ROADMAP.map((m, i) => {
              const s = STATUS[m.status];
              return (
                <li key={m.tag}>
                  <Reveal delay={i * 0.06}>
                    <div
                      className={cn(
                        "grid grid-cols-[70px_auto_1fr] items-center gap-4 rounded-[12px] border px-5 py-4 transition-colors",
                        m.status === "next" ? "border-petrol/40 bg-petrol/5" : "border-line bg-paper",
                      )}
                    >
                      <span className={cn("font-mono text-[11px] tracking-[0.24em]", m.status === "done" ? "text-ink3 line-through" : "text-ink")}>
                        {m.tag}
                      </span>
                      <span className="flex items-center gap-2 font-mono text-[9px] tracking-[0.24em] uppercase">
                        <span className={cn("h-[7px] w-[7px] rounded-full", s.dot, m.status === "next" && "pulse-dot")} aria-hidden />
                        <span className={s.text}>{s.label}</span>
                      </span>
                      <div className="text-right">
                        <div className={cn("font-display text-lg font-semibold tracking-tight", m.status === "done" ? "text-ink3 line-through decoration-ember" : "text-ink")}>
                          {m.title}
                        </div>
                        <div className="text-xs text-ink3">{m.body}</div>
                      </div>
                    </div>
                  </Reveal>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}