"use client";

// Benchmark table: ours vs a 117M GPT-2 and a Pythia-70M baseline.
// Values fill in (and bars sweep) once real numbers are measured.

import { BENCHMARKS } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { motion } from "motion/react";
import { cn } from "@/lib/cn";

function Cell({ v, sub }: { v: string | null; sub?: string }) {
  if (v === null)
    return <span className="font-mono text-sm text-ink3">—</span>;
  return (
    <span>
      <span className="font-mono text-base font-medium text-ink">{v}</span>
      {sub && <span className="block font-mono text-[9px] tracking-[0.14em] text-ink3 uppercase">{sub}</span>}
    </span>
  );
}

export function Benchmarks() {
  const measured = BENCHMARKS.filter((b) => b.ours.measured).length;

  return (
    <section id="bench" className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-[1200px] px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <SectionTag>03 · BENCH</SectionTag>
            <h2 className="mt-4 max-w-[16ch] font-display text-5xl font-semibold leading-[0.92] tracking-tight sm:text-6xl">
              Five gates stand between it and <Emph>competence.</Emph>
            </h2>
          </div>
          <p className="max-w-[38ch] text-sm leading-6 text-ink2">
            The five mandatory GIBC tasks, run through <span className="font-mono text-[12px]">lm_eval</span>.
            Baselines shown for context — a 117M GPT-2 and a 70M Pythia.
          </p>
        </div>

        <div className="mt-12 overflow-hidden rounded-[16px] border border-line bg-paper">
          {/* header */}
          <div className="grid grid-cols-[1.2fr_1fr_0.7fr_0.7fr_0.7fr_1.2fr] items-center gap-4 border-b border-line bg-paper2 px-5 py-3 font-mono text-[9px] tracking-[0.26em] text-ink3 uppercase">
            <span>TASK / METRIC</span>
            <span>MIRALM · 47M</span>
            <span>GPT-2 · 117M</span>
            <span>PYTHIA · 70M</span>
            <span className="hidden sm:block">Δ</span>
            <span className="hidden sm:block">NOTES</span>
          </div>

          {BENCHMARKS.map((b, i) => {
            const ours = b.ours.value ? parseFloat(b.ours.value) : null;
            const ref = b.gpt2 ? parseFloat(b.gpt2) : null;
            const delta = ours !== null && ref !== null ? ((ours - ref) / ref) * 100 : null;
            return (
              <div
                key={b.id}
                className={cn(
                  "grid grid-cols-1 gap-3 px-5 py-5 sm:grid-cols-[1.2fr_1fr_0.7fr_0.7fr_0.7fr_1.2fr] sm:items-center sm:gap-4",
                  i !== BENCHMARKS.length - 1 && "border-b border-line",
                  b.ours.measured && "bg-ember/5",
                )}
              >
                <div>
                  <div className="font-mono text-base font-medium uppercase tracking-[0.12em]">{b.task}</div>
                  <div className="mt-0.5 font-mono text-[10px] tracking-[0.2em] text-ink3 uppercase">{b.metric}</div>
                </div>

                <div className="relative">
                  <Cell v={b.ours.value} sub={b.ours.measured ? "measured" : "pending"} />
                  {ours !== null && (
                    <motion.div
                      aria-hidden
                      className="pointer-events-none absolute -inset-x-2 -inset-y-1 rounded-md bg-ember/10"
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: [0, 1, 1] }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8 }}
                    />
                  )}
                </div>

                <Cell v={b.gpt2} sub="baseline" />
                <Cell v={b.pythia} sub="baseline" />

                <span className={cn("font-mono text-xs", delta === null ? "text-ink3" : delta >= 0 ? "text-petrol" : "text-emberdeep")}>
                  {delta === null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`}
                </span>

                <span className="text-xs leading-5 text-ink2">{b.note}</span>
              </div>
            );
          })}
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-x-8 gap-y-2 font-mono text-[10px] tracking-[0.22em] text-ink3 uppercase">
          <span className="flex items-center gap-2">
            <span className="h-[6px] w-[6px] rounded-full bg-ember" aria-hidden /> {measured}/{BENCHMARKS.length} MEASURED
          </span>
          <span className="flex items-center gap-2">
            <span className="h-[6px] w-[6px] rounded-full bg-linestrong" aria-hidden /> Δ = MIRALM vs GPT-2
          </span>
          <span>WIKI-103 REPORTED AS WORD PPL · LOWER IS BETTER</span>
        </div>
      </div>
    </section>
  );
}