"use client";

// Principles + a count-up facts strip (real, enforced numbers).
import { PRINCIPLES, PARAMS, ACTIVE, VOCAB, EXPERTS_N } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { Reveal } from "@/components/reveal";
import { useCountUp } from "@/hooks/use-count-up";
import { motion } from "motion/react";

function Fact({ target, label, format }: { target: number; label: string; format?: (n: number) => string }) {
  const { ref, text } = useCountUp(target, 1.6, format);
  return (
    <div ref={ref} className="group min-h-[72px] border-t-2 border-ink pt-3">
      <motion.span className="block truncate font-display text-[clamp(1.5rem,2.7vw,2.1rem)] font-semibold tracking-tight tabular-nums">
        {text}
      </motion.span>
      <div className="mt-2 font-mono text-[9px] leading-snug tracking-[0.22em] text-ink3 uppercase">{label}</div>
    </div>
  );
}

export function Principles() {
  return (
    <section className="relative bg-paper py-24 sm:py-32">
      <div className="mx-auto max-w-[1200px] px-6">
        <SectionTag>THE CONTRACT</SectionTag>

        <div className="mt-10 grid gap-14 lg:grid-cols-[1.25fr_1fr]">
          {/* principles list */}
          <div className="space-y-0">
            {PRINCIPLES.map((p, i) => (
              <Reveal key={p.n} className="group grid grid-cols-[52px_1fr] gap-5 border-b border-line py-7 first:pt-0">
                <span className="font-mono text-sm text-ink3 transition-colors group-hover:text-ember">{p.n}</span>
                <div>
                  <h3 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
                    {p.title}
                  </h3>
                  <p className="mt-2 max-w-[52ch] text-sm leading-6 text-ink2">{p.body}</p>
                  <span className="mt-3 inline-block font-mono text-[10px] tracking-[0.26em] text-ink3 uppercase">
                    {p.stat.value} · {p.stat.label}
                  </span>
                </div>
              </Reveal>
            ))}
          </div>

          {/* facts strip */}
          <div className="lg:sticky lg:top-28 lg:self-start">
            <div className="rounded-[16px] border border-line bg-paper2 p-6">
              <div className="flex items-center justify-between font-mono text-[10px] tracking-[0.28em] text-ink3 uppercase">
                <span>FACTS · ZERO WEIGHTS OUTSIDE THESE LINES</span>
              </div>
              <div className="mt-6 grid grid-cols-2 gap-x-6 gap-y-8">
                <Fact target={PARAMS} label="total parameters" format={(n) => n.toLocaleString("en-US")} />
                <Fact target={ACTIVE} label="active per token" format={(n) => n.toLocaleString("en-US")} />
                <Fact target={VOCAB} label="vocabulary size" format={(n) => n.toLocaleString("en-US")} />
                <Fact target={EXPERTS_N} label="routed experts" />
              </div>
            </div>

            <Reveal delay={0.15} className="mt-6 border-l-2 border-ember bg-slab px-6 py-6 text-cream">
              <p className="font-display text-xl leading-snug">
                No distilled weights. No warm starts. <Emph>Just a 384-wide net</Emph> and a
                patient training loop.
              </p>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}