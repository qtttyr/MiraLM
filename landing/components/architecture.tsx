"use client";

// Pinned horizontal conveyor: the block stack scrolls sideways as you
// descend — EMBED → (Mamba∥Attention)×7 → GATE → MoE → HEAD.

import { useRef } from "react";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { STACK, PARAMS, ACTIVE, D_MODEL } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { cn } from "@/lib/cn";

const WIDTH = 420;

const kindStyle: Record<string, string> = {
  data: "border-linestrong text-ink",
  block: "border-slabline bg-slab text-cream",
  gate: "border-emberdeep bg-ember text-cream",
  moe: "border-petrol bg-petrol text-cream",
  head: "border-ink bg-ink text-cream",
};

export function Architecture() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const x = useSpring(useTransform(scrollYProgress, [0, 1], [0, -(STACK.length * WIDTH - 200)]), { stiffness: 80, damping: 28 });

  return (
    <section id="model" ref={ref} className="relative bg-paper">
      {/* pin the scroll area */}
      <div className="sticky top-0 flex h-[100svh] flex-col justify-center overflow-hidden">
        <div className="mx-auto mb-6 flex w-full max-w-[1200px] flex-wrap items-end justify-between gap-4 px-6">
          <div>
            <SectionTag>01 · MODEL</SectionTag>
            <h2 className="mt-4 max-w-[16ch] font-display text-5xl font-semibold leading-[0.92] tracking-tight sm:text-6xl">
              A single track, from token to <Emph>answer.</Emph>
            </h2>
          </div>
          <p className="max-w-[38ch] text-sm leading-6 text-ink2">
            Seven interleaved <em className="font-display italic">Mamba ∥ Attention</em> blocks feed a
            top-2 MoE. Every block shares the 50M budget — enforced by two gates.
          </p>
        </div>

        {/* conveyor */}
        <motion.div style={{ x }} className="flex w-max gap-6 pl-[max(24px,calc((100vw-1200px)/2+24px))] pr-24">
          {STACK.map((n, i) => (
            <div key={n.id} className="flex items-center gap-3">
              <div
                className={cn(
                  "group relative w-[420px] shrink-0 overflow-hidden rounded-[16px] border py-6 pl-6 pr-5",
                  kindStyle[n.kind],
                )}
              >
                <span aria-hidden className="absolute -right-3 -top-4 font-mono text-[90px] font-bold leading-none opacity-10 select-none">
                  {n.kind === "block" ? "∥" : n.kind === "moe" ? "×8" : n.id.toUpperCase()}
                </span>
                <span className="font-mono text-[10px] tracking-[0.3em] opacity-60 uppercase">
                  {String(i).padStart(2, "0")} · {n.kind.toUpperCase()}
                </span>
                <div className="mt-4 font-display text-4xl font-semibold tracking-tight">{n.name}</div>
                <div className="mt-1 font-mono text-xs tracking-[0.16em] opacity-70">{n.sub}</div>

                {n.kind === "block" && (
                  <div className="mt-5 flex gap-2 font-mono text-[9px] tracking-[0.18em] uppercase">
                    <span className="rounded-[4px] border border-current/30 px-2 py-1">MAMBA · SSM ∞</span>
                    <span className="rounded-[4px] border border-current/30 px-2 py-1">GQA · RoPE · SWIGLU</span>
                  </div>
                )}
                {n.kind === "moe" && (
                  <div className="mt-5 grid grid-cols-4 gap-1.5">
                    {Array.from({ length: 8 }).map((_, k) => (
                      <div
                        key={k}
                        className={cn(
                          "h-1.5 rounded-full",
                          k < 3 ? "bg-cream" : k < 6 ? "bg-cream/60" : "bg-cream/30",
                        )}
                      />
                    ))}
                  </div>
                )}
              </div>
              {/* arrow between nodes */}
              <div aria-hidden className={cn("w-6 text-center text-linestrong", i === STACK.length - 1 && "hidden")}>
                →
              </div>
            </div>
          ))}

          {/* terminus card */}
          <div className="flex w-[300px] shrink-0 flex-col justify-center border-l border-linestrong pl-8">
            <span className="font-mono text-[10px] tracking-[0.3em] text-ink3 uppercase">TOTAL</span>
            <div className="mt-3 font-display text-5xl font-semibold">
              {PARAMS.toLocaleString("en-US")}
            </div>
            <div className="mt-1 font-mono text-xs text-ink2">
              params — {ACTIVE.toLocaleString("en-US")} active per token · d={D_MODEL}
            </div>
            <p className="mt-4 text-sm leading-6 text-ink2">
              Every count is exact, projected and re-checked on the real model before a run is allowed to start.
            </p>
          </div>
        </motion.div>

        {/* scroll hint */}
        <div className="mx-auto mt-10 flex w-full max-w-[1200px] items-center gap-4 px-6 font-mono text-[10px] tracking-[0.3em] text-ink3 uppercase">
          <span aria-hidden className="h-px flex-1 bg-line" />
          <span>CONTINUE — THE TRACK KEEPS SPOOLING →</span>
          <span aria-hidden className="h-px flex-1 bg-line" />
        </div>
      </div>
    </section>
  );
}