"use client";

// Hero: editorial headline + the dark oscilloscope slab.
// The slab tilts and the headline parallaxes as you scroll away.

import { useRef } from "react";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { Oscilloscope } from "@/components/oscilloscope";
import { Emph } from "@/components/typography";
import { PARAMS } from "@/lib/content";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

const LINES = [
  { body: <>Forty-seven million parameters.</>, i: false },
  { body: (<><Emph>Assembled by hand.</Emph> Zero pretrained weights.</>), i: true },
];

export function Hero() {
  const reduced = usePrefersReducedMotion();
  const sectionRef = useRef<HTMLElement>(null);

  const { scrollYProgress } = useScroll({ target: sectionRef, offset: ["start start", "end start"] });
  const slabTilt = useSpring(useTransform(scrollYProgress, [0, 0.35], [0, -7]), { stiffness: 60, damping: 20 });
  const slabY = useTransform(scrollYProgress, [0, 0.5], [0, 90]);
  const slabScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.94]);
  const textY = useTransform(scrollYProgress, [0, 0.5], [0, -30]);

  const stagger = 0.1;

  return (
    <section ref={sectionRef} id="top" className="relative overflow-hidden pt-28 pb-14 sm:pt-32">
      {/* corner registration marks */}
      <span aria-hidden className="absolute left-6 top-24 h-3 w-3 border-t border-l border-ink3/70" />
      <span aria-hidden className="absolute right-6 top-24 h-3 w-3 border-t border-r border-ink3/70" />

      <motion.div style={reduced ? undefined : { y: textY }} className="relative z-10 mx-auto max-w-[1200px] px-6">
        {/* micro meta line */}
        <motion.div
          initial={reduced ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="mb-8 flex items-center gap-3 font-mono text-[10px] sm:text-[11px] tracking-[0.3em] text-ink3 uppercase"
        >
          <span className="inline-block h-[6px] w-[6px] rounded-full bg-ember pulse-dot" aria-hidden />
          From-scratch language machine
          <span aria-hidden className="hidden h-px flex-1 bg-line sm:block" />
          <span className="hidden sm:inline">GIBC V2 · TRACK 01 · {PARAMS.toLocaleString("en-US")} PARAMETERS</span>
        </motion.div>

        {/* headline */}
        <h1 className="max-w-[1100px] text-balance">
          {LINES.map((l, i) => (
            <span key={i} className="block overflow-hidden">
              <motion.span
                className="block font-display text-[clamp(2.5rem,8.25vw,5.2rem)] leading-[1.02] font-semibold tracking-tight"
                initial={reduced ? false : { y: "112%", rotate: 3 }}
                animate={{ y: 0, rotate: 0 }}
                transition={{ duration: 1.05, delay: 0.08 + i * stagger, ease: [0.16, 1, 0.3, 1] }}
              >
                {l.body}
              </motion.span>
            </span>
          ))}
        </h1>

        <p className="mt-8 max-w-[52ch] font-display text-lg leading-relaxed text-ink2 sm:text-xl">
          A sparse twist on the small transformer — <Emph>GQA + RoPE + SwiGLU</Emph> and{" "}
          <Emph>Mamba</Emph> interleaved, crowned by a top-2 Mixture of Experts with a
          domain-seeded router. Trained under a hard 50M ceiling on a single GPU.
        </p>
      </motion.div>

      {/* the slab */}
      <div className="relative mx-auto mt-12 max-w-[1200px] px-6" style={{ perspective: 1400 }}>
        <motion.div
          style={reduced ? undefined : { rotateX: slabTilt, y: slabY, scale: slabScale }}
          className="relative overflow-hidden rounded-[20px] border border-slabline bg-slab shadow-[0_30px_80px_-30px_rgba(26,23,17,0.5)]"
        >
          <div className="absolute inset-0 slab-grid opacity-70" aria-hidden />
          <div className="absolute inset-0 bg-gradient-to-t from-slab2 to-transparent" aria-hidden />

          {/* top bar of the panel */}
          <div className="relative flex items-center justify-between border-b border-slabline px-5 py-3">
            <span className="font-mono text-[10px] tracking-[0.28em] text-creamdim uppercase">
              MoE · router activity monitor
            </span>
            <span className="flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] text-creamdim">
              <span className="pulse-dot h-[6px] w-[6px] rounded-full bg-trace" aria-hidden />
              sweep 1s/div
            </span>
          </div>

          {/* oscilloscope body */}
          <div className="relative h-[340px] sm:h-[400px]">
            <Oscilloscope />
          </div>

          {/* bottom readout row */}
          <div className="relative grid grid-cols-2 gap-px border-t border-slabline bg-slabline sm:grid-cols-4">
            {[
              ["ACTIVE EXPERTS", "2 / 8"],
              ["ROUTING MODE", "TOP-K"],
              ["GUIDE CURRICULUM", "2,000 STEP"],
              ["SIGNAL", "LIVE"],
            ].map(([k, v]) => (
              <div key={k} className="bg-slab px-5 py-3">
                <div className="font-mono text-[9px] tracking-[0.24em] text-creamdim uppercase">{k}</div>
                <div className="mt-1 font-mono text-sm text-trace">{v}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* scroll cue */}
      <div className="mx-auto mt-10 flex max-w-[1200px] items-center gap-4 px-6 font-mono text-[10px] tracking-[0.3em] text-ink3 uppercase">
        <span>AWAITING INPUT…</span>
        <span aria-hidden className="h-px w-24 bg-linestrong ruler-notches" />
        <motion.span
          animate={reduced ? undefined : { y: [0, 5, 0] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          aria-hidden
        >
          ↓
        </motion.span>
      </div>
    </section>
  );
}