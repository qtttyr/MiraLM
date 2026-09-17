"use client";

// The "live" console mock: a dark instrument that pretends to generate.
// JSON / SQL / CoT tabs — response reveals token by token while the
// two active experts flicker above the stream.

import { useEffect, useMemo, useRef, useState } from "react";
import { MOCK_LIVE } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { cn } from "@/lib/cn";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

const TAB_LABEL: Record<string, string> = { json: "JSON", sql: "SQL", cot: "REASON" };

export function TryConsole() {
  const reduced = usePrefersReducedMotion();
  const [active, setActive] = useState(0);
  const [shown, setShown] = useState(0);
  const [tick, setTick] = useState(0);
  const raf = useRef(0);

  const entry = MOCK_LIVE[active];
  const tokens = useMemo(() => entry.response.split(/(\s+)/g), [entry]);

  // fake async generation
  useEffect(() => {
    if (reduced) {
      setShown(tokens.length);
      return;
    }
    setShown(0);
    let acc = 0;
    let timer = 0;
    const step = () => {
      acc += 1;
      setShown(acc);
      setTick((t) => t + 1);
      if (acc < tokens.length) timer = window.setTimeout(step, 26 + Math.random() * 40);
    };
    timer = window.setTimeout(step, 320);
    const clear = () => {
      window.clearTimeout(timer);
      cancelAnimationFrame(raf.current);
    };
    return clear;
  }, [tokens, reduced]);

  const activeExperts = entry.experts;
  const hot = activeExperts[tick % activeExperts.length];

  return (
    <section id="live" className="relative bg-paper py-24 sm:py-32">
      <div className="mx-auto max-w-[1200px] px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <SectionTag>04 · LIVE</SectionTag>
            <h2 className="mt-4 max-w-[16ch] font-display text-5xl font-semibold leading-[0.92] tracking-tight sm:text-6xl">
              Ask it in the <Emph>shape you want back.</Emph>
            </h2>
          </div>
          <p className="max-w-[38ch] text-sm leading-6 text-ink2">
            Structured output is a <em className="font-display italic">protocol</em>: explicit special
            tokens for JSON, SQL and reasoning. Machine-parseable by construction — no format hacks.
          </p>
        </div>

        <div className="relative mt-12">
          {/* tabs */}
          <div className="flex gap-2">
            {MOCK_LIVE.map((m, i) => (
              <button
                key={m.domain}
                onClick={() => setActive(i)}
                className={cn(
                  "rounded-t-[10px] border border-b-0 px-5 py-3 font-mono text-[11px] tracking-[0.2em] uppercase transition-colors",
                  i === active
                    ? "border-slabline bg-slab text-trace"
                    : "border-line bg-paper2 text-ink3 hover:text-ink",
                )}
              >
                ▍{TAB_LABEL[m.domain]}
              </button>
            ))}
          </div>

          {/* slab console */}
          <div className="overflow-hidden rounded-b-[16px] rounded-tr-[16px] border border-slabline bg-slab text-cream">
            <div className="relative slab-grid">
              <div className="absolute inset-0 bg-gradient-to-t from-slab2 to-transparent" aria-hidden />

              <div className="relative grid gap-0 lg:grid-cols-[1.05fr_1fr]">
                {/* input panel */}
                <div className="border-b border-slabline p-6 lg:border-b-0 lg:border-r">
                  <div className="flex items-center justify-between font-mono text-[10px] tracking-[0.26em] text-creamdim uppercase">
                    <span>INPUT</span>
                    <span className="flex gap-1.5" aria-hidden>
                      <span className="h-2 w-2 rounded-full bg-emberdeep/80" />
                      <span className="h-2 w-2 rounded-full bg-creamdim/40" />
                      <span className="h-2 w-2 rounded-full bg-creamdim/40" />
                    </span>
                  </div>
                  <p className="mt-4 font-mono text-[13px] leading-relaxed text-cream">
                    <span className="text-trace">&gt;&nbsp;</span>
                    {entry.prompt}
                  </p>

                  {/* active experts */}
                  <div className="mt-6 flex items-center gap-2">
                    <span className="font-mono text-[9px] tracking-[0.26em] text-creamdim uppercase">active</span>
                    {activeExperts.map((e) => (
                      <span
                        key={e}
                        className={cn(
                          "rounded-[6px] border px-2 py-1 font-mono text-[11px] transition-all duration-150",
                          e === hot && !reduced
                            ? "border-ember bg-ember/15 text-emberglow"
                            : "border-slabline text-creamdim",
                        )}
                      >
                        E0{e}
                      </span>
                    ))}
                  </div>
                </div>

                {/* output panel */}
                <div className="relative p-6">
                  <div className="font-mono text-[10px] tracking-[0.26em] text-creamdim uppercase">
                    OUTPUT · <span className="text-trace">tok {shown}/{tokens.length}</span>
                  </div>
                  <p className="mt-4 font-mono text-[13px] leading-relaxed text-trace">
                    {tokens.slice(0, shown).map((tk, i) => (
                      <span key={i} className={cn(tk.trim() === "" ? "inline-block" : "", "px-px")}>{tk}</span>
                    ))}
                    <span className="caret ml-0.5 inline-block h-[14px] w-[7px] translate-y-[2px] bg-emberglow" aria-hidden />
                  </p>
                  <span aria-hidden className="absolute right-6 bottom-6 font-mono text-[9px] tracking-[0.26em] text-creamdim uppercase">
                    {MOCK_LIVE[active].domain} · demo
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3 font-mono text-[10px] tracking-[0.24em] text-ink3 uppercase">
            <span className="pulse-dot h-[6px] w-[6px] rounded-full bg-ember" aria-hidden />
            Concept render — the real model answers in the web demo.
          </div>
        </div>
      </div>
    </section>
  );
}