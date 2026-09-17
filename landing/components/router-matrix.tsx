"use client";

// Router matrix: the 8-expert × 8-domain activity heatmap.
// Hover a domain → its column swells; hover an expert → its row glows.
// A fully static, CSS-driven interaction (no re-render).

import { useState } from "react";
import { DOMAINS } from "@/lib/content";
import { SectionTag, Emph } from "@/components/typography";
import { cn } from "@/lib/cn";

// Synthetic routing preference: base matrix + domain bias.
function cell(exp: number, dom: number) {
  const bias = [0.9, 0.7, 0.6, 0.8, 0.75, 0.85, 0.3, 0.25]; // expert specialty
  const noise = ((exp * 7 + dom * 13) % 10) / 22 - 0.2;
  let v = bias[exp] * (1 - Math.abs(exp - dom) * 0.14) + noise;
  if (exp === dom) v = Math.min(1, v + 0.18);
  return Math.max(0.06, Math.min(1, v));
}

export function RouterMatrix() {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);

  return (
    <section id="router" className="relative bg-paper2 py-24 sm:py-32">
      <div className="mx-auto max-w-[1200px] px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <SectionTag>02 · ROUTER</SectionTag>
            <h2 className="mt-4 max-w-[15ch] font-display text-5xl font-semibold leading-[0.92] tracking-tight sm:text-6xl">
              Every token is routed to <Emph>two</Emph> of eight experts.
            </h2>
          </div>
          <p className="max-w-[38ch] text-sm leading-6 text-ink2">
            The router never sees the whole net — just a cheap linear projection. A 2,000-step
            guide curriculum seeds each expert with a domain, then anneals away.
          </p>
        </div>

        <div className="mt-12 grid gap-10 lg:grid-cols-[1fr_320px]">
          {/* heatmap */}
          <div className="relative overflow-hidden rounded-[16px] border border-line bg-paper p-5">
            <div className="flex items-center justify-between border-b border-line pb-3 font-mono text-[10px] tracking-[0.24em] text-ink3 uppercase">
              <span>ROUTING PREFERENCE · GATE TIME-SERIES (SIM)</span>
              <span className="flex items-center gap-2">
                <span className="h-[6px] w-[6px] rounded-full bg-ember pulse-dot" aria-hidden /> LIVE
              </span>
            </div>

            <div className="mt-4 grid grid-cols-[86px_repeat(8,1fr)] gap-1">
              {/* header */}
              <div />
              {DOMAINS.map((d) => (
                <button
                  key={d.id}
                  onPointerEnter={() => setHover((p) => ({ ...p!, c: DOMAINS.indexOf(d) }))}
                  onPointerLeave={() => setHover((p) => (p?.c === DOMAINS.indexOf(d) ? null : p))}
                  className={cn(
                    "rounded-[6px] py-1 font-mono text-[10px] tracking-[0.14em] uppercase transition-colors",
                    hover?.c === DOMAINS.indexOf(d) ? "bg-slab text-cream" : "bg-line/60 text-ink2",
                  )}
                >
                  {d.short}
                </button>
              ))}

              {DOMAINS.map((dRow, r) => (
                <div key={dRow.id} className="contents">
                  <button
                    onPointerEnter={() => setHover((p) => ({ ...p!, r }))}
                    onPointerLeave={() => setHover((p) => (p?.r === r ? null : p))}
                    className={cn(
                      "justify-self-start self-center rounded-[6px] px-2 py-1 font-mono text-[10px] tracking-[0.14em] uppercase transition-colors",
                      hover?.r === r ? "bg-slab text-cream" : "text-ink2",
                    )}
                  >
                    {dRow.expert}
                  </button>
                  {DOMAINS.map((dCol, c) => {
                    const v = cell(r, c);
                    const dimV = Math.round(v * 100) / 100;
                    const activeCell = hover && (hover.r === r || hover.c === c);
                    return (
                      <div
                        key={dCol.id}
                        className={cn(
                          "group relative aspect-[3/2] rounded-[6px] transition-transform duration-300",
                          hover?.c === c && "scale-y-110",
                          hover?.r === r && "scale-x-110",
                        )}
                        style={{
                          background: `color-mix(in srgb, var(--ember) ${(v * 100).toFixed(0)}%, var(--line))`,
                          opacity: activeCell ? 1 : hover ? 0.45 : 1,
                        }}
                      >
                        <span className="pointer-events-none absolute inset-0 grid place-items-center font-mono text-[9px] text-paper opacity-0 transition-opacity group-hover:opacity-80">
                          .{dimV.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            <div className="mt-4 flex items-center gap-3 font-mono text-[9px] tracking-[0.2em] text-ink3 uppercase">
              <span>0.00</span>
              <span aria-hidden className="h-1.5 flex-1 rounded-full" style={{ background: "linear-gradient(90deg, var(--line), var(--ember))" }} />
              <span>1.00</span>
            </div>
          </div>

          {/* side readout */}
          <aside className="flex flex-col gap-4">
            <div className="rounded-[16px] border border-line bg-paper p-5">
              <h3 className="font-mono text-[10px] tracking-[0.28em] text-ink3 uppercase">SPECIALTY MAP</h3>
              <ul className="mt-4 grid gap-2.5">
                {DOMAINS.map((d) => (
                  <li key={d.id} className="flex items-baseline justify-between gap-3">
                    <span className={cn("font-mono text-xs font-medium uppercase", d.css)}>
                      E0{DOMAINS.indexOf(d)} · {d.short}
                    </span>
                    <span className="text-[11px] text-ink3">{d.desc}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-[16px] bg-slab p-5 text-cream">
              <p className="font-mono text-[10px] tracking-[0.28em] text-creamdim uppercase">WHY TWO?</p>
              <p className="mt-3 font-display text-lg leading-snug">
                Top-2 keeps capacity bloated without the compute bill — <Emph>dense at heart, sparse in transit.</Emph>
              </p>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}