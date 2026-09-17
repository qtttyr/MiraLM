import { PARAMS, ACTIVE, VOCAB, EXPERTS_N, TOP_K, TOKENS_TARGET, BUDGET, SEQ_LEN } from "@/lib/content";
import type { CSSProperties } from "react";

const ITEMS = [
  `PARAMS ${PARAMS.toLocaleString("en-US")}`,
  `ACTIVE ${ACTIVE.toLocaleString("en-US")}`,
  `BUDGET ${(BUDGET / 1e6)}M`,
  `EXPERTS ${EXPERTS_N}`,
  `TOP-${TOP_K} ROUTING`,
  `VOCAB ${VOCAB}`,
  `SEQ ${SEQ_LEN}`,
  `TOKENS ${TOKENS_TARGET}`,
  "ZERO PRETRAINED WEIGHTS",
  "ONE GPU · FP16",
  "STRUCTURED OUTPUT PROTOCOL",
  "GQA · RoPE · SwiGLU",
  "MAMBA SSM",
  "DOMAIN-SEEDED ROUTER",
];

export function Ticker() {
  const row = (key: string) => (
    <div key={key} className="flex shrink-0 items-center" aria-hidden>
      {ITEMS.map((t) => (
        <span key={t + key} className="flex items-center font-mono text-[12px] tracking-[0.22em] whitespace-nowrap">
          <span className="px-5 text-ink">{t}</span>
          <span className="text-cream/60">◆</span>
        </span>
      ))}
    </div>
  );

  return (
    <div className="relative z-10 overflow-hidden border-y border-ember-deep bg-ember py-2.5 text-cream">
      <div className="ticker-track" style={{ "--ticker-dur": "46s" } as CSSProperties}>
        {row("a")}
        {row("b")}
      </div>
    </div>
  );
}