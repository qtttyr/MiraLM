"use client";

// Fixed left-side "film sprocket" ruler driven by scroll progress.
import { useScrollProgress } from "@/hooks/use-scroll";
import { motion, useTransform } from "motion/react";

export function ProgressRuler() {
  const { smooth } = useScrollProgress();
  const y = useTransform(smooth, (v) => `${v * 100}%`);

  return (
    <div className="hidden lg:block fixed left-6 top-0 bottom-0 z-20 w-px bg-line" aria-hidden>
      <motion.div
        className="absolute -left-[4px] -top-[5px]"
        style={{ top: y }}
      >
        <div className="flex items-center gap-2">
          <div className="h-[11px] w-[9px] rounded-[2px] border border-linestrong bg-paper" />
          <span className="font-mono text-[10px] tracking-widest text-ink3">TAPE</span>
        </div>
      </motion.div>
      <div className="absolute inset-y-1 -left-[2px] w-px ruler-notches opacity-60" />
    </div>
  );
}