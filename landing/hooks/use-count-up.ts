"use client";

// Animate a number from 0 → target when it enters the viewport.
import { useEffect, useRef } from "react";
import { animate, useInView, useMotionValue, useTransform } from "motion/react";

export function useCountUp<T extends HTMLElement = HTMLDivElement>(
  target: number,
  duration = 1.4,
  format?: (n: number) => string,
) {
  const ref = useRef<T | null>(null);
  const inView = useInView(ref, { once: true, margin: "-12% 0px" });
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => (format ? format(v) : v.toLocaleString("en-US")));

  useEffect(() => {
    if (!inView) return;
    const controls = animate(mv, target, { duration, ease: [0.16, 1, 0.3, 1] });
    return () => controls.stop();
  }, [inView, mv, target, duration]);

  return { ref, text };
}