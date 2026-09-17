"use client";

// Smooth-scroll provider (Lenis) — the base of the "analogue instrument" feel.
// Lenis lerps the page, so the oscilloscope tilt and horizontal
// conveyor read as physical motion, not CSS.

import { ReactLenis } from "lenis/react";
import type { ReactNode } from "react";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

export function SmoothScrollProvider({ children }: { children: ReactNode }) {
  const reduced = usePrefersReducedMotion();
  return (
    <ReactLenis
      root
      options={{
        duration: 1.15,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: !reduced,
        touchMultiplier: 1.6,
      }}
    >
      {children}
    </ReactLenis>
  );
}