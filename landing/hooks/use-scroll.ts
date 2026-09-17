"use client";

// Document scroll progress as a smooth motion value (drives the
// sprocket ruler + section parallax). Safe to call in any component.

import { useScroll, useSpring } from "motion/react";

export function useScrollProgress() {
  const { scrollYProgress } = useScroll();
  const smooth = useSpring(scrollYProgress, { stiffness: 90, damping: 26, mass: 0.4 });
  return { scrollYProgress, smooth };
}