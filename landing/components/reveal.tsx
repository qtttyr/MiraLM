"use client";

// Staggered reveal: rises + de-blurs content when it crosses into view,
// and can draw an enclosing hairline frame (border-draw top+left then fade).
import { cn } from "@/lib/cn";
import { motion } from "motion/react";
import type { ReactNode } from "react";

export function Reveal({
  children,
  className,
  delay = 0,
  frame = false,
  degree = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  frame?: boolean;
  degree?: number;
}) {
  return (
    <motion.div
      className={cn("relative", className)}
      initial={{ opacity: 0, y: degree === 0 ? 26 : 0, x: degree === 0 ? 0 : degree, filter: "blur(6px)" }}
      whileInView={{ opacity: 1, y: 0, x: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-10% 0px" }}
      transition={{ duration: 0.9, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {frame && (
        <>
          <motion.span
            aria-hidden
            className="pointer-events-none absolute inset-y-0 left-0 w-px bg-linestrong"
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: delay + 0.15, ease: [0.16, 1, 0.3, 1] }}
          />
          <motion.span
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-px bg-linestrong"
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: delay + 0.1, ease: [0.16, 1, 0.3, 1] }}
          />
        </>
      )}
      {children}
    </motion.div>
  );
}