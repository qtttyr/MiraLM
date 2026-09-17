"use client";

// Warm oscilloscope: 8 channel traces (one per expert) on the dark slab.
// The trace under your cursor swells; a blip patrols the active channel.
// Meters on the right react live via direct style writes (no re-render).

import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

const CH = 8;

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  t: number,
  focus: number, // -1..1 from pointer x
  active: number, // 0..7 winning channel
  reduced: boolean,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 1.2;

  const margin = 6;
  const step = (h - margin * 2) / CH;
  ctx.font = `${Math.max(9, step * 0.4)}px "IBM Plex Mono", monospace`;

  for (let i = 0; i < CH; i++) {
    const y0 = margin + i * step + step / 2;
    const dist = Math.abs(focus - (-1 + (2 * i) / (CH - 1)));
    const swell = Math.max(0.25, 1.15 - dist * 0.7);
    const isActive = i === active;

    // channel id label
    ctx.fillStyle = isActive ? "#e1491a" : "#8f8771";
    ctx.fillText(`E0${i}`, margin + 2, y0 + 3);

    // axis
    ctx.strokeStyle = isActive ? "rgba(225,73,26,0.5)" : "rgba(50,41,27,0.7)";
    ctx.beginPath();
    ctx.moveTo(34, y0);
    ctx.lineTo(w - 42, y0);
    ctx.stroke();

    // waveform
    const amp = (isActive ? 5 : 3) * swell * (step / 10);
    ctx.strokeStyle = isActive ? "#e6a13e" : "rgba(230,161,62,0.55)";
    ctx.beginPath();
    const phase = t * (1.4 + i * 0.13);
    for (let x = 36, k = 0; x <= w - 44; x++, k++) {
      const yy =
        y0 +
        Math.sin((x + phase) * 0.028) * amp * 0.7 +
        Math.sin((x * 0.7) + phase * 1.7) * amp * 0.3 +
        Math.sin(x * 0.05 + t * 3 + i) * amp * 0.5;
      if (k === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    }
    ctx.stroke();

    // blip on the active channel, riding right-to-left
    if (isActive && !reduced) {
      const bx = w - 44 - ((t * 2.2) % (w - 80));
      const by = y0 + Math.sin(bx * 0.05) * amp * 0.5;
      ctx.fillStyle = "#ff7a3d";
      ctx.beginPath();
      ctx.arc(bx, by, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // faint center line (screen marker)
  ctx.strokeStyle = "rgba(176,168,138,0.18)";
  ctx.setLineDash([3, 7]);
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);
}

export function Oscilloscope({ reduced }: { reduced?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const metersRef = useRef<HTMLDivElement>(null);
  const prefReduced = usePrefersReducedMotion();
  const reducedMotion = reduced ?? prefReduced;
  const focusRef = useRef(0);
  const activeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;

    let raf = 0;
    let w = 0;
    let h = 0;
    let t = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = wrap.clientWidth;
      h = wrap.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const onMove = (e: PointerEvent) => {
      const r = wrap.getBoundingClientRect();
      focusRef.current = ((e.clientX - r.left) / r.width) * 2 - 1;
      activeRef.current = Math.max(0, Math.min(CH - 1, Math.round(((focusRef.current + 1) / 2) * (CH - 1))));
    };
    const onLeave = () => (focusRef.current = 0);
    wrap.addEventListener("pointermove", onMove);
    wrap.addEventListener("pointerleave", onLeave);

    const tick = () => {
      t += 0.016;
      draw(ctx, w, h, t, focusRef.current, activeRef.current, reducedMotion);

      // drive side meters directly
      const meters = metersRef.current;
      if (meters) {
        const bars = meters.children;
        for (let i = 0; i < CH; i++) {
          const dist = Math.abs(focusRef.current - (-1 + (2 * i) / (CH - 1)));
          const active = i === activeRef.current ? 0.12 : 0;
          const val = Math.max(0.08, 1 - dist) - active;
          const el = bars[i] as HTMLElement;
          el.style.height = `${Math.round(val * 100)}%`;
          el.style.background = i === activeRef.current ? "var(--ember)" : "#3a3020";
        }
      }
      if (!reducedMotion) raf = requestAnimationFrame(tick);
    };
    tick();
    if (reducedMotion) raf = 0;

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      wrap.removeEventListener("pointermove", onMove);
      wrap.removeEventListener("pointerleave", onLeave);
    };
  }, [reducedMotion]);

  return (
    <div ref={wrapRef} className="relative h-full min-h-[320px] w-full overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0" aria-label="interactive expert activity" />
      {/* right channel meters */}
      <div
        ref={metersRef}
        className="absolute top-0 bottom-0 right-5 flex w-2 items-stretch justify-between gap-1"
        aria-hidden
      >
        {Array.from({ length: CH }).map((_, i) => (
          <div key={i} className="w-full self-end origin-bottom transition-colors duration-300 bg-[#3a3020]" style={{ height: "10%" }} />
        ))}
      </div>
    </div>
  );
}