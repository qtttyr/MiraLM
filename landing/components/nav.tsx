"use client";

// Fixed top bar: wordmark + section links + a "training status" LED.
import { useLenis } from "lenis/react";
import { NAV, GITHUB } from "@/lib/content";
import { cn } from "@/lib/cn";
import { useState } from "react";

export function Nav() {
  const lenis = useLenis();
  const [scrolled, setScrolled] = useState(false);

  // Subscribe to lenis scroll to style the bar once it leaves the top
  useLenis(({ scroll }) => setScrolled(scroll > 24));

  const go = (href: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const node = document.querySelector(href);
    if (node) lenis?.scrollTo(node as HTMLElement, { offset: -72, duration: 1.3 });
    else window.location.hash = href;
  };

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-40 transition-colors duration-500",
        scrolled ? "border-b border-line bg-paper/85 backdrop-blur-sm" : "border-b border-transparent",
      )}
    >
      <nav className="mx-auto flex max-w-[1200px] items-center justify-between px-6 py-4" aria-label="Sections">
        <a href="#top" onClick={go("#top")} className="flex items-center gap-3 font-mono text-sm tracking-tight text-ink">
          <span className="grid h-6 w-6 place-items-center rounded-[4px] bg-ink font-mono text-[11px] text-cream" aria-hidden>
            M
          </span>
          MIRALM<span className="text-ink3">·47M</span>
        </a>

        <ul className="hidden items-center gap-7 md:flex">
          {NAV.map((n) => (
            <li key={n.href}>
              <a
                href={n.href}
                onClick={go(n.href)}
                className="link-ink font-mono text-[11px] tracking-[0.14em] text-ink2 hover:text-ink uppercase"
              >
                {n.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-4">
          <span className="hidden items-center gap-2 font-mono text-[10px] tracking-[0.18em] text-ink2 sm:flex">
            <span className="pulse-dot h-[6px] w-[6px] rounded-full bg-ember" aria-hidden />
            STEP 12,842
          </span>
          <a
            href={GITHUB}
            target="_blank"
            rel="noreferrer"
            className="rounded-full border border-ink px-4 py-2 font-mono text-[11px] tracking-[0.14em] text-ink transition-colors hover:bg-ink hover:text-cream"
          >
            GITHUB ↗
          </a>
        </div>
      </nav>
    </header>
  );
}