#!/usr/bin/env python3
"""Generate the animated SVG figures for the README.

Design language is taken verbatim from the project page (landing/app/globals.css),
palette "Paper & Ember": ivory paper, espresso ink, ember orange, petrol teal,
plus dark "slab" instrument panels. Deliberately NO purple, no blue, no neon.
Display serif for headlines, mono micro-labels with wide tracking, hairline
rules, corner registration marks and paper grain - the print-lab look.

Technical note: GitHub sanitises INLINE svg in markdown (it strips <style> and
SMIL), but a standalone .svg referenced with ![](assets/x.svg) is served as an
image document where both internal <style> and SMIL <animate> run. Everything
here is built for that path, and the XML is validated on write so a malformed
figure fails the build instead of rendering as an empty box on the page.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets"

# ---- "Paper & Ember" tokens, mirrored from the landing ----------------------
PAPER = "#f0e9db"
INK = "#1a1711"
INK2 = "#55503f"
INK3 = "#8f8771"
LINE = "#d4c9b3"
LINE_STRONG = "#bcae90"
SLAB = "#14110c"
SLAB_LINE = "#32291b"
CREAM_DIM = "#b6a889"
EMBER = "#e1491a"
EMBER_DEEP = "#b93a12"
EMBER_GLOW = "#ff7a3d"
PETROL = "#175e58"
TRACE = "#e6a13e"

SERIF = "Georgia, 'Iowan Old Style', 'Times New Roman', serif"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=11, fill=INK, anchor="start", weight="400",
      family=MONO, spacing=None, style=None, opacity=None) -> str:
    a = f' letter-spacing="{spacing}"' if spacing else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    st = f' font-style="{style}"' if style else ""
    return (f'  <text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
            f'{a}{st}{o}>{esc(s)}</text>')


def reg_marks(w: int, h: int, pad=14, color=INK3, size=12) -> str:
    """Corner registration marks - the print-lab signature."""
    o = []
    for (x, y, dx, dy) in ((pad, pad, 1, 1), (w - pad, pad, -1, 1),
                           (pad, h - pad, 1, -1), (w - pad, h - pad, -1, -1)):
        o.append(f'  <path d="M {x} {y + dy*size} L {x} {y} L {x + dx*size} {y}" '
                 f'fill="none" stroke="{color}" stroke-width="1" '
                 f'stroke-opacity="0.6"/>')
    return "\n".join(o)


def svg(w: int, h: int, body: str, title: str, desc: str, bg: str = PAPER) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"\n'
        f'     width="{w}" height="{h}" role="img" aria-labelledby="t d">\n'
        f'  <title id="t">{title}</title>\n'
        f'  <desc id="d">{desc}</desc>\n'
        f'  <defs>\n'
        f'    <filter id="grain" x="0" y="0" width="100%" height="100%">\n'
        f'      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3"'
        f' stitchTiles="stitch"/>\n'
        f'      <feColorMatrix type="saturate" values="0"/>\n'
        f'    </filter>\n'
        f'  </defs>\n'
        f'  <rect width="{w}" height="{h}" fill="{bg}"/>\n'
        f'  <rect width="{w}" height="{h}" filter="url(#grain)" opacity="0.045"/>\n'
        f'{body}\n'
        f'</svg>\n'
    )


def pulse_dot(x, y, color, dur="1.8s") -> str:
    return (f'  <circle cx="{x}" cy="{y}" r="3.5" fill="{color}">'
            f'<animate attributeName="opacity" values="1;0.25;1" dur="{dur}" '
            f'repeatCount="indefinite"/></circle>')


def hairline(x0, y, x1, color=LINE, wdt=1, opacity=1.0, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{color}" '
            f'stroke-width="{wdt}" stroke-opacity="{opacity}"{d}/>')


def ruler(x0, y, width, segs=50, color=LINE_STRONG, active=0) -> str:
    """Segmented tick rule that fills from the left - the budget meter."""
    o = []
    step = width / segs
    for i in range(segs):
        filled = i < active
        col = EMBER if filled else color
        op = 1.0 if filled else 0.45
        h = 11 if filled else 7
        o.append(f'  <line x1="{x0 + i*step:.2f}" y1="{y}" '
                 f'x2="{x0 + i*step:.2f}" y2="{y - h}" stroke="{col}" '
                 f'stroke-width="1.6" stroke-opacity="{op}"/>')
    return "\n".join(o)


# ---------------------------------------------------------------- hero ------
def hero() -> str:
    w, h = 900, 232
    fill = int(50 * 47_640_968 / 50_000_000)
    cx = 24 + (w - 48) * 47_640_968 / 50_000_000
    b = [reg_marks(w, h),
         pulse_dot(24, 28, EMBER),
         t(36, 32, "FROM-SCRATCH LANGUAGE MACHINE", 10, INK3, spacing="0.30em"),
         t(w - 24, 32, "GIBC V2 · TRACK 01", 10, INK3, "end", spacing="0.30em"),
         t(24, 88, "MiraLM-47M", 52, INK, weight="700", family=SERIF),
         t(28, 112, "assembled by hand, learned from nothing", 15, EMBER,
           style="italic", family=SERIF),
         hairline(24, 132, w - 24, LINE_STRONG, 1, 0.8),
         t(24, 156, "PARAMETER BUDGET", 9, INK3, spacing="0.26em"),
         t(w - 24, 156, "HARD CEILING 50,000,000", 9, INK3, "end", spacing="0.26em"),
         ruler(24, 176, w - 48, 50, LINE_STRONG, fill),
         f'  <line x1="{cx:.1f}" y1="160" x2="{cx:.1f}" y2="180" '
         f'stroke="{EMBER}" stroke-width="1.5">'
         f'<animate attributeName="stroke-opacity" values="1;0.35;1" dur="2s" '
         f'repeatCount="indefinite"/></line>',
         t(24, 204, "47,640,968 trainable", 13, INK, weight="600"),
         t(232, 204, "38,793,608 active / token", 13, PETROL, weight="600"),
         t(w - 24, 204, "95 MB fp16 · 1× T4 · 4,096 tok/step", 11, INK2, "end")]
    return svg(w, h, "\n".join(b), "MiraLM-47M",
               "Spec-sheet banner: the 47,640,968-parameter budget fills a "
               "segmented rule toward the 50,000,000 ceiling.")


# -------------------------------------------------------------- budget -----
def budget() -> str:
    rows = [("Embedding — tied, shared with lm_head", 9_216_000, PETROL),
            ("Mamba blocks ×7", 7_007_616, EMBER_DEEP),
            ("Attention + SwiGLU ×7", 19_617_024, EMBER),
            ("MoE experts ×8 — top-2 active", 11_796_480, TRACE),
            ("Router + norms", 3_848, INK3)]
    w, h = 900, 384
    total = sum(r[1] for r in rows)
    x0, scale = 250, 600 / 50_000_000
    b = [reg_marks(w, h),
         t(24, 32, "FIG. 01", 10, EMBER, weight="700", spacing="0.26em"),
         t(24, 52, "Parameter budget", 24, INK, family=SERIF),
         t(w - 24, 32, "STATIC PROJECTION  ≡  TORCH NUMEL", 9, INK3, "end",
           spacing="0.22em"),
         hairline(24, 66, w - 24, LINE, 1, 0.9)]
    y = 92
    for i, (name, val, col) in enumerate(rows):
        bw = max(1.5, val * scale)
        b.append(t(24, y + 12, name, 11, INK2))
        b.append(f'  <rect x="{x0}" y="{y}" width="{bw:.1f}" height="14" '
                 f'fill="{col}" fill-opacity="0.14"/>')
        b.append(f'  <rect x="{x0}" y="{y}" width="0" height="14" fill="{col}">'
                 f'<animate attributeName="width" from="0" to="{bw:.1f}" '
                 f'dur="0.85s" begin="{i*0.12}s" fill="freeze"/></rect>')
        b.append(t(x0 + bw + 10, y + 12, f"{val:,}", 11, INK, weight="600"))
        b.append(hairline(24, y + 22, w - 24, LINE, 0.5, 0.7))
        y += 32
    b.append(hairline(24, y, w - 24, LINE_STRONG, 1.2, 1))
    b.append(t(24, y + 32, "TOTAL TRAINABLE", 10, INK3, spacing="0.24em"))
    b.append(t(24, y + 70, f"{total:,}", 34, INK, weight="700", family=SERIF))
    b.append(t(300, y + 70, "≤ 50,000,000", 13, INK3))
    b.append(f'  <rect x="470" y="{y+44}" width="92" height="30" rx="15" '
             f'fill="{EMBER}" fill-opacity="0.12" stroke="{EMBER}"/>')
    b.append(t(516, y + 64, "PASS", 13, EMBER, "middle", "700", spacing="0.18em"))
    b.append(t(w - 24, y + 64, f"headroom {50_000_000-total:,}", 12, PETROL,
               "end", weight="600"))
    b.append(ruler(24, y + 100, w - 48, 50, LINE_STRONG,
                  int(50 * total / 50_000_000)))
    return svg(w, h, "\n".join(b), "Parameter budget",
               "Fig. 01: a print-lab bar chart of the 47,640,968 parameters by "
               "component against the 50,000,000 ceiling.")


# --------------------------------------------------------- architecture -----
def arch() -> str:
    w, h = 900, 330
    b = ['  <defs>',
         '    <pattern id="grid" width="34" height="34" patternUnits="userSpaceOnUse">',
         f'      <path d="M 34 0 L 0 0 0 34" fill="none" stroke="{SLAB_LINE}" '
         f'stroke-width="1" stroke-opacity="0.55"/>',
         '    </pattern>',
         '  </defs>',
         f'  <rect width="{w}" height="{h}" fill="{SLAB}"/>',
         f'  <rect x="1" y="1" width="{w-2}" height="{h-2}" rx="18" fill="none" '
         f'stroke="{SLAB_LINE}" stroke-width="1.2"/>',
         '  <rect x="20" y="52" width="860" height="196" fill="url(#grid)" '
         'fill-opacity="0.5"/>',
         t(26, 34, "MoE · ROUTER ACTIVITY MONITOR", 10, CREAM_DIM,
           spacing="0.26em"),
         pulse_dot(w - 152, 30, TRACE, "1s"),
         t(w - 26, 34, "SWEEP 1s/DIV", 10, CREAM_DIM, "end", spacing="0.2em"),
         hairline(20, 52, w - 20, SLAB_LINE, 1, 1)]
    kinds = [("EMB", PETROL), ("MAMBA", EMBER_DEEP), ("ATTN", TRACE),
             ("MAMBA", EMBER_DEEP), ("ATTN", TRACE), ("MAMBA", EMBER_DEEP),
             ("ATTN", TRACE), ("GATE", PETROL), ("MoE", EMBER_GLOW),
             ("HEAD", PETROL)]
    bw, gap, x0, y0 = 74, 8, 26, 76
    for i, (name, col) in enumerate(kinds):
        x = x0 + i * (bw + gap)
        b.append(f'  <rect x="{x}" y="{y0}" width="{bw}" height="46" rx="5" '
                 f'fill="{col}" fill-opacity="0.10" stroke="{col}" '
                 f'stroke-opacity="0.55" stroke-width="1">'
                 f'<animate attributeName="fill-opacity" values="0.08;0.34;0.08" '
                 f'dur="3.4s" begin="{i*0.17}s" repeatCount="indefinite"/></rect>')
        b.append(t(x + bw / 2, y0 + 28, name, 10, col, "middle", "700",
                   spacing="0.1em"))
    b.append(t(26, 150, "ROUTING", 10, CREAM_DIM, spacing="0.26em"))
    b.append(t(26, 168, "8 experts · 2 active per token", 10, CREAM_DIM,
               opacity="0.7"))
    ex0, ey = 210, 198
    active = (4, 6)
    for e in range(8):
        ex = ex0 + e * 58
        on = e in active
        col = EMBER_GLOW if on else SLAB_LINE
        anim = (f'<animate attributeName="fill-opacity" values="0.3;1;0.3" '
                f'dur="1.5s" begin="{e*0.11}s" repeatCount="indefinite"/>'
                if on else '')
        b.append(f'  <circle cx="{ex}" cy="{ey}" r="15" fill="{col}" '
                 f'fill-opacity="{0.9 if on else 0.55}">{anim}</circle>')
        b.append(t(ex, ey + 4, f"E{e}", 9, SLAB if on else CREAM_DIM, "middle",
                   "700"))
    for tgt in active:
        x1 = ex0 + tgt * 58
        b.append(f'  <path d="M {x1} {ey-16} C {x1+60} {ey-46} 700 {ey-46} '
                 f'766 {ey-16}" fill="none" stroke="{EMBER_GLOW}" '
                 f'stroke-width="1.3" stroke-opacity="0.65" '
                 f'stroke-dasharray="5 5">'
                 f'<animate attributeName="stroke-dashoffset" from="20" to="0" '
                 f'dur="1s" repeatCount="indefinite"/></path>')
    b.append(f'  <rect x="766" y="{ey-34}" width="102" height="46" rx="6" '
             f'fill="{EMBER_GLOW}" fill-opacity="0.14" stroke="{EMBER_GLOW}" '
             f'stroke-opacity="0.7"/>')
    b.append(t(817, ey - 13, "TOP-2", 12, EMBER_GLOW, "middle", "700",
               spacing="0.12em"))
    b.append(t(817, ey + 3, "SwiGLU", 9, CREAM_DIM, "middle"))
    b.append(hairline(20, 256, w - 20, SLAB_LINE, 1, 1))
    cells = [("ACTIVE EXPERTS", "2 / 8"), ("ROUTING MODE", "TOP-K"),
             ("GUIDE CURRICULUM", "2,000 STEP"), ("EXPERT DIM", "1280")]
    cw = (w - 40) / 4
    for i, (k, v) in enumerate(cells):
        cx = 26 + i * cw
        if i:
            b.append(f'  <line x1="{cx-16:.0f}" y1="270" x2="{cx-16:.0f}" '
                     f'y2="306" stroke="{SLAB_LINE}"/>')
        b.append(t(cx, 280, k, 9, CREAM_DIM, spacing="0.2em"))
        b.append(t(cx, 300, v, 13, TRACE, weight="600"))
    return svg(w, h, "\n".join(b), "Architecture",
               "Dark instrument panel: alternating Mamba and attention blocks "
               "feeding a top-2-of-8 MoE router, with a live readout row.",
               bg=SLAB)


# --------------------------------------------------------------- curve -----
def curve() -> str:
    w, h = 900, 300
    x0, y0, pw, ph = 62, 64, 540, 186
    pts = [2.90, 2.86, 2.81, 2.78, 2.73, 2.70, 2.66, 2.63, 2.61, 2.58, 2.56,
           2.53, 2.51, 2.49, 2.47, 2.45, 2.43, 2.41, 2.39, 2.37, 2.35, 2.33]
    lo, hi = 2.2, 3.0

    def xy(i, v):
        return (x0 + i * pw / (len(pts) - 1),
                y0 + ph - (v - lo) / (hi - lo) * ph)

    d = " ".join(f"{'M' if i == 0 else 'L'} {xy(i, v)[0]:.1f} {xy(i, v)[1]:.1f}"
                 for i, v in enumerate(pts))
    b = [reg_marks(w, h),
         t(24, 32, "FIG. 02", 10, EMBER, weight="700", spacing="0.26em"),
         t(24, 52, "Training loss", 24, INK, family=SERIF),
         t(w - 24, 32, "11,000 STEPS · 1× T4", 9, INK3, "end", spacing="0.22em"),
         hairline(24, 66, w - 24, LINE, 1, 0.9)]
    for gv in (2.2, 2.4, 2.6, 2.8, 3.0):
        _, gy = xy(0, gv)
        b.append(hairline(x0, gy, x0 + pw, LINE, 0.6, 0.85, dash="2 4"))
        b.append(t(x0 - 10, gy + 4, f"{gv:.1f}", 10, INK3, "end"))
    b.append(f'  <path d="{d}" fill="none" stroke="{LINE_STRONG}" stroke-width="5" '
             f'stroke-linecap="round" stroke-linejoin="round" opacity="0.26"/>')
    b.append(f'  <path d="{d}" fill="none" stroke="{EMBER}" stroke-width="2" '
             f'stroke-linecap="round" stroke-linejoin="round" '
             f'stroke-dasharray="700" stroke-dashoffset="700">'
             f'<animate attributeName="stroke-dashoffset" from="700" to="0" '
             f'dur="2.2s" fill="freeze"/></path>')
    lx, ly = xy(len(pts) - 1, pts[-1])
    b.append(f'  <circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" fill="{EMBER}">'
             f'<animate attributeName="r" values="4;7;4" dur="2s" '
             f'repeatCount="indefinite"/></circle>')
    b.append(t(lx + 14, ly + 5, "2.33", 14, EMBER, weight="700", family=SERIF))
    b.append(f'  <line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" '
             f'stroke="{LINE_STRONG}"/>')
    b.append(t(x0, y0 + ph + 22, "step 0", 10, INK3))
    b.append(t(x0 + pw, y0 + ph + 22, "11,000", 10, INK3, "end"))
    bx = x0 + pw + 62
    b.append(t(bx, y0 + 6, "READOUT", 10, INK3, spacing="0.24em"))
    for i, (k, v, c) in enumerate([("final loss", "2.33", EMBER),
                                   ("tokens seen", "45M", INK),
                                   ("throughput", "803 tok/s", INK),
                                   ("hardware", "1× T4", PETROL),
                                   ("batch", "4 × 1024", INK),
                                   ("precision", "fp16 AMP", INK)]):
        yy = y0 + 32 + i * 28
        b.append(t(bx, yy, k, 10, INK3))
        b.append(t(w - 24, yy, v, 12, c, "end", "600"))
        b.append(hairline(bx, yy + 8, w - 24, LINE, 0.5, 0.7))
    return svg(w, h, "\n".join(b), "Training loss",
               "Fig. 02: training loss falling from 2.90 to 2.33 across 11,000 "
               "steps on a single T4, drawn on load.")


FIGURES = {"hero": hero, "budget": budget, "arch": arch, "curve": curve}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, fn in FIGURES.items():
        content = fn()
        ET.fromstring(content)          # fail loudly on malformed XML
        (OUT / f"{name}.svg").write_text(content, encoding="utf-8")
        print(f"  assets/{name}.svg  {len(content):>6} bytes  valid XML")
    print(f"\n{len(FIGURES)} figures written and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
