#!/usr/bin/env python3
"""Generate the animated SVG figures for the README.

GitHub sanitises INLINE svg in markdown (so <style> and SMIL get stripped),
but a standalone .svg referenced with ![](assets/x.svg) is served as an image
document where both internal <style> and SMIL <animate> run. Everything here
is built to animate that way, and the output is XML-validated on write.

Palette: GitHub-dark friendly, one accent per concept, no decoration that does
not carry information.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets"

INK = "#0d1117"
PANEL = "#161b22"
LINE = "#30363d"
MUTED = "#8b949e"
TEXT = "#e6edf3"
AMBER = "#f0b429"
CYAN = "#39c5cf"
VIOLET = "#a371f7"
GREEN = "#3fb950"
RED = "#f85149"

FONT = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")
CYCLE = [CYAN, VIOLET, AMBER, CYAN]


def svg(width: int, height: int, body: str, title: str, desc: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
     width="{width}" height="{height}" role="img" aria-labelledby="t d">
  <title id="t">{title}</title>
  <desc id="d">{desc}</desc>
  <rect width="{width}" height="{height}" fill="{INK}"/>
{body}
</svg>
"""


def text(x, y, s, size=12, fill=TEXT, anchor="start", weight="400", opacity=None):
    op = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return (f'  <text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{op} text-anchor="{anchor}">'
            f'{s}</text>')


# ---------------------------------------------------------------- hero ------
def hero() -> str:
    w, h = 880, 190
    b = ['  <defs>',
         '    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">']
    for i, c in enumerate(CYCLE):
        nxt = CYCLE[(i + 1) % 4]
        b.append(f'      <stop offset="{i/3:.2f}" stop-color="{c}">'
                 f'<animate attributeName="stop-color" values="{c};{nxt};{c}" '
                 f'dur="9s" repeatCount="indefinite"/></stop>')
    b += ['    </linearGradient>', '  </defs>']
    b.append(f'  <rect x="0" y="52" width="{w}" height="3" fill="url(#g)">'
             f'<animate attributeName="x" values="0;{w}" dur="7s" '
             f'repeatCount="indefinite"/></rect>')
    b.append(f'  <rect x="0" y="52" width="120" height="3" fill="{CYAN}" '
             f'opacity="0.9"><animate attributeName="x" values="{w};-120" '
             f'dur="3.5s" repeatCount="indefinite"/></rect>')
    b.append(text(440, 40, "MiraLM-47M", 46, TEXT, "middle", "700"))
    b.append(text(440, 86, "Sparse MoE  x  Mamba  x  GQA  -  47,640,968 params",
                  15, CYAN, "middle", "500"))
    bx, by, bw = 300, 116, 280
    b.append(f'  <rect x="{bx}" y="{by}" width="{bw}" height="10" rx="5" '
             f'fill="{PANEL}" stroke="{LINE}"/>')
    fillw = int(bw * 47_640_968 / 50_000_000)
    b.append(f'  <rect x="{bx}" y="{by}" width="0" height="10" rx="5" '
             f'fill="{AMBER}"><animate attributeName="width" '
             f'values="0;{fillw}" dur="1.6s" fill="freeze"/></rect>')
    b.append(text(bx - 12, by + 10, "0", 12, MUTED, "end"))
    b.append(text(bx + bw + 12, by + 10, "50M", 12, MUTED))
    b.append(text(440, by + 34, "47,640,968 trainable  ·  2,359,032 headroom"
                                "  ·  38,793,608 active", 12, MUTED, "middle"))
    b.append(text(440, 172, "trained from scratch  ·  no pretrained weights"
                            "  ·  1x T4  ·  GIBC V2 Track 01",
                  12, GREEN, "middle"))
    return svg(w, h, "\n".join(b), "MiraLM-47M",
               "Animated banner with the parameter budget filling to 47,640,968 "
               "of the 50,000,000 cap.")


# -------------------------------------------------------------- budget -----
def budget() -> str:
    rows = [("Embedding (tied, shared with lm_head)", 9_216_000, CYAN),
            ("Mamba blocks  x7", 7_007_616, VIOLET),
            ("Attention + SwiGLU  x7", 19_617_024, AMBER),
            ("MoE experts  x8 (top-2 active)", 11_796_480, GREEN),
            ("Router + norms", 3_848, MUTED)]
    w, h = 880, 250
    total = sum(r[1] for r in rows)
    scale = 480 / 50_000_000
    b = [text(28, 30, "PARAMETER BUDGET", 13, TEXT, weight="700"),
         text(852, 30, "50,000,000 hard cap", 12, MUTED, "end")]
    y = 58
    for i, (name, val, col) in enumerate(rows):
        bw = max(2, int(val * scale))
        b.append(f'  <rect x="28" y="{y}" width="0" height="22" rx="3" '
                 f'fill="{col}" opacity="0.85">'
                 f'<animate attributeName="width" from="0" to="{bw}" '
                 f'dur="0.9s" begin="{i*0.13}s" fill="freeze"/></rect>')
        b.append(text(28, y - 5, name, 11, MUTED))
        b.append(text(28 + bw + 10, y + 16, f"{val:,}", 12, TEXT, weight="600"))
        y += 36
    b.append(f'  <line x1="28" y1="{y-6}" x2="508" y2="{y-6}" stroke="{LINE}"/>')
    b.append(f'  <rect x="28" y="{y}" width="480" height="24" rx="4" fill="none" '
             f'stroke="{AMBER}" stroke-width="1.5"/>')
    b.append(text(28, y + 40, f"total {total:,}   ·   PASS   ·   "
                              f"projection == torch numel()",
                  12, AMBER, weight="600"))
    b.append(text(852, y + 40, "enforced twice: static + real model", 11,
                  MUTED, "end"))
    return svg(w, h, "\n".join(b), "Parameter budget",
               "Animated breakdown of the 47,640,968 parameters by component "
               "against the 50,000,000 cap.")


# --------------------------------------------------------- architecture -----
def arch() -> str:
    w, h = 880, 300
    kinds = [("EMBED", CYAN), ("Mamba", VIOLET), ("Attention", AMBER),
             ("Mamba", VIOLET), ("Attention", AMBER), ("Mamba", VIOLET),
             ("Attention", AMBER), ("GATE", CYAN), ("MOE x8", GREEN),
             ("HEAD", CYAN)]
    bw, gap, x0, y0 = 74, 12, 28, 74
    b = [text(28, 32, "HYBRID STACK", 13, TEXT, weight="700"),
         text(28, 52, "Jamba-style alternation, 7 Mamba + 7 attention, then a "
                      "sparse top-2 MoE capstone", 11, MUTED),
         f'  <line x1="28" y1="{y0-12}" x2="852" y2="{y0-12}" stroke="{LINE}"/>']
    for i, (name, col) in enumerate(kinds):
        x = x0 + i * (bw + gap)
        b.append(f'  <rect x="{x}" y="{y0}" width="{bw}" height="52" rx="6" '
                 f'fill="{col}" fill-opacity="0.10" stroke="{col}" '
                 f'stroke-opacity="0.55" stroke-width="1">'
                 f'<animate attributeName="fill-opacity" '
                 f'values="0.10;0.42;0.10" dur="3s" begin="{i*0.18}s" '
                 f'repeatCount="indefinite"/></rect>')
        fs = 10 if len(name) > 6 else 12
        b.append(text(x + bw / 2, y0 + 32, name, fs, col, "middle", "600"))
    y2 = y0 + 108
    b.append(text(28, y2, "ROUTING", 12, TEXT, weight="700"))
    b.append(text(28, y2 + 18, "8 experts, 2 active per token", 11, MUTED))
    ex0, ey = 150, y2 - 12
    for e in range(8):
        ex = ex0 + e * 62
        active = e in (4, 6)
        col = GREEN if active else LINE
        op = "0.9" if active else "0.30"
        anim = (f'<animate attributeName="fill-opacity" '
                f'values="0.25;0.95;0.25" dur="1.6s" begin="{e*0.12}s" '
                f'repeatCount="indefinite"/>') if active else ""
        b.append(f'  <circle cx="{ex}" cy="{ey}" r="17" fill="{col}" '
                 f'fill-opacity="{op}">{anim}</circle>')
        b.append(text(ex, ey + 4, f"E{e}", 9, INK if active else MUTED,
                      "middle", "700"))
    for tgt in (4, 6):
        x1 = ex0 + tgt * 62
        b.append(f'  <path d="M {x1} {ey-18} L 760 {ey-44}" stroke="{GREEN}" '
                 f'stroke-width="1.2" stroke-opacity="0.5" fill="none" '
                 f'stroke-dasharray="4 4">'
                 f'<animate attributeName="stroke-dashoffset" from="16" to="0" '
                 f'dur="0.9s" repeatCount="indefinite"/></path>')
    b.append(f'  <rect x="760" y="{ey-68}" width="92" height="48" rx="6" '
             f'fill="{GREEN}" fill-opacity="0.12" stroke="{GREEN}" '
             f'stroke-opacity="0.6"/>')
    b.append(text(806, ey - 50, "top-2", 12, GREEN, "middle", "700"))
    b.append(text(806, ey - 32, "SwiGLU", 10, MUTED, "middle"))
    return svg(w, h, "\n".join(b), "Architecture",
               "Animated diagram of the 14-layer hybrid stack and sparse "
               "top-2-of-8 MoE routing.")


# --------------------------------------------------------------- curve -----
def curve() -> str:
    w, h = 880, 300
    x0, y0, pw, ph = 60, 40, 520, 210
    pts = [2.90, 2.85, 2.80, 2.78, 2.72, 2.70, 2.66, 2.64, 2.61, 2.59, 2.56,
           2.54, 2.51, 2.49, 2.47, 2.45, 2.43, 2.41, 2.39, 2.37, 2.35, 2.33]
    lo, hi = 2.2, 3.0

    def xy(i, v):
        return (x0 + i * pw / (len(pts) - 1),
                y0 + ph - (v - lo) / (hi - lo) * ph)

    d = " ".join(f"{'M' if i == 0 else 'L'} {xy(i, v)[0]:.1f} {xy(i, v)[1]:.1f}"
                 for i, v in enumerate(pts))
    b = [text(28, 26, "TRAINING LOSS", 13, TEXT, weight="700"),
         text(28, 44, "11k steps on 1x T4, single consumer GPU", 11, MUTED)]
    for gv in (2.2, 2.4, 2.6, 2.8, 3.0):
        _, gy = xy(0, gv)
        b.append(f'  <line x1="{x0}" y1="{gy:.1f}" x2="{x0+pw}" y2="{gy:.1f}" '
                 f'stroke="{LINE}" stroke-opacity="0.5"/>')
        b.append(text(x0 - 10, gy + 4, f"{gv:.1f}", 10, MUTED, "end"))
    b.append(f'  <path d="{d}" fill="none" stroke="{CYAN}" stroke-width="2" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    lx, ly = xy(len(pts) - 1, pts[-1])
    b.append(f'  <circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="{CYAN}">'
             f'<animate attributeName="r" values="3;6;3" dur="2s" '
             f'repeatCount="indefinite"/></circle>')
    b.append(text(lx + 14, ly + 4, "2.33", 13, CYAN, weight="700"))
    b.append(f'  <line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" '
             f'stroke="{LINE}"/>')
    bx = x0 + pw + 70
    b.append(text(bx, y0 + 24, "WHAT MOVED", 12, TEXT, weight="700"))
    for i, (k, v, c) in enumerate([("params", "47,640,968", AMBER),
                                   ("active/token", "38,793,608", GREEN),
                                   ("experts", "8 → 2", VIOLET),
                                   ("guide anneal", "2000 steps", CYAN),
                                   ("hardware", "1× T4", TEXT),
                                   ("from scratch", "yes", GREEN)]):
        b.append(text(bx, y0 + 52 + i * 26, k, 11, MUTED))
        b.append(text(852, y0 + 52 + i * 26, v, 12, c, "end", "600"))
    return svg(w, h, "\n".join(b), "Training loss",
               "Animated chart of training loss falling from 2.90 to 2.33.")


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

