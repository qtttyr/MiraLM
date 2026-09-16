"""Interactive demo + screenshots for the GIBC V2 submission video.

Usage:
    # Interactive single-shot
    python scripts/demo.py --ckpt-dir checkpoints/mira-sft/last \
        --prompt 'Return a JSON with id=1 and name=Alice'

    # Fixed demo sequence (printing all three domains)
    python scripts/demo.py --ckpt-dir checkpoints/mira-sft/last --all

    # Heatmap (requires a model checkpoint with trained MoE)
    python scripts/demo.py --ckpt-dir checkpoints/mira/last --heatmap results/heatmap.png

    # Loss curve screenshot from trace.csv
    python scripts/demo.py --trace checkpoints/mira/trace.csv \
        --curve-out results/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.prompts import DEMO_PROMPTS, DEMO_INTRO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="run the fixed demo prompts")
    ap.add_argument("--heatmap", type=Path, default=None, help="save expert heatmap PNG")
    ap.add_argument("--trace", type=Path, default=None, help="trainer trace.csv for loss curve")
    ap.add_argument("--curve-out", type=Path, default=None, help="save loss curve PNG")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    # ---- trace / curve screenshot (no checkpoint needed) -------------------
    if args.trace is not None and args.curve_out is not None:
        from src.demo.heatmap import plot_trace
        plot_trace(args.trace, args.curve_out)
        print(f"loss curve saved: {args.curve_out}")

    if args.prompt is None and not args.all and args.heatmap is None:
        print("nothing to do: pass --prompt, --all, --heatmap, or --trace+--curve-out")
        return 0

    # ---- interactive / all / heatmap (require checkpoint) ------------------
    from src.demo.runner import MiraDemoModel
    demo = MiraDemoModel(args.ckpt_dir)

    if args.heatmap is not None:
        from src.demo.heatmap import expert_domain_matrix, render_heatmap
        from src.demo.prompts import DEMO_PROMPTS
        mat, erows, dcols = expert_domain_matrix(
            demo.model, demo.tokenizer, DEMO_PROMPTS, max_len=64, device=demo.device,
        )
        render_heatmap(mat, args.heatmap, dcols, erows)
        print(f"heatmap saved: {args.heatmap}")

    if args.prompt:
        out = demo.generate(args.prompt, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature)
        vis, hid = demo.split_answer(out)
        print(f"\nprompt: {args.prompt}\nanswer: {vis.strip()}")
        if hid:
            print(f"hidden: {hid.strip()}")
        return 0

    if args.all:
        print(DEMO_INTRO)
        for domain, prompts in DEMO_PROMPTS.items():
            print(f"\n--- {domain.upper()} ---")
            for p in prompts:
                out = demo.generate(p, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature)
                vis, _ = demo.split_answer(out)
                print(f"Q: {p}")
                print(f"A: {vis.strip()}")
            print()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())