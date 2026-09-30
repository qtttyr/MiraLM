#!/usr/bin/env python3
"""MiraLM — record REAL model outputs for the submission demo.

The landing page's console must not ship invented text: the rules require the
submission to be the model itself, and a jury that discovers the "live" console
was hand-written would discount everything else. This script runs the real SFT
checkpoint over the curated demo prompts and writes the transcript to JSON, so
the site can replay genuine generations and say so honestly.

    python scripts/record_demo.py --ckpt-dir checkpoints/mira-sft/last \
        --output results/demo_outputs.json

Greedy decoding is used on purpose: sampled text is not reproducible, and a
recorded transcript the jury can diff against a fresh run is worth more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.prompts import DEMO_PROMPTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("results/demo_outputs.json"))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from src.demo.runner import MiraDemoModel

    demo = MiraDemoModel(args.ckpt_dir, device=args.device)

    entries = []
    for domain, prompts in DEMO_PROMPTS.items():
        for p in prompts:
            raw = demo.generate(p, max_new_tokens=args.max_new_tokens, do_sample=False)
            visible, hidden = demo.split_answer(raw)
            entries.append({
                "domain": domain,
                "prompt": p,
                "visible_answer": visible.strip(),
                "hidden_reasoning": hidden.strip(),
            })
            print(f"[{domain}] {visible.strip()[:90]}", file=sys.stderr)

    out = {
        "checkpoint": str(args.ckpt_dir),
        "decoding": "greedy (do_sample=False)",
        "max_new_tokens": args.max_new_tokens,
        "note": "verbatim model output; no hand editing",
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nrecorded {len(entries)} real generations -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
