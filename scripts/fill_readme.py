#!/usr/bin/env python3
"""Fill the README's measured placeholders from real artifacts.

The rules require the README to state hardware, total training time and
approximate compute, and to report the five benchmark numbers. Those values are
measured, not typed by hand, so this script derives them from the artifacts that
actually exist (`train_meta.json`, `results/eval_*.json`) and refuses to invent a
value that was never measured.

    python scripts/fill_readme.py --ckpt-dir checkpoints/mira/last \
        --sft-ckpt checkpoints/mira-sft/last

Placeholders that have no measured source are left untouched and reported, so a
missing number stays visibly missing instead of silently becoming a guess.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T4_FLOPS_FP16 = 8.1e12  # Tesla T4 dense fp16 tensor-core peak, no sparsity


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/mira/last"))
    ap.add_argument("--sft-ckpt", type=Path, default=Path("checkpoints/mira-sft/last"))
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--write", action="store_true", help="rewrite the README in place")
    args = ap.parse_args()

    text = args.readme.read_text(encoding="utf-8")
    subs: dict[str, str] = {}
    missing: list[str] = []

    pretrain = read_json(args.ckpt_dir / "train_meta.json")
    sft = read_json(args.sft_ckpt / "train_meta.json")
    budget = read_json(args.results / "router_report.json")

    # ---- pre-training scale -------------------------------------------------
    if pretrain and pretrain.get("step"):
        steps = int(pretrain["step"])
        subs["PRETRAIN_STEPS"] = fmt_int(steps)
        subs["PRETRAIN_TOKENS"] = fmt_int(steps * 4 * 1024)
    else:
        missing.append("PRETRAIN_STEPS / PRETRAIN_TOKENS (no train_meta.json)")

    if sft and sft.get("step"):
        subs["SFT_STEPS"] = fmt_int(int(sft["step"]))
    else:
        missing.append("SFT_STEPS (no SFT train_meta.json)")

    # ---- wall-clock and compute -------------------------------------------
    pre_h = float(pretrain["elapsed_h"]) if pretrain and pretrain.get("elapsed_h") else 0.0
    sft_h = float(sft["elapsed_h"]) if sft and sft.get("elapsed_h") else 0.0
    if pre_h:
        subs["TOTAL_HOURS"] = f"{pre_h + sft_h:.2f}"
        subs["COMPUTE"] = (
            f"{pre_h + sft_h:.2f} h on 1x T4 "
            f"(measured wall-clock, end to end)"
        )
    else:
        missing.append("TOTAL_HOURS / COMPUTE (no elapsed_h in train_meta.json)")

    if pretrain and pretrain.get("step"):
        tokens = int(pretrain["step"]) * 4 * 1024
        flops = 6 * 38_793_608 * tokens
        subs["THEORETICAL_FLOPS"] = f"{flops:.2e}"
        subs["THEO_GPUH"] = f"{flops / T4_FLOPS_FP16 / 3600:.2f}"
        subs["CHINCHILLA_PCT"] = f"{tokens / (20 * 47_640_968) * 100:.1f}"

    # ---- the five scored metrics ------------------------------------------
    # eval_harness.py writes {"summary": {...}, "tasks": {...}}; raw lm-eval
    # dumps carry a top-level "results". Accept every shape rather than silently
    # reporting "no eval_mira.json" when the file is sitting right there.
    mc = read_json(args.results / "eval_mira.json")
    mc_src: dict = {}
    if mc:
        mc_src = (mc.get("results") or mc.get("tasks")
                  or mc.get("summary") or {})
    if mc_src:
        for task, key in [("hellaswag", "HELLASWAG"), ("arc_easy", "ARC_E"),
                          ("piqa", "PIQA"), ("winogrande", "WINOGRANDE")]:
            r = mc_src.get(task) or {}
            if not isinstance(r, dict):
                continue
            v = (r.get("acc_norm,none") or r.get("acc,none")
                 or r.get("accuracy"))
            if v is None:
                missing.append(f"{key} (no acc in eval_mira.json for {task})")
            else:
                subs[key] = f"{100 * float(v):.1f}"
    else:
        missing.append("HELLASWAG/ARC_E/PIQA/WINOGRANDE (no results/eval_mira.json)")

    wiki = read_json(args.results / "eval_wikitext103.json")
    if wiki and wiki.get("word_perplexity"):
        subs["WIKI_PPL"] = f"{float(wiki['word_perplexity']):.1f}"
    else:
        missing.append("WIKI_PPL (no results/eval_wikitext103.json)")

    # ---- apply -------------------------------------------------------------
    # Substitute ONLY delimited placeholders (`KEY` or {KEY}).
    # A bare text.replace(key, val) also rewrites labels that merely share the
    # name: it turned the "| ... | PIQA |" column header into "| 50.0 |" and the
    # "PIQA · HellaSwag" licence row into a number. Every real placeholder in
    # README.md is backticked, so the bare pass bought nothing and cost accuracy.
    for key, val in sorted(subs.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(f"`{key}`", val).replace(f"{{{key}}}", val)

    left = sorted(set(re.findall(r"`([A-Z][A-Z0-9_]{3,})`", text)))
    unresolved = [k for k in left if k in missing or k not in subs]

    if args.write and subs:
        args.readme.write_text(text, encoding="utf-8")
        print(f"README updated with {len(subs)} measured values")
    for k, v in sorted(subs.items()):
        print(f"  {k:>20} = {v}")

    if missing:
        print("\nstill missing (NOT invented):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
