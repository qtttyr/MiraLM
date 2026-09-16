#!/usr/bin/env python3
"""MiraLM — authoritative parameter gate on the REAL model.

Builds the actual MiraLM module and counts trainable parameters via
torch numel(), then cross-validates against the static projection from
src/budget.py. The counts MUST match exactly — the static budget is only
trustworthy if it reproduces the real weights 1:1.

Also verifies weight tying and reports the sparse active-parameter view.

Usage:
    python scripts/check_params.py                               # both configs
    python scripts/check_params.py --config configs/model_sparsemind.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import budget as B  # noqa: E402
from src.config import ModelConfig  # noqa: E402
from src.model.architecture import MiraLM  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/model_sparsemind.yaml",
    "configs/model_dense_fallback.yaml",
]


def fmt(n: int) -> str:
    return f"{n:,}"


def check(cfg_path: str) -> tuple[bool, str]:
    cfg = ModelConfig.from_yaml(ROOT / cfg_path)
    torch.manual_seed(0)

    model = MiraLM(cfg)
    real = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projected = B.total_params(cfg)
    active = B.active_params(cfg)
    tied = model.lm_head.weight is model.embed_tokens.weight

    match = real == projected
    ok = real <= B.PARAM_LIMIT
    headroom = B.PARAM_LIMIT - real
    fp16_mb = real * 2 / 1e6

    W = 76
    bar = "─" * W
    rows = [f"┌{bar}┐",
            f"│{'MiraLM — authoritative parameter gate (real model)'.center(W)}│",
            f"│ {('config: ' + cfg_path).ljust(W - 2)} │",
            f"│ {('model : ' + cfg.name + ' · ' + cfg.model_type).ljust(W - 2)} │",
            f"├{bar}┤",
            f"│ {('real  trainable params : ' + fmt(real)).ljust(W - 2)} │",
            f"│ {('static projection      : ' + fmt(projected)).ljust(W - 2)} │",
            f"│ {('projection == real     : ' + ('✓ yes' if match else '✗ NO — MISMATCH')).ljust(W - 2)} │",
            f"│ {('budget limit           : ' + fmt(B.PARAM_LIMIT)).ljust(W - 2)} │",
            f"│ {('headroom               : ' + fmt(headroom)).ljust(W - 2)} │",
            f"│ {('status                 : ' + ('✓ PASS' if ok else '✗ FAIL')).ljust(W - 2)} │",
            f"│ {('weight tying (lm_head) : ' + ('✓ tied' if tied else 'untied')).ljust(W - 2)} │",
            f"│ {('active params (sparse) : ' + fmt(active)).ljust(W - 2)} │",
            f"│ {('store size (fp16)      : ' + f'≈ {fp16_mb:.1f} MB').ljust(W - 2)} │",
            f"└{bar}┘"]
    text = "\n".join(rows)
    if not match:
        text += f"\n⚠  real count {fmt(real)} differs from projection {fmt(projected)} — budget must be fixed\n"
    return ok and match, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="single config path")
    args = ap.parse_args()

    configs = DEFAULT_CONFIGS if not args.config else [args.config]
    all_ok = True
    for cfg_path in configs:
        ok, text = check(cfg_path)
        print(text)
        print()
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())