#!/usr/bin/env python3
"""MiraLM — static parameter-budget gate (DAY-1 gate).

Projects the exact number of trainable parameters directly from a YAML config.
The layout math lives in src/budget.py (shared with tests and with
scripts/check_params.py, which counts the REAL model).

This gate must PASS (<= 50,000,000) BEFORE any training run starts, and its
report (results/param_budget_*.txt) is part of the submission requirements.

Usage:
    python scripts/param_budget.py                             # both configs
    python scripts/param_budget.py --config configs/model_sparsemind.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.budget import PARAM_LIMIT, build_lines, total_params, active_params  # noqa: E402
from src.config import ModelConfig  # noqa: E402

BYTES_FP16 = 2
BYTES_INT8 = 1


def fmt(n: int) -> str:
    return f"{n:,}"


def render(cfg: ModelConfig, cfg_path: str) -> tuple[str, int, bool]:
    lines = build_lines(cfg)
    total = total_params(cfg)
    active = active_params(cfg)
    ok = total <= PARAM_LIMIT

    W = 96
    c = 34  # component column width
    bar = "─" * W
    header = "MiraLM — static parameter-budget gate (pre-build projection)"
    headroom = PARAM_LIMIT - total
    fp16_mb = total * BYTES_FP16 / 1e6
    int8_mb = total * BYTES_INT8 / 1e6

    def row(s: str) -> str:
        return "│ " + s.ljust(W - 4) + " │"

    rows = [f"┌{bar}┐"]
    rows.append(f"│{header.center(W)}│")
    rows.append(row(f"config : {cfg_path}"))
    rows.append(row(f"model  : {cfg.name} · {cfg.model_type} · {cfg.n_layers} layers · tied embedding"))
    rows.append(f"├{bar}┤")
    rows.append(row(f"{'component':<{c}} {'count':>6} {'unit params':>13} {'total params':>15}   formula"))
    rows.append(row(f"{'-' * c} {'-' * 6} {'-' * 13} {'-' * 15}   {'-' * 22}"))
    for l in lines:
        rows.append(row(f"{l.name:<{c}} {l.count:>6} {fmt(l.unit):>13} {fmt(l.total):>15}   {l.formula}"))
    rows.append(f"├{bar}┤")
    status = f"✓ PASS  (headroom {fmt(headroom)})" if ok else f"✗ FAIL  (OVER BY {fmt(-headroom)})"
    rows.append(row(f"{'Trainable parameters':<{c}} {'':>6} {'':>13} {fmt(total):>15}"))
    rows.append(row(f"{'Budget limit':<{c}} {'':>6} {'':>13} {fmt(PARAM_LIMIT):>15}"))
    rows.append(row(f"{'Status':<{c}} {'':>6} {'':>13} {status:>15}"))
    rows.append(row(f"{'Active params (sparse routing)':<{c}} {'':>6} {'':>13} {fmt(active):>15}   ({active / total * 100:.1f}% of total)"))
    rows.append(row(f"{'Store size':<{c}} {'':>6} {'':>13} {f'fp16 ≈ {fp16_mb:.1f} MB':>15}   int8 ≈ {int8_mb:.1f} MB"))
    rows.append(f"└{bar}┘")
    return "\n".join(rows), total, ok


DEFAULT_CONFIGS = [
    "configs/model_sparsemind.yaml",
    "configs/model_dense_fallback.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="single config path")
    args = ap.parse_args()

    configs = DEFAULT_CONFIGS if not args.config else [args.config]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_ok = True
    for i, cfg_path in enumerate(configs):
        cfg = ModelConfig.from_yaml(ROOT / cfg_path)
        text, total, ok = render(cfg, cfg_path)
        print(text)
        print()
        all_ok = all_ok and ok

        (results_dir / f"param_budget_{cfg.model_type}.txt").write_text(text + "\n", encoding="utf-8")
        mode = "w" if i == 0 else "a"
        with open(results_dir / "param_budget_master.txt", mode, encoding="utf-8") as f:
            f.write(f"{cfg.name}\t{total:,}\t{active_params(cfg):,}\t{'PASS' if ok else 'FAIL'}\t{cfg_path}\n")

        if not ok:
            print(f"✗ {cfg.name}: {total:,} > {PARAM_LIMIT:,} — BLOCKED", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())