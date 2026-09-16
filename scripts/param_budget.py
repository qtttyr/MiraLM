#!/usr/bin/env python3
"""MiraLM — static parameter-budget gate (DAY-1 gate).

Projects the exact number of trainable parameters directly from a YAML config
using the same layout math that src/model implements.

This gate must PASS (<= 50,000,000) BEFORE any training run starts, and its
report (results/param_budget_*.txt) is part of the submission requirements.
The authoritative check is scripts/check_params.py, which builds the real
module graph and counts param.numel(); tests/test_budget.py keeps the two
aligned.

Usage:
    python scripts/param_budget.py                            # both configs
    python scripts/param_budget.py --config configs/model_sparsemind.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig  # noqa: E402

PARAM_LIMIT = 50_000_000
BYTES_FP16 = 2
BYTES_INT8 = 1


@dataclass
class BudgetLine:
    name: str
    count: int
    unit: int
    formula: str

    @property
    def total(self) -> int:
        return self.count * self.unit


def mamba_block_unit(cfg: ModelConfig) -> int:
    d = cfg.d_model
    di = cfg.mamba_d_inner
    dt = cfg.mamba_dt_rank
    st = cfg.mamba.d_state
    cv = cfg.mamba.d_conv
    return (
        d * 2 * di                 # in_proj -> [x; z]
        + di * cv + di             # depthwise conv1d (groups=di) + bias
        + di * (dt + 2 * st)       # x_proj -> [dt; B; C]
        + dt * di + di             # dt_proj (softplus delta) + bias
        + st * di                  # A_log
        + di                       # D
        + di * d                   # out_proj
    )


def attention_block_unit(cfg: ModelConfig) -> int:
    d = cfg.d_model
    h = cfg.n_heads * cfg.head_dim_resolved
    kv = cfg.n_kv_heads * cfg.head_dim_resolved
    return (
        d * h * 2                  # q_proj + o_proj
        + d * kv * 2               # k_proj + v_proj
        + 3 * d * cfg.d_ff         # SwiGLU (gate/up/down)
        + 2 * d                    # pre-attn + pre-ffn RMSNorm
    )


def moe_unit(cfg: ModelConfig) -> int:
    d = cfg.d_model
    expert = 3 * d * cfg.moe.expert_dim  # SWiGLU per expert
    router = d * cfg.moe.n_experts + (cfg.moe.n_experts if cfg.moe.router_bias else 0)
    return expert, router


def build_lines(cfg: ModelConfig) -> list[BudgetLine]:
    d = cfg.d_model
    lines: list[BudgetLine] = []

    lines.append(BudgetLine(
        "Embedding (tied, output head shared)", 1, d * cfg.vocab_size,
        f"vocab_size × d_model",
    ))

    if cfg.use_mamba:
        lines.append(BudgetLine(
            "Mamba block (SSM)", cfg.n_mamba_layers, mamba_block_unit(cfg),
            "in_proj + depthwise-conv + x_proj + dt_proj + A_log + D + out_proj",
        ))
        lines.append(BudgetLine(
            "Mamba pre-norm (RMSNorm)", cfg.n_mamba_layers, d,
            "1 × d_model per mamba block",
        ))

    lines.append(BudgetLine(
        "Attention block (GQA + RoPE + SwiGLU)", cfg.n_attention_layers,
        attention_block_unit(cfg),
        "q/o + k/v projections + SwiGLU(3·d·d_ff) + 2 norms",
    ))

    if cfg.use_moe:
        expert_unit, router_unit = moe_unit(cfg)
        lines.append(BudgetLine(
            "MoE experts (SWiGLU)", cfg.moe.n_experts, expert_unit,
            "3 · d_model · expert_dim per expert",
        ))
        lines.append(BudgetLine(
            "MoE router + semantic biases", 1, router_unit,
            "d_model · n_experts + n_experts biases",
        ))

    lines.append(BudgetLine("Final norm (RMSNorm)", 1, d, "1 × d_model"))
    return lines


def fmt(n: int) -> str:
    return f"{n:,}"


def render(cfg: ModelConfig, cfg_path: str) -> tuple[str, int, int, bool]:
    lines = build_lines(cfg)
    total = sum(l.total for l in lines)
    ok = total <= PARAM_LIMIT

    active = total
    if cfg.use_moe:
        active = sum(l.total for l in lines if l.name != "MoE experts (SWiGLU)")
        expert_unit, _ = moe_unit(cfg)
        active += cfg.moe.n_experts_active * expert_unit

    W = 92
    c = 32  # component column width
    bar = "─" * W
    header = f"MiraLM — static parameter-budget gate (pre-build projection)"
    headroom = PARAM_LIMIT - total

    rows = [f"┌{bar}┐"]
    rows.append(f"│{header.center(W)}│")
    rows.append(f"│ config  : {cfg_path!s:<{W - 11}}│")
    rows.append(f"│ model   : {cfg.name!s:<{W - 11}} · {cfg.model_type} · {cfg.n_layers} layers· tied emb".ljust(W) + "│")
    rows.append(f"├{bar}┤")
    rows.append(f"│ {'component':<{c}} {'count':>6} {'unit params':>13} {'total params':>15}   formula".ljust(W) + "│")
    rows.append(f"│ {'-'*c} {'-'*6} {'-'*13} {'-'*15}   {'-'*22}".ljust(W) + "│")
    for l in lines:
        row = f"│ {l.name:<{c}} {l.count:>6} {fmt(l.unit):>13} {fmt(l.total):>15}   {l.formula}".ljust(W) + "│"
        rows.append(row)
    rows.append(f"├{bar}┤")
    rows.append(f"│ {'Trainable parameters':<{c}} {'':>6} {'':>13} {fmt(total):>15}   (0-shot gate) ───".ljust(W) + "│")
    rows.append(f"│ {'Budget':<{c}} {'':>6} {'':>13} {fmt(PARAM_LIMIT):>15}   hard limit".ljust(W) + "│")
    status = "✓ PASS  (headroom {:,})".format(headroom) if ok else "✗ FAIL  (OVER BY {:,})".format(-headroom)
    rows.append(f"│ {'Status':<{c}} {'':>6} {'':>13} {status:>15}   ".ljust(W) + "│")
    rows.append(f"│ {'Active params (sparse routing)':<{c}} {'':>6} {'':>13} {fmt(active):>15}   ({active/total*100:.1f}% of total)".ljust(W) + "│")
    fp16_mb = total * BYTES_FP16 / 1e6
    int8_mb = total * BYTES_INT8 / 1e6
    rows.append(f"│ {'Store size':<{c}} {'':>6} {'':>13} {f'fp16 ≈ {fp16_mb:.1f} MB':>15}   int8 ≈ {int8_mb:.1f} MB".ljust(W) + "│")
    rows.append(f"└{bar}┘")
    return "\n".join(rows), total, active, ok


DEFAULT_CONFIGS = [
    "configs/model_sparsemind.yaml",
    "configs/model_dense_fallback.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="single config path")
    args = ap.parse_args()

    configs = [args.config] if args.config else DEFAULT_CONFIGS
    if not configs:
        print("ERROR: no configs found", file=sys.stderr)
        return 2

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_ok = True
    for cfg_path in configs:
        cfg = ModelConfig.from_yaml(ROOT / cfg_path)
        text, total, active, ok = render(cfg, cfg_path)
        print(text)
        print()
        all_ok = all_ok and ok

        out = results_dir / f"param_budget_{cfg.model_type}.txt"
        out.write_text(text + "\n", encoding="utf-8")
        summary = results_dir / "param_budget_master.txt"
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"{cfg.name}\t{total:,}\t{active:,}\t{'PASS' if ok else 'FAIL'}\t{cfg_path}\n")

        if not ok:
            print(f"✗ {cfg.name}: {total:,} > {PARAM_LIMIT:,} — BLOCKED", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())