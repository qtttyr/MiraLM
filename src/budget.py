"""Single source of truth for the ≤50M parameter budget.

Pure layout math, dependency-light (no torch). Consumed by:
    scripts/param_budget.py   -> static projection report
    scripts/check_params.py   -> authoritative gate on the REAL model
    tests/test_architecture.py-> cross-check: real numel == projection

Every formula here is mirrored 1:1 by the module constructors in
src/model/ (see accounting notes in each module).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig

PARAM_LIMIT = 50_000_000


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
        + st * di                  # A_log (di, st)
        + di                       # D
        + di * d                   # out_proj
        + d                        # in-block pre-norm RMSNorm
    )


def attention_block_unit(cfg: ModelConfig) -> int:
    d = cfg.d_model
    h = cfg.n_heads * cfg.head_dim_resolved
    kv = cfg.n_kv_heads * cfg.head_dim_resolved
    return (
        2 * d * h                  # q_proj + o_proj
        + 2 * d * kv               # k_proj + v_proj
        + 3 * d * cfg.d_ff         # SwiGLU (gate/up/down)
        + 2 * d                    # pre-attn + pre-ffn RMSNorm
    )


def moe_lines(cfg: ModelConfig) -> tuple[BudgetLine, BudgetLine, BudgetLine]:
    d = cfg.d_model
    expert = 3 * d * cfg.moe.expert_dim
    router = d * cfg.moe.n_experts + (cfg.moe.n_experts if cfg.moe.router_bias else 0)
    return (
        BudgetLine("MoE experts (SwiGLU)", cfg.moe.n_experts, expert,
                   "3 · d_model · expert_dim per expert"),
        BudgetLine("MoE router + semantic biases", 1, router,
                   "d_model · n_experts + n_experts biases"),
        BudgetLine("MoE gate norm (RMSNorm)", 1, d, "1 × d_model"),
    )


def build_lines(cfg: ModelConfig) -> list[BudgetLine]:
    d = cfg.d_model
    lines: list[BudgetLine] = []

    lines.append(BudgetLine(
        "Embedding (tied, output head shared)", 1, d * cfg.vocab_size,
        "vocab_size × d_model",
    ))

    if cfg.use_mamba:
        lines.append(BudgetLine(
            "Mamba block (SSM, incl. pre-norm)", cfg.n_mamba_layers, mamba_block_unit(cfg),
            "in_proj + depthwise-conv + x_proj + dt_proj + A_log + D + out_proj + norm",
        ))

    lines.append(BudgetLine(
        "Attention block (GQA + RoPE + SwiGLU)", cfg.n_attention_layers,
        attention_block_unit(cfg),
        "q/o projections + k/v projections + SwiGLU(3·d·d_ff) + 2 norms",
    ))

    if cfg.use_moe:
        lines.extend(moe_lines(cfg))

    lines.append(BudgetLine("Final norm (RMSNorm)", 1, d, "1 × d_model"))
    return lines


def total_params(cfg: ModelConfig) -> int:
    return sum(line.total for line in build_lines(cfg))


def active_params(cfg: ModelConfig) -> int:
    """Params actually exercised (sparse view): dense backbone + active experts."""
    total = sum(line.total for line in build_lines(cfg))
    if not cfg.use_moe:
        return total
    per_expert = 3 * cfg.d_model * cfg.moe.expert_dim
    return total - cfg.moe.n_experts * per_expert + cfg.moe.n_experts_active * per_expert


def headroom(cfg: ModelConfig) -> int:
    return PARAM_LIMIT - total_params(cfg)


def is_within_budget(cfg: ModelConfig) -> bool:
    return total_params(cfg) <= PARAM_LIMIT