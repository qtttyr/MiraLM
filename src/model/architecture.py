"""Full model architecture: hybrid stack + MoE capstone + weight tying.

Builds n_layers alternating Mamba / Attention per cfg.layer_role(i),
appends an optional MoECapstone (gated by cfg.use_moe), a final RMSNorm,
and a tied lm_head. All trainable parameters are counted in src/budget.py;
tests/test_architecture.py enforces real-numel == projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from .attention import AttentionBlock, RMSNorm
from .mamba_block import MambaBlock
from .moe import MoECapstone


@dataclass
class ModelOutput:
    logits: torch.Tensor
    metrics: Optional[dict] = None


class MiraLM(nn.Module):
    """The full ≤50M model. lm_head.weight is tied to embed_tokens.weight."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=cfg.initializer_std)

        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            if cfg.layer_role(i) == "mamba":
                self.blocks.append(
                    MambaBlock(
                        d_model=d,
                        d_state=cfg.mamba.d_state,
                        d_conv=cfg.mamba.d_conv,
                        expand=cfg.mamba.expand,
                        dt_rank=cfg.mamba_dt_rank,
                        init_std=cfg.initializer_std,
                        norm_eps=cfg.norm_eps,
                    )
                )
            else:
                self.blocks.append(
                    AttentionBlock(
                        d_model=d,
                        n_heads=cfg.n_heads,
                        n_kv_heads=cfg.n_kv_heads,
                        head_dim=cfg.head_dim_resolved,
                        d_ff=cfg.d_ff,
                        max_seq_len=cfg.max_seq_len,
                        norm_eps=cfg.norm_eps,
                        init_std=cfg.initializer_std,
                    )
                )

        self.moe_gate_norm: Optional[RMSNorm] = None
        self.moe: Optional[MoECapstone] = None
        if cfg.use_moe:
            self.moe_gate_norm = RMSNorm(d, eps=cfg.norm_eps)
            self.moe = MoECapstone(
                d_model=d,
                n_experts=cfg.moe.n_experts,
                n_experts_active=cfg.moe.n_experts_active,
                expert_dim=cfg.moe.expert_dim,
                router_bias=cfg.moe.router_bias,
                init_std=cfg.initializer_std,
            )

        self.final_norm = RMSNorm(d, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=cfg.initializer_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        guide_w: float = 0.0,
        return_probs: bool = False,
    ) -> ModelOutput:
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x)

        metrics: Optional[dict] = None
        if self.moe is not None:
            moe_in = x if self.moe_gate_norm is None else self.moe_gate_norm(x)
            x, metrics = self.moe(
                moe_in,
                domain=domain,
                guide_w=guide_w,
                return_probs=return_probs,
            )
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return ModelOutput(logits, metrics)