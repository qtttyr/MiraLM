"""Sparse top-k Mixture-of-Experts capstone with semantically-seeded routing.

Parameter accounting mirrors scripts/param_budget.py -> moe_unit():
    per expert  : 3 * d_model * expert_dim    (SwiGLU)
    router      : d_model * n_experts (+ n_experts if router_bias)
    gate norm   : d_model

The router is seeded by domain-tagged batches: a guide loss annealed over the
first training steps concentrates each sample's routing mass on its domain
expert, then decays to zero so the router is free afterwards. Expert loads are
kept balanced, and the heatmap in the demo surfaces routing live.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SwiGLUFFN

EXPERT_DOMAINS = [
    "math", "code", "logic", "commonsense",
    "sql", "json", "general_1", "general_2",
]


class Router(nn.Module):
    def __init__(self, d_model: int, n_experts: int, router_bias: bool, init_std: float = 0.02):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=router_bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        if router_bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Z-loss (ST-MoE / DeepSeek style): penalizes exploding router logits."""
    logsumexp = torch.logsumexp(logits, dim=-1)
    return logsumexp.square().mean()


def load_balance_loss(probs: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
    """Switch-transformer auxiliary loss: n_experts * sum_i f_i * P_i."""
    e = probs.shape[-1]
    f = torch.zeros(e, device=probs.device)
    p = torch.zeros(e, device=probs.device)
    for i in range(e):
        selected = (top_idx == i).any(dim=-1)
        f[i] = selected.float().mean()
        if selected.any():
            p[i] = probs[selected, i].mean()
    return (f * p).sum() * e


def guide_loss(probs: torch.Tensor, domain_per_token: torch.Tensor, guide_w: float,
               tail_eps: float = 1e-4) -> torch.Tensor:
    """Annealed curriculum: softly pin each token to its domain expert."""
    n_tokens, e = probs.shape
    target = torch.full_like(probs, tail_eps)
    target[torch.arange(n_tokens, device=probs.device), domain_per_token.clamp(0, e - 1)] = 1.0 - (e - 1) * tail_eps
    return F.kl_div(F.log_softmax(probs, dim=-1), target, reduction="batchmean") * guide_w


class MoECapstone(nn.Module):
    """Applies the sparse experts to every token (no token dropping).

    Returns the residual output plus a metrics dict consumed by the trainer:
    z_loss, aux_loss, guide_loss, expert_load, and optionally router_probs.
    """

    def __init__(self, d_model: int, n_experts: int, n_experts_active: int,
                 expert_dim: int, router_bias: bool, init_std: float = 0.02):
        super().__init__()
        assert n_experts_active < n_experts
        self.n_experts = n_experts
        self.n_experts_active = n_experts_active

        self.router = Router(d_model, n_experts, router_bias, init_std)
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, expert_dim, init_std=init_std) for _ in range(n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        guide_w: float = 0.0,
        return_probs: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        tokens = x.reshape(b * t, d)

        logits = self.router(tokens)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=self.n_experts_active, dim=-1)

        out = torch.zeros_like(tokens)
        for e in range(self.n_experts):
            selected = (top_idx == e).any(dim=-1)
            if selected.any():
                expert_out = self.experts[e](tokens[selected])
                w = probs[selected, e].unsqueeze(-1)
                out[selected] = out[selected] + w * expert_out

        metrics = {
            "z_loss": z_loss(logits),
            "aux_loss": load_balance_loss(probs, top_idx),
        }
        if domain is not None and guide_w > 0:
            domain_per_token = domain.repeat_interleave(t)
            metrics["guide_loss"] = guide_loss(probs, domain_per_token, guide_w)

        load = (top_idx[:, :1] == torch.arange(self.n_experts, device=x.device)
                .view(1, -1)).sum(dim=0).float()
        metrics["expert_load"] = load / max(b * t, 1)

        if return_probs:
            metrics["router_probs"] = probs.view(b, t, self.n_experts)

        return x + out.view(b, t, d), metrics