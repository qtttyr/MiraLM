"""Dense attention backbone: Grouped-Query Attention + RoPE + SwiGLU + RMSNorm.

Parameter accounting mirrors scripts/param_budget.py -> attention_block_unit():
    q/o projections : 2 * d_model * (n_heads   * head_dim)
    k/v projections : 2 * d_model * (n_kv_heads * head_dim)
    SwiGLU FFN      : 3 * d_model * d_ff
    RMSNorms        : 2 * d_model (pre-attn + pre-ffn)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no bias, learnable scale only)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(self.weight.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), cos/sin cached for full context.

    No trainable parameters — excluded from the parameter budget.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim//2)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))  # (1,1,T,d/2)
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

    def _slice(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        assert seq_len + offset <= self.max_seq_len, f"seq_len {seq_len}+{offset} > max_seq_len {self.max_seq_len}"
        cos = self.cos_cached[:, :, offset : offset + seq_len]  # (1,1,T,d/2)
        sin = self.sin_cached[:, :, offset : offset + seq_len]
        cos = cos.repeat_interleave(2, dim=-1)  # (1,1,T,H): c1,c1,c2,c2,...
        sin = sin.repeat_interleave(2, dim=-1)
        return cos, sin

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos, sin = self._slice(seq_len, offset)
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA key/value heads to the query head count (variance-preserving)."""
    if n_rep == 1:
        return x
    b, h, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, hd).reshape(b, h * n_rep, t, hd)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 max_seq_len: int, rope_base: float = 10000.0, init_std: float = 0.02):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, base=rope_base)

        for w in (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.o_proj.weight):
            nn.init.normal_(w, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        b, t, _ = x.shape
        h, kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, kv, hd).transpose(1, 2)

        q, k = self.rope.rotate(q, k, offset=offset)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,h,T,hd)
        attn = attn.transpose(1, 2).contiguous().view(b, t, h * hd)
        return self.o_proj(attn)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: down(SiLU(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int, init_std: float = 0.02):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        for w in (self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight):
            nn.init.normal_(w, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AttentionBlock(nn.Module):
    """Pre-norm transformer block: norm->attn (residual), norm->ffn (residual)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 d_ff: int, max_seq_len: int, norm_eps: float = 1e-5,
                 rope_base: float = 10000.0, init_std: float = 0.02):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim,
                                          max_seq_len, rope_base=rope_base, init_std=init_std)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, d_ff, init_std=init_std)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), offset=offset)
        x = x + self.ffn(self.norm2(x))
        return x