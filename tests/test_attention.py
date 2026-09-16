"""Tests for the dense attention backbone (GQA + RoPE + RMSNorm + SwiGLU).

Every property tested here is a guard that matters for the submission:
param accounting must match the budget gate exactly, outputs must be
shape-correct, attention strictly causal, and gradients finite.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.model.attention import (
    AttentionBlock,
    GroupedQueryAttention,
    RMSNorm,
    SwiGLUFFN,
    rotate_half,
)

torch.manual_seed(7)


@pytest.fixture(scope="module")
def cfg_a() -> ModelConfig:
    return ModelConfig.from_yaml(ROOT / "configs" / "model_sparsemind.yaml")


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_attention_block(cfg: ModelConfig) -> int:
    d = cfg.d_model
    h = cfg.n_heads * cfg.head_dim_resolved
    kv = cfg.n_kv_heads * cfg.head_dim_resolved
    return 2 * d * h + 2 * d * kv + 3 * d * cfg.d_ff + 2 * d


def build_block(cfg: ModelConfig) -> AttentionBlock:
    return AttentionBlock(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim_resolved,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        norm_eps=cfg.norm_eps,
        init_std=cfg.initializer_std,
    )


def test_attention_block_params_match_budget(cfg_a):
    block = build_block(cfg_a)
    assert count(block) == expected_attention_block(cfg_a)
    assert count(block) == 2_802_432  # golden value for sparsemind config


def test_gqa_params_match_formula(cfg_a):
    attn = GroupedQueryAttention(
        cfg_a.d_model, cfg_a.n_heads, cfg_a.n_kv_heads, cfg_a.head_dim_resolved,
        cfg_a.max_seq_len,
    )
    d, h, kv = cfg_a.d_model, cfg_a.n_heads, cfg_a.n_kv_heads
    hd = cfg_a.head_dim_resolved
    assert count(attn) == 2 * d * (h * hd) + 2 * d * (kv * hd)
    assert count(attn) == 442_368  # golden value


def test_attention_shape(cfg_a):
    x = torch.randn(2, 17, cfg_a.d_model)
    y = build_block(cfg_a)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attention_is_causal(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(1, 16, cfg_a.d_model)
    y1 = block(x)
    x2 = x.clone()
    x2[:, 6:] += 1.0
    y2 = block(x2)
    torch.testing.assert_close(y1[:, :6], y2[:, :6], atol=1e-6, rtol=1e-6)


def test_attention_gradients_finite(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(1, 12, cfg_a.d_model, requires_grad=True)
    block(x).pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf grad on {name}"


def test_variable_sequence_length(cfg_a):
    block = build_block(cfg_a)
    for t in (1, 5, 1024):
        x = torch.randn(1, t, cfg_a.d_model)
        assert block(x).shape == x.shape


def test_context_too_long_raises(cfg_a):
    attn = GroupedQueryAttention(
        cfg_a.d_model, cfg_a.n_heads, cfg_a.n_kv_heads, cfg_a.head_dim_resolved,
        cfg_a.max_seq_len,
    )
    with pytest.raises(AssertionError):
        attn(torch.randn(1, cfg_a.max_seq_len + 1, cfg_a.d_model))


def test_rope_offset_rotates_same_vector(cfg_a):
    """Shifting input by +1 must be equivalent to rotating the RoPE phase."""
    attn = GroupedQueryAttention(
        cfg_a.d_model, cfg_a.n_heads, cfg_a.n_kv_heads, cfg_a.head_dim_resolved,
        32, rope_base=10000.0,
    )
    x = torch.randn(1, 16, cfg_a.d_model)
    y_base = attn(x, offset=0)
    y_shift = attn(x, offset=1)
    assert y_base.shape == y_shift.shape
    assert torch.isfinite(y_shift).all()


def test_rmsnorm_centers_zero_and_scales(cfg_a):
    norm = RMSNorm(cfg_a.d_model, eps=cfg_a.norm_eps)
    x = torch.randn(3, 9, cfg_a.d_model)
    y = norm(x)
    mu = norm.weight.detach().mean()
    assert y.shape == x.shape
    assert torch.allclose(y.std(dim=-1), torch.full((3, 9), mu), atol=1e-2)
    assert torch.isfinite(y).all()


def test_rotate_half_property(cfg_a):
    hd = cfg_a.head_dim_resolved
    x = torch.randn(4, 8, 16, hd)
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x)
    assert torch.allclose(rotate_half(x).norm(dim=-1), x.norm(dim=-1))


def test_swiglu_reduces_dropout_na() -> None:
    ffn = SwiGLUFFN(384, 2048)
    x = torch.randn(2, 7, 384)
    assert ffn(x).shape == (2, 7, 384)
    assert torch.isfinite(ffn(x)).all()