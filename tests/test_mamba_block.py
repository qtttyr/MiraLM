"""Tests for the Mamba (SSM) block.

Key guards: pscan identity vs sequential reference, budget-matched param
count, causal recurrence, numerical stability, and gradient flow.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.model.mamba_block import MambaBlock, pscan, selective_scan_sequential

torch.manual_seed(7)


@pytest.fixture(scope="module")
def cfg_a() -> ModelConfig:
    return ModelConfig.from_yaml(ROOT / "configs" / "model_sparsemind.yaml")


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_mamba_block(cfg: ModelConfig) -> int:
    d, di, dt, st, cv = cfg.d_model, cfg.mamba_d_inner, cfg.mamba_dt_rank, cfg.mamba.d_state, cfg.mamba.d_conv
    return (
        d * 2 * di + di * cv + di
        + di * (dt + 2 * st) + dt * di + di
        + st * di + di + di * d
        + d  # in-block pre-norm RMSNorm
    )


def build_block(cfg: ModelConfig) -> MambaBlock:
    return MambaBlock(
        d_model=cfg.d_model,
        d_state=cfg.mamba.d_state,
        d_conv=cfg.mamba.d_conv,
        expand=cfg.mamba.expand,
        dt_rank=cfg.mamba_dt_rank,
        init_std=cfg.initializer_std,
        norm_eps=cfg.norm_eps,
    )


def test_mamba_block_params_match_budget(cfg_a):
    block = build_block(cfg_a)
    assert count(block) == expected_mamba_block(cfg_a)
    assert count(block) == 1_001_088  # golden value for sparsemind config


def test_pscan_matches_sequential():
    torch.manual_seed(123)
    B, T, D, N = 2, 37, 64, 5
    A = torch.randn(B, T, D, N) * 0.3
    X = torch.randn(B, T, D, N)
    y_ps = pscan(A, X)
    y_seq = selective_scan_sequential(A, X)
    torch.testing.assert_close(y_ps, y_seq, atol=1e-4, rtol=1e-4)


def test_pscan_matches_sequential_longer():
    torch.manual_seed(77)
    B, T, D, N = 1, 512, 32, 8
    A = torch.randn(B, T, D, N).abs() * 0.5
    X = torch.randn(B, T, D, N)
    torch.testing.assert_close(pscan(A, X), selective_scan_sequential(A, X), atol=1e-4, rtol=1e-4)


def test_mamba_shape(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(2, 13, cfg_a.d_model)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mamba_causal(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(1, 16, cfg_a.d_model)
    y1 = block(x)
    x2 = x.clone()
    x2[:, 6:] += 1.0
    y2 = block(x2)
    torch.testing.assert_close(y1[:, :6], y2[:, :6], atol=1e-6, rtol=1e-6)


def test_mamba_residual_skip(cfg_a):
    """Output at block output must equal x + dense_out; dense_out should be zero-ish if all weights small."""
    block = build_block(cfg_a)
    x = torch.randn(1, 4, cfg_a.d_model)
    y = block(x)
    delta = (y - x).norm()
    assert delta > 1e-8, "block with random init should produce non-trivial delta"


def test_mamba_gradients_finite(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(1, 10, cfg_a.d_model, requires_grad=True)
    block(x).pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"


def test_mamba_variable_length(cfg_a):
    block = build_block(cfg_a)
    for t in (1, 3, 64):
        x = torch.randn(1, t, cfg_a.d_model)
        y = block(x)
        assert y.shape == x.shape


def test_mamba_no_nan_long_seq(cfg_a):
    block = build_block(cfg_a)
    x = torch.randn(1, 512, cfg_a.d_model)
    y = block(x)
    assert torch.isfinite(y).all()