"""MoE capstone: router math, loss contract, expert specialization dynamics."""

import pytest
import torch

from src.model.attention import SwiGLUFFN
from src.model.moe import MoECapstone, Router, z_loss, load_balance_loss, guide_loss
from src.model.attention import RMSNorm
from src.budget import PARAM_LIMIT

D, E, K, ED = 128, 8, 2, 384


@pytest.fixture(scope="module")
def moe():
    torch.manual_seed(0)
    return MoECapstone(
        d_model=D, n_experts=E, n_experts_active=K,
        expert_dim=ED, router_bias=True, init_std=0.02,
    )


def param_count(moe: MoECapstone) -> int:
    return sum(p.numel() for p in moe.parameters() if p.requires_grad)


def test_moe_param_count_matches_accounting(moe):
    expected = (
        E * 3 * D * ED      # experts (SwiGLU)
        + D * E + E         # router (with bias)
    )
    assert param_count(moe) == expected


def test_moe_param_count_golden(moe):
    assert param_count(moe) == E * 3 * D * ED + D * E + E  # 1_180_680


def test_router_shape_and_probability(moe):
    x = torch.randn(4, 32, D)
    out, metrics = moe(x, return_probs=True)
    assert out.shape == x.shape
    probs = metrics["router_probs"]
    assert probs.shape == (4, 32, E)
    assert torch.allclose(probs.sum(-1), torch.ones(4, 32), atol=1e-6)


def test_each_token_routed_to_exactly_k_experts(moe):
    x = torch.randn(3, 64, D)
    logits = moe.router(x.reshape(3 * 64, D))
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(3 * 64), atol=1e-5)
    top_p, top_i = torch.topk(probs, k=K, dim=-1)
    assert top_i.shape == (3 * 64, K)
    assert top_p.shape == (3 * 64, K)
    # top-k probabilities are a strict subset of the distribution
    assert (top_p.sum(-1) < 1.0 - 1e-4).all()


def test_expert_loads_partition_probability(moe):
    x = torch.randn(2, 128, D)
    _, metrics = moe(x)
    load = metrics["expert_load"]
    assert load.shape == (E,)
    assert abs(float(load.sum()) - 1.0) < 1e-6
    # random inputs should touch every expert (no collapse)
    assert (load > 0).all()


def test_z_loss_is_nonnegative_and_small(moe):
    logits = torch.randn(8, 16, E) * 2.0
    l = z_loss(logits)
    assert l.item() > 0
    logits_small = torch.randn(8, 16, E) * 0.01
    assert z_loss(logits_small).item() < l.item()


def test_load_balance_loss_sensitive_to_collapse():
    torch.manual_seed(0)
    n, k = 512, 2
    uniform = torch.ones(n, E) / E
    idx_u = torch.randint(0, E, (n, k))
    l_uniform = load_balance_loss(uniform, idx_u)

    collapsed = torch.zeros(n, E)
    collapsed[:, 0] = 1.0
    idx_c = torch.zeros(n, k, dtype=torch.long)
    l_collapsed = load_balance_loss(collapsed, idx_c)

    assert l_collapsed > l_uniform


def test_guide_loss_peaks_on_domain_expert():
    torch.manual_seed(0)
    n, e = 64, E
    probs = torch.full((n, e), 1.0 / e)
    l_e0 = guide_loss(probs, torch.zeros(n, dtype=torch.long), guide_w=1.0)
    probs_ok = probs.clone()
    probs_ok[:, 0] = 0.9
    probs_ok[:, 1:] = 0.1 / (e - 1)
    l_matched = guide_loss(probs_ok, torch.zeros(n, dtype=torch.long), guide_w=1.0)
    assert l_matched < l_e0


def test_forward_guide_loss_present_and_annealed(moe):
    x = torch.randn(2, 16, D)
    domain = torch.tensor([0, 3])
    out, m = moe(x, domain=domain, guide_w=0.5)
    assert "guide_loss" in m
    assert m["guide_loss"].item() > 0
    # zero weight -> no guide term
    _, m0 = moe(x, domain=domain, guide_w=0.0)
    assert "guide_loss" not in m0
    # no domain -> no guide term
    _, mn = moe(x, guide_w=0.5)
    assert "guide_loss" not in mn


def test_forward_with_guide_updates_router_gradients(moe):
    x = torch.randn(2, 16, D)
    domain = torch.tensor([2, 5])
    out, m = moe(x, domain=domain, guide_w=1.0)
    w0 = moe.router.proj.weight.detach().clone()
    loss = out.norm() + m["guide_loss"]
    loss.backward()
    assert moe.router.proj.weight.grad is not None
    assert torch.isfinite(moe.router.proj.weight.grad).all()
    # guide loss pulls router toward domain expert: simulate a step
    for p in moe.parameters():
        if p.grad is not None:
            p.data -= 0.1 * p.grad
    with torch.no_grad():
        logits = moe.router(torch.randn(1, 8, D).reshape(8, D))
        probs = torch.softmax(logits, dim=-1)
    assert probs.mean(0).argmax().item() in {2, 5}


def test_forward_return_probs(moe):
    x = torch.randn(2, 8, D)
    out, m = moe(x, return_probs=True)
    assert m["router_probs"].shape == (2, 8, E)
    assert torch.allclose(m["router_probs"].sum(-1), torch.ones(2, 8), atol=1e-6)


def test_moe_identities_within_model(moe):
    x = torch.randn(1, 16, D)
    out, _ = moe(x)
    assert torch.isfinite(out).all()
    assert out.shape == x.shape


def test_moe_param_budget_headroom(moe):
    assert param_count(moe) < PARAM_LIMIT