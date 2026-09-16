"""End-to-end model tests: real param count == static projection, layer order,
forward shapes, MoE capstone integration, and weight tying."""

import pytest
import torch

from src.config import ModelConfig
from src import budget as B
from src.model.architecture import MiraLM, ModelOutput
from src.model.attention import AttentionBlock
from src.model.mamba_block import MambaBlock
from src.model.moe import MoECapstone


def real_params(model: MiraLM) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- parameter gate: static projection must equal real model ----------

@pytest.mark.parametrize("cfg_path", [
    "configs/model_sparsemind.yaml",
    "configs/model_dense_fallback.yaml",
])
def test_real_params_match_projection_and_limit(cfg_path):
    cfg = ModelConfig.from_yaml(cfg_path)
    model = MiraLM(cfg)
    real = real_params(model)
    assert real == B.total_params(cfg), "static budget diverges from real model!"
    assert real <= B.PARAM_LIMIT


def test_golden_sparsemind_param_count():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    assert real_params(model) == 47_640_968


def test_golden_dense_param_count():
    cfg = ModelConfig.from_yaml("configs/model_dense_fallback.yaml")
    model = MiraLM(cfg)
    assert real_params(model) == 46_902_784


# ------------------------- architecture structure ---------------------------

def test_weight_tying_tied_and_shared():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    assert model.lm_head.weight is model.embed_tokens.weight
    # a tied embedding must not double-count parameters
    assert len(
        {id(p) for p in model.parameters() if p.requires_grad}
    ) == sum(1 for p in model.parameters() if p.requires_grad)


def test_layer_roles_alternate():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    roles = [cfg.layer_role(i) for i in range(cfg.n_layers)]
    assert roles.count("mamba") == cfg.n_mamba_layers
    assert roles.count("attention") == cfg.n_attention_layers
    types = [type(b).__name__ for b in model.blocks]
    for i, role in enumerate(roles):
        expected = MambaBlock.__name__ if role == "mamba" else AttentionBlock.__name__
        assert types[i] == expected


def test_moe_capstone_present_only_when_configured():
    cfg_s = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    cfg_d = ModelConfig.from_yaml("configs/model_dense_fallback.yaml")
    assert isinstance(MiraLM(cfg_s).moe, MoECapstone)
    assert MiraLM(cfg_d).moe is None


# ------------------------------ forward passes ------------------------------

@pytest.mark.parametrize("cfg_path", [
    "configs/model_sparsemind.yaml",
    "configs/model_dense_fallback.yaml",
])
def test_forward_shapes(cfg_path):
    cfg = ModelConfig.from_yaml(cfg_path)
    model = MiraLM(cfg)
    model.eval()
    b, t = 2, 24
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    with torch.no_grad():
        out: ModelOutput = model(ids)
    assert out.logits.shape == (b, t, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_forward_with_domain_and_guide_sparsemind():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    b, t = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    domain = torch.tensor([0, 3])
    out = model(ids, domain=domain, guide_w=0.5, return_probs=True)
    assert out.metrics is not None
    for key in ("z_loss", "aux_loss", "guide_loss", "expert_load"):
        assert key in out.metrics
    assert out.metrics["guide_loss"].item() > 0
    assert out.metrics["router_probs"].shape == (b, t, cfg.moe.n_experts)


def test_forward_gradients_flow_sparsemind():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    domain = torch.tensor([1, 6])
    out = model(ids, domain=domain, guide_w=0.1)
    loss = out.logits.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_forward_no_moe_metrics_dense():
    cfg = ModelConfig.from_yaml("configs/model_dense_fallback.yaml")
    model = MiraLM(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (2, 12)))
    assert out.metrics is None


def test_final_norm_and_logits_stable_sparsemind():
    cfg = ModelConfig.from_yaml("configs/model_sparsemind.yaml")
    model = MiraLM(cfg)
    model.eval()
    with torch.no_grad():
        # long-ish sequence through the full stack stays finite
        out = model(torch.randint(0, cfg.vocab_size, (1, 96)))
    assert torch.isfinite(out.logits).all()
    assert out.logits.abs().max().item() < 1e3