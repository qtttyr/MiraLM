"""HuggingFace interface: config round-trips, auto-registration, save/load,
labels/loss semantics, padding masking, generation, and tied-weight counting."""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from src import budget as B
from src.config import ModelConfig, MambaConfig
from src.model import MiraLMForCausalLM, MiraConfig, register_mira, ModelOutput
from src.model.hf_interface import IGNORE_INDEX

SPARSEMIND = "configs/model_sparsemind.yaml"
DENSE = "configs/model_dense_fallback.yaml"


def unique_param_count(model) -> int:
    """Count trainable params by unique storage (tied weights counted once)."""
    seen = {}
    for p in model.parameters():
        if p.requires_grad:
            seen[p.data_ptr()] = p.numel()
    return sum(seen.values())


def mini_config() -> ModelConfig:
    return ModelConfig(
        name="Mira-mini",
        vocab_size=1024,
        d_model=96,
        n_layers=4,
        n_heads=6,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        use_mamba=True,
        mamba=MambaConfig(d_state=8, d_conv=3, expand=2),
        use_moe=False,
    )


def mini_hf() -> MiraLMForCausalLM:
    torch.manual_seed(0)
    return MiraLMForCausalLM(MiraConfig.from_model_config(mini_config()))


# ------------------------------ config mapping ------------------------------

def test_config_roundtrip_through_model_config():
    cfg = ModelConfig.from_yaml(SPARSEMIND)
    hf = MiraConfig.from_model_config(cfg)
    round = hf.to_model_config()
    for f in ModelConfig.__dataclass_fields__:
        if f in ("mamba", "moe"):
            continue
        assert getattr(round, f) == getattr(cfg, f), f"field diverged: {f}"
    assert round.mamba.__dict__ == cfg.mamba.__dict__
    assert round.moe.__dict__ == cfg.moe.__dict__


def test_config_serializes_to_json(tmp_path):
    cfg = MiraConfig.from_model_config(ModelConfig.from_yaml(SPARSEMIND))
    cfg.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, MiraConfig)
    assert loaded.model_type == "mira"
    assert loaded.d_model == cfg.d_model
    assert loaded.n_layers == cfg.n_layers


# --------------------------- registration & param gate ----------------------

def test_auto_registration_registers_model_type():
    assert register_mira() is True


def test_auto_model_resolves_sparsemind_checkpoint(tmp_path):
    cfg = ModelConfig.from_yaml(SPARSEMIND)
    MiraLMForCausalLM(MiraConfig.from_model_config(cfg)).save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(model, MiraLMForCausalLM)


def test_sparsemind_unique_params_matches_budget_and_golden():
    torch.manual_seed(0)
    model = MiraLMForCausalLM(MiraConfig.from_model_config(ModelConfig.from_yaml(SPARSEMIND)))
    assert unique_param_count(model) == B.total_params(model.config.to_model_config())
    assert unique_param_count(model) == 47_640_968


def test_dense_unique_params_matches_budget():
    torch.manual_seed(0)
    model = MiraLMForCausalLM(MiraConfig.from_model_config(ModelConfig.from_yaml(DENSE)))
    assert unique_param_count(model) == B.total_params(model.config.to_model_config()) == 46_902_784


# ------------------------------- forward / loss -----------------------------

def test_forward_labels_loss_matches_manual_shift():
    model = mini_hf()
    b, t = 2, 16
    ids = torch.randint(0, model.config.vocab_size, (b, t))
    labels = torch.randint(0, model.config.vocab_size, (b, t))
    out = model(ids, labels=labels)
    assert out.loss is not None
    manual = F.cross_entropy(
        out.logits[:, :-1, :].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
    )
    assert torch.allclose(out.loss, manual, atol=1e-6)


def test_forward_loss_is_differentiable():
    model = mini_hf()
    ids = torch.randint(0, model.config.vocab_size, (4, 12))
    labels = torch.randint(0, model.config.vocab_size, (4, 12))
    out = model(ids, labels=labels)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None and p.requires_grad]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_attention_mask_masks_padded_labels():
    model = mini_hf()
    b, t, pad_from = 2, 16, 10
    ids = torch.randint(0, model.config.vocab_size, (b, t))
    labels = torch.randint(0, model.config.vocab_size, (b, t))
    mask = torch.ones(b, t, dtype=torch.long)
    mask[:, pad_from:] = 0

    loss_masked = model(ids, labels=labels, attention_mask=mask).loss

    # mask=0 at position i should ignore the label PREDICTED at i (position i as label target).
    target = labels.clone()
    target[:, pad_from:] = IGNORE_INDEX
    manual = model(ids, labels=target, attention_mask=None).loss
    assert torch.allclose(loss_masked, manual, atol=1e-6)


def test_forward_negative_labels_ignored():
    model = mini_hf()
    b, t = 2, 10
    ids = torch.randint(0, model.config.vocab_size, (b, t))
    labels = torch.full((b, t), IGNORE_INDEX, dtype=torch.long)  # nothing to learn
    out = model(ids, labels=labels)
    assert out.loss.item() == 0.0
    assert out.logits.shape == (b, t, model.config.vocab_size)


def test_forward_sparsemind_metrics_attached():
    cfg = ModelConfig.from_yaml(SPARSEMIND)
    torch.manual_seed(0)
    model = MiraLMForCausalLM(MiraConfig.from_model_config(cfg))
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    domain = torch.tensor([0, 3])
    out = model(ids, labels=ids, domain=domain, guide_w=0.5)
    assert out.mira_metrics is not None
    assert "guide_loss" in out.mira_metrics
    assert out.loss is not None
    assert torch.isfinite(out.loss)


# ------------------------------- generation ---------------------------------

def test_generate_greedy_shape_and_finiteness():
    model = mini_hf()
    model.eval()
    b, t, new = 1, 8, 8
    ids = torch.randint(0, model.config.vocab_size, (b, t))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, do_sample=False)
    assert out.shape == (b, t + new)
    assert torch.isfinite(out).all()
    assert (out[:, t:] >= 0).all() & (out[:, t:] < model.config.vocab_size).all()


# ---------------------------- save / load round-trip ------------------------

def test_save_load_preserves_weights_tying_and_generation(tmp_path):
    torch.manual_seed(0)
    model = mini_hf()
    model.eval()
    seed = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        before = model(seed).logits

    model.save_pretrained(tmp_path)
    loaded = MiraLMForCausalLM.from_pretrained(tmp_path)
    loaded.eval()

    with torch.no_grad():
        after = loaded(seed).logits
    assert torch.allclose(before, after, atol=1e-5)
    # tying must survive the round-trip
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight

    with torch.no_grad():
        gen_before = model.generate(seed, max_new_tokens=4, do_sample=False)
        gen_after = loaded.generate(seed, max_new_tokens=4, do_sample=False)
    assert torch.equal(gen_before, gen_after)


def test_save_load_for_dense_round_trip(tmp_path):
    cfg = ModelConfig.from_yaml(DENSE)
    torch.manual_seed(0)
    model = MiraLMForCausalLM(MiraConfig.from_model_config(cfg))
    model.eval()
    model.save_pretrained(tmp_path)
    loaded = MiraLMForCausalLM.from_pretrained(tmp_path)
    assert unique_param_count(loaded) == 46_902_784
    seed = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(seed).logits, loaded(seed).logits, atol=1e-5)


def test_input_output_embedding_accessors():
    model = mini_hf()
    emb = model.get_input_embeddings()
    assert isinstance(emb, torch.nn.Embedding)
    model.set_input_embeddings(emb)
    assert model.get_output_embeddings() is model.lm_head