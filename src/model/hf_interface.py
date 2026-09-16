"""HuggingFace / lm-evaluation-harness interface for MiraLM.

This module wraps the raw `MiraLM` module in a first-class `transformers`
model so the official benchmark harness (lm_eval --model hf) and the demo
beat-work with no glue code:

    MiraConfig          -> PretrainedConfig, serializable to config.json
    MiraLMForCausalLM   -> PreTrainedModel: causal LM with labels/loss,
                           generation, tied lm_head, save/load round-trips

Calling `register_mira()` (done automatically at import) registers both
classes under model_type "mira", so `AutoModelForCausalLM.from_pretrained`
resolves a locally saved MiraLM checkpoint without trust_remote_code.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .architecture import MiraLM
from ..config import ModelConfig, MambaConfig, MoEConfig

IGNORE_INDEX = -100
MODEL_TYPE = "mira"


class MiraConfig(PretrainedConfig):
    """transformers flavor of the MiraLM ModelConfig.

    MiraConfig and ModelConfig stay in 1:1 correspondence:
        MiraConfig.from_model_config(ModelConfig)   -> config.json
        MiraConfig().to_model_config()               -> ModelConfig
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        name: str = "MiraLM-47M-SparseMind",
        model_type: str = "sparsemind",
        vocab_size: int = 24000,
        d_model: int = 384,
        n_layers: int = 14,
        n_heads: int = 8,
        head_dim: Optional[int] = None,
        n_kv_heads: int = 4,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        norm_eps: float = 1e-5,
        tie_weights: bool = True,
        dropout: float = 0.0,
        initializer_std: float = 0.02,
        use_mamba: bool = True,
        mamba: Optional[dict] = None,
        use_moe: bool = True,
        moe: Optional[dict] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("tie_word_embeddings", None)  # serialized back on reload; derive from tie_weights
        super().__init__(tie_word_embeddings=tie_weights, **kwargs)

        self.name = name
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.tie_weights = tie_weights
        self.dropout = dropout
        self.initializer_std = initializer_std
        self.use_mamba = use_mamba
        self.mamba = mamba or {}
        self.use_moe = use_moe
        self.moe = moe or {}

        # generation defaults (overwritten by the tokenizer at eval time)
        self.bos_token_id = kwargs.pop("bos_token_id", 0)
        self.eos_token_id = kwargs.pop("eos_token_id", 0)
        self.pad_token_id = kwargs.pop("pad_token_id", 0)
        self.use_cache = False  # no KV-cache in the pure-PyTorch blocks

    # -------------------------------------------------------------- mapping
    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "MiraConfig":
        return cls(
            name=cfg.name,
            model_type=cfg.model_type,
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            norm_eps=cfg.norm_eps,
            tie_weights=cfg.tie_weights,
            dropout=cfg.dropout,
            initializer_std=cfg.initializer_std,
            use_mamba=cfg.use_mamba,
            mamba=asdict(cfg.mamba),
            use_moe=cfg.use_moe,
            moe=asdict(cfg.moe),
        )

    def to_model_config(self) -> ModelConfig:
        d = {
            "name": self.name,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "norm_eps": self.norm_eps,
            "tie_weights": self.tie_weights,
            "dropout": self.dropout,
            "initializer_std": self.initializer_std,
            "use_mamba": self.use_mamba,
            "mamba": self.mamba,
            "use_moe": self.use_moe,
            "moe": self.moe,
        }
        return ModelConfig.from_dict(d)


# ---------------------------------------------------------------------------
# transformers auto-registry
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_mira(force: bool = False) -> bool:
    """Register MiraLM classes in transformers' Auto{Config,Model} registries.

    After a call (or at import), `AutoModelForCausalLM.from_pretrained(dir)`
    resolves model_type "mira" without needing trust_remote_code.
    """
    global _REGISTERED
    if _REGISTERED and not force:
        return True
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register(MODEL_TYPE, MiraConfig)
    AutoModelForCausalLM.register(MiraConfig, MiraLMForCausalLM)
    _REGISTERED = True
    return True


# ---------------------------------------------------------------------------
# Causal language model wrapper
# ---------------------------------------------------------------------------

class MiraLMForCausalLM(PreTrainedModel, GenerationMixin):
    """Causal-LM wrapper around MiraLM for lm-evaluation-harness & demo.

    Every trainable parameter lives inside `self.model`; `lm_head` is the
    *same* tied module produced by MiraLM (never a second copy), so the
    unique-parameter count equals the MiraLM budget exactly. Weight tying is
    preserved across save/load via `_tied_weights_keys`.
    """

    config_class = MiraConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    _supports_sdpa = False
    _supports_cache_class = False
    _supports_static_cache = False
    supports_gradient_checkpointing = False

    def __init__(self, config: MiraConfig):
        super().__init__(config)
        self.model = MiraLM(config.to_model_config())
        self.lm_head = self.model.lm_head
        self.post_init()

    # The raw MiraLM constructor already initializes every parameter; the HF
    # init hook must not touch (re-randomize) them. Tying was done inside
    # MiraLM, not by HF, so skip the base tied-weight pass too.
    def _init_weights(self, module: nn.Module) -> None:
        pass

    # ------------------------------------------------------------------ ids
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    # -------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        domain: Optional[torch.LongTensor] = None,
        guide_w: float = 0.0,
        return_probs: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard causal-LM call.

        `labels` is the next-token-shifted target (`-100` = ignore; the
        harness and padding masks are honored through `attention_mask`).
        Extension pass-through keep the MoE guide-loss, z-loss and aux-loss
        accessible to the trainer via `output.mira_metrics`.
        """
        del position_ids, output_attentions, output_hidden_states
        del past_key_values, use_cache, kwargs

        result = self.model(
            input_ids,
            domain=domain,
            guide_w=guide_w,
            return_probs=return_probs,
        )
        logits = result.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_mask == 0, IGNORE_INDEX)
            valid = shift_labels != IGNORE_INDEX
            if not valid.any():
                # entirely-masked batch: zero loss, still differentiable
                loss = shift_logits.sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        output.mira_metrics = result.metrics
        return output

    # ------------------------------------------------------------ generate
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain: Optional[torch.LongTensor] = None,
        guide_w: float = 0.0,
        **kwargs,
    ) -> dict:
        """The blocks have no incremental cache: feed the full prefix each step."""
        del kwargs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "domain": domain,
            "guide_w": guide_w,
        }

    # ------------------------------------------------------------ utilities
    def num_unique_params(self) -> int:
        """Trainable params counting each tied weight ONCE (by storage)."""
        seen: Dict[int, int] = {}
        for p in self.parameters():
            if p.requires_grad:
                seen[p.data_ptr()] = p.numel()
        return sum(seen.values())

    @classmethod
    def from_mira_config(cls, cfg: Union[ModelConfig, MiraConfig]) -> "MiraLMForCausalLM":
        """Convenience constructor from either config flavor."""
        if isinstance(cfg, ModelConfig):
            cfg = MiraConfig.from_model_config(cfg)
        return cls(cfg)


# Auto-register: lets AutoModelForCausalLM.from_pretrained resolve model_type
# "mira" the moment this module is imported (idempotent, guarded).
register_mira()