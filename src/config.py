"""Model configuration for MiraLM.

Every knob that affects the parameter budget lives here and is single-source
of truth for: the YAML configs, the static budget projection
(scripts/param_budget.py) and the real-model gate (scripts/check_params.py).
All three MUST agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class MambaConfig:
    """Selective state-space block (Mamba v1 style, pure PyTorch)."""

    d_state: int = 16          # SSM state dimension
    d_conv: int = 4            # depthwise conv kernel
    expand: int = 2            # d_inner = d_model * expand

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MambaConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MoEConfig:
    """Sparse top-k Mixture-of-Experts capstone."""

    n_experts: int = 8
    n_experts_active: int = 2
    expert_dim: int = 1280     # SWiGLU hidden width per expert
    router_bias: bool = True   # semantic seeding: per-expert log prior
    z_loss_coef: float = 1e-3  # stabilizes large router logits
    aux_loss_coef: float = 1e-2  # load-balancing (Switch-style)
    guide_steps: int = 2000    # annealed curriculum: route toward domain expert

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _coerce(field_type, value):
    """Cast YAML values to their dataclass-annotated type.

    PyYAML leaves numbers like `1e-3` (no decimal point) as str; coerce here so
    configs written either way always parse into numbers.
    """
    if field_type in (float, int) and not isinstance(value, field_type):
        try:
            return field_type(value)
        except (TypeError, ValueError):
            return value
    return value


def _coerce_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    hints = {k: t for k, t in __import__("typing").get_type_hints(cls).items() if k in d}
    return {k: _coerce(hints[k], v) if k in hints else v for k, v in d.items()}


@dataclass
class TrainingConfig:
    """Training hyper-parameters (does NOT affect the parameter budget)."""

    max_steps: int = 20000
    batch_size: int = 8          # sequences per optimizer step
    micro_batch: int = 4         # sequences per forward/backward (grad accum)
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_frac: float = 0.01
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "fp16"      # fp16 (T4) | bf16 (A100/H100) | fp32
    guide_steps: int = 2000      # MoE guide-curriculum horizon
    guide_ramp: int = 200        # cosine ramp-in/out for the guide weight
    log_interval: int = 20
    save_interval: int = 1000
    compile: bool = False
    seed: int = 1234

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**_coerce_dict(cls, d))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ModelConfig:
    name: str = "MiraLM-47M-SparseMind"
    model_type: str = "sparsemind"     # "sparsemind" | "dense"

    # Core
    vocab_size: int = 24000
    d_model: int = 384
    n_layers: int = 14
    n_heads: int = 8
    head_dim: Optional[int] = None     # None -> d_model // n_heads
    n_kv_heads: int = 4                # grouped-query attention
    d_ff: int = 2048                   # SwiGLU expansion (attention layers)
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    tie_weights: bool = True           # lm_head = embedding (required!)
    dropout: float = 0.0
    initializer_std: float = 0.02

    # Optional hybrid blocks
    use_mamba: bool = True             # alternating Mamba / attention if True
    mamba: MambaConfig = field(default_factory=MambaConfig)
    use_moe: bool = True
    moe: MoEConfig = field(default_factory=MoEConfig)

    # ------------------------------------------------------------------
    # Derived layout
    # ------------------------------------------------------------------
    @property
    def head_dim_resolved(self) -> int:
        return self.head_dim or self.d_model // self.n_heads

    @property
    def mamba_d_inner(self) -> int:
        return self.d_model * self.mamba.expand

    @property
    def mamba_dt_rank(self) -> int:
        return math.ceil(self.mamba_d_inner / 16)

    def layer_role(self, i: int) -> str:
        """Jamba-style alternation: even indices Mamba (if enabled)."""
        if self.use_mamba and i % 2 == 0:
            return "mamba"
        return "attention"

    @property
    def n_attention_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.layer_role(i) == "attention")

    @property
    def n_mamba_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.layer_role(i) == "mamba")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "ModelConfig":
        assert 1 <= self.n_layers <= 64, "n_layers out of range"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.vocab_size >= 128, "vocab_size too small"
        assert self.tie_weights, "tie_weights must be on: it saves ~half the budget"
        assert self.moe.n_experts_active < self.moe.n_experts, "top-k must be < n_experts"
        assert self.moe.expert_dim % 8 == 0, "expert_dim should be multiple of 8"
        return self

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = set(cls.__dataclass_fields__)
        mamba = MambaConfig.from_dict(d.get("mamba", {}))
        moe = MoEConfig.from_dict(d.get("moe", {}))
        kwargs = {k: v for k, v in d.items() if k in known and k not in ("mamba", "moe")}
        return cls(**kwargs, mamba=mamba, moe=moe).validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(_coerce_dict(cls, yaml.safe_load(f)))

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "mamba" and k != "moe"}
        d["mamba"] = self.mamba.__dict__.copy()
        d["moe"] = self.moe.__dict__.copy()
        return d

    def save_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)