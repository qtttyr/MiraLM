"""Model implementations for MiraLM.

Modules:
    attention.py     GQA + RoPE + RMSNorm + SwiGLU (dense backbone)
    mamba_block.py   pure-PyTorch selective state space (pscan)
    moe.py           top-k sparse experts + router (z-loss, load balance)
    architecture.py  hybrid stack + MoE capstone + weight tying
    hf_interface.py  HuggingFace wrapper required by lm-evaluation-harness
"""

from .architecture import MiraLM, ModelOutput
from .attention import AttentionBlock, GroupedQueryAttention, RMSNorm, RotaryEmbedding, SwiGLUFFN
from .mamba_block import MambaBlock, selective_scan_sequential, pscan
from .moe import MoECapstone, Router, z_loss, load_balance_loss, guide_loss, EXPERT_DOMAINS
from .hf_interface import MiraConfig, MiraLMForCausalLM, register_mira, MODEL_TYPE

__all__ = [
    "MiraLM",
    "ModelOutput",
    "AttentionBlock",
    "GroupedQueryAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUFFN",
    "MambaBlock",
    "selective_scan_sequential",
    "pscan",
    "MoECapstone",
    "Router",
    "z_loss",
    "load_balance_loss",
    "guide_loss",
    "EXPERT_DOMAINS",
    "MiraConfig",
    "MiraLMForCausalLM",
    "register_mira",
    "MODEL_TYPE",
]