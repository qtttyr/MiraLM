"""Model implementations for MiraLM.

Modules:
    attention.py     GQA + RoPE + RMSNorm + SwiGLU (dense backbone)
    mamba_block.py   pure-PyTorch selective state space (pscan)
    moe.py           top-k sparse experts + router (z-loss, load balance)
    architecture.py  hybrid stack + MoE capstone + weight tying
    hf_interface.py  HuggingFace wrapper required by lm-evaluation-harness
"""