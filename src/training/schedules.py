"""Training math: LR schedules and the MoE guide-curriculum weight.

`lr_schedule` and `guide_weight` are pure functions of the step — easily
unit-tested and free of trainer state.
"""

from __future__ import annotations

import math


def cosine_warmup_lr(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_frac: float,
) -> float:
    """Warmup then cosine-decay to min_lr (LLaMA-style schedule)."""
    warmup_steps = max(1, int(max_steps * warmup_frac))
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (min_lr + (max_lr - min_lr) * cosine) if max_lr > min_lr else max_lr


def guide_weight(
    step: int,
    guide_steps: int,
    guide_ramp: int = 200,
) -> float:
    """MoE guide-curriculum annealing.

    Rises from 0 to 1 over `guide_ramp` steps, is pinned at 1, then fades to 0
    as the curriculum closes. After `guide_steps` the router is on its own
    (regular load-balance + z-loss still apply).
    """
    if guide_steps <= 0 or step >= guide_steps:
        return 0.0
    ramp = max(1, guide_ramp)
    if step < ramp:
        return 0.5 * (1.0 - math.cos(math.pi * step / ramp))          # ramp-in
    if step >= guide_steps - ramp:
        window = max(1, ramp - 1) if ramp > 1 else 1
        t = (step - (guide_steps - ramp)) / window  # 0 at start, 1 at final step
        return 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))  # 1.0 -> 0.0
    return 1.0


def linear_warmup_fraction(step: int, warmup_frac: float, max_steps: int) -> float:
    """0->1 linear warmup over the first warmup_frac steps (diagnostics)."""
    warmup_steps = max(1, int(max_steps * warmup_frac))
    return min(1.0, (step + 1) / warmup_steps)