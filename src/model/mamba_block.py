"""Pure-PyTorch Mamba (SSM) block with 1D selective scan.

Parameter accounting mirrors scripts/param_budget.py -> mamba_block_unit():
    in_proj        d_model * 2 * d_inner
    depthwise conv d_inner * d_conv + d_inner
    x_proj         d_inner * (dt_rank + 2 * d_state)
    dt_proj        dt_rank * d_inner + d_inner
    A_log          d_state * d_inner
    D              d_inner
    out_proj       d_inner * d_model

The recurrence is evaluated with a Hillis-Steele parallel scan (pscan) in
O(log T) sequential depth; correctness is locked by an identical
sequential-reference test (tests/test_mamba_block.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RMSNorm


def pscan(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Parallel selective scan.

    For each sequence index t computes::

        y_t = sum_{i <= t} (prod_{j=i+1..t} A_j) * X_i

    A, X: (B, T, D, N) -> (B, T, D, N).
    """
    a = A.clone()
    x = X.clone()
    k = 1
    while k < x.shape[1]:
        a_new = a.clone()
        x_new = x.clone()
        a_new[:, k:] = a[:, :-k] * a[:, k:]
        x_new[:, k:] = x[:, :-k] * a[:, k:] + x[:, k:]
        a = a_new
        x = x_new
        k *= 2
    return x


def selective_scan_sequential(dA: torch.Tensor, dBx: torch.Tensor) -> torch.Tensor:
    """Sequential reference: h_t = dA_t * h_{t-1} + dBx_t. Tests only."""
    b, t, d, n = dBx.shape
    h = torch.zeros(b, d, n, dtype=dBx.dtype, device=dBx.device)
    ys: list[torch.Tensor] = []
    for i in range(t):
        h = dA[:, i] * h + dBx[:, i]
        ys.append(h)
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Selective state-space block (Mamba v1 style), scanning in fp32."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dt_rank: int, init_std: float = 0.02, norm_eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._reset_parameters(init_std)

    def _reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        nn.init.constant_(self.conv1d.bias, 0.0)
        nn.init.uniform_(self.A_log, a=-5.0, b=-1.0)
        bias_val = math.log(math.exp(1e-3) - 1)
        nn.init.constant_(self.dt_proj.bias, bias_val)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        di, st, dt = self.d_inner, self.d_state, self.dt_rank

        h = self.norm(x)
        xr, z = self.in_proj(h).chunk(2, dim=-1)     # xr, z: (B, T, di)
        skip = xr

        xr = xr.transpose(1, 2)                      # (B, di, T)
        xr = F.pad(xr, (self.d_conv - 1, 0))
        xr = self.conv1d(xr)[..., :t]                # (B, di, T)
        xr = xr.transpose(1, 2)
        xr = xr + skip                               # conv residual

        x_dbl = self.x_proj(xr)                      # (B, T, dt+2*st)
        dts, Bs, Cs = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dts))           # (B, T, di)
        Bs = Bs[:, :, None, :]                       # (B, T, 1, st)
        Cs = Cs[:, :, None, :]                       # (B, T, 1, st)

        A = -torch.exp(self.A_log)                   # (di, st)
        dtf = dt.float()
        xf = xr.float()
        dA = torch.exp(dtf[:, :, :, None] * A)       # (B, T, di, st)
        dBx = dtf[:, :, :, None] * xf[:, :, :, None] * Bs.float()
        y = pscan(dA, dBx) * Cs.float()              # (B, T, di, st)
        y = y.sum(-1)                                # (B, T, di)
        y = y + self.D * xf                          # diagonal skip
        y = y.to(x.dtype)                            # (B, T, di)

        out = self.out_proj(y * F.silu(z))
        return x + out