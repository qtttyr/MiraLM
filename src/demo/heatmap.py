"""Demo visualisations: expert-activation heatmaps and training curves.

The heatmap is the "innovation" artifact for the submission: it shows that
the router's experts genuinely specialize per domain — each row is an
expert, each column a corpus domain, cell = fraction of tokens routed to
that expert.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def expert_domain_matrix(
    model,
    tok,
    texts_by_domain: dict[str, list[str]],
    max_len: int = 64,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Run the model on per-domain prompt sets, aggregate top-1 routing.

    Returns (matrix [n_experts, n_domains], expert_labels, domain_labels)
    where matrix[e, d] = fraction of tokens in domain d routed first to e.
    """
    dom_names = list(texts_by_domain)
    if model.moe is None:
        raise RuntimeError("model has no MoE capstone; heatmap unavailable")

    n_experts = model.moe.n_experts
    agg = np.zeros((n_experts, len(dom_names)))

    for di, dom in enumerate(dom_names):
        texts = texts_by_domain[dom]
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False)[:max_len]
            ti = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(ti, return_probs=True)
            probs = out.metrics["router_probs"]        # (1, t, E)
            top = probs[0].argmax(dim=-1).cpu().numpy()  # (t,)
            counts = np.bincount(top, minlength=n_experts)
            agg[:, di] += counts

    # column-normalise so each domain sums to 1.0 (routing share)
    colsum = agg.sum(axis=0, keepdims=True)
    agg = np.divide(agg, colsum, out=np.zeros_like(agg), where=colsum != 0)

    labels = [f"E{i}" for i in range(n_experts)]
    return agg, labels, dom_names


def render_heatmap(
    matrix: np.ndarray,
    out_path: Path | None = None,
    domain_labels: Optional[list[str]] = None,
    expert_labels: Optional[list[str]] = None,
    title: str = "MoE expert activation by training domain",
) -> Optional[np.ndarray]:
    """Render a matplotlib heatmap (and optionally an ASCII fallback)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — falling back to ASCII")
        return _render_ascii(matrix, domain_labels, expert_labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(domain_labels or [f"d{i}" for i in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(expert_labels or [f"E{i}" for i in range(matrix.shape[0])])
    ax.set_xlabel("domain")
    ax.set_ylabel("expert")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="routing share")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.45 else "black", fontsize=7)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return None


def _render_ascii(
    matrix: np.ndarray,
    domain_labels: Optional[list[str]] = None,
    expert_labels: Optional[list[str]] = None,
) -> np.ndarray:
    cols = domain_labels or [f"d{i}" for i in range(matrix.shape[1])]
    rows = expert_labels or [f"E{i}" for i in range(matrix.shape[0])]
    width = max(len(c) for c in cols) + 1
    lines = [" " * 5 + "".join(f"{c:<{width}}" for c in cols)]
    for i, r in enumerate(rows):
        lines.append(f"{r:<4}" + "".join(f"{matrix[i, j]:.{width - 1}f}{' ':<1}" for j in range(matrix.shape[1])))
    for ln in lines:
        print(ln)
    return matrix


def plot_trace(
    csv_path: Path,
    out_path: Path,
) -> None:
    """Render loss / lr / MoE-metrics curves from a trainer trace.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty trace: {csv_path}")
    xs = [int(r["step"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    def col(key):
        return [float(r[key]) for r in rows if r.get(key, "") != ""]

    axes[0, 0].plot(xs, col("loss"), "k-")
    axes[0, 0].set_title("total loss"); axes[0, 0].set_xlabel("step"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(xs, col("lr"), "C0-")
    axes[0, 1].set_title("learning rate"); axes[0, 1].set_xlabel("step"); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(xs, col("z_loss"), "C1-", label="z")
    axes[1, 0].plot(xs, col("aux_loss"), "C2-", label="aux")
    axes[1, 0].plot(xs, col("guide_loss"), "C3-", label="guide")
    axes[1, 0].legend(); axes[1, 0].set_title("MoE losses"); axes[1, 0].set_xlabel("step"); axes[1, 0].grid(alpha=0.3)

    if any("load" in r for r in rows):
        load_rows = [r for r in rows if r.get("load", "")]
        if load_rows:
            expert_cols = load_rows[-1]["load"].split()
            xl = [int(r["step"]) for r in load_rows]
            for k, _ in enumerate(expert_cols):
                axes[1, 1].plot(xl, [float(r["load"].split()[k]) for r in load_rows], label=f"E{k}", lw=1)
            axes[1, 1].legend(fontsize=7, ncol=4)
        axes[1, 1].set_title("expert load / step"); axes[1, 1].set_xlabel("step"); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)