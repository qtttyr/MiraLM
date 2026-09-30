#!/usr/bin/env python3
"""MiraLM — router health report over REAL packed data.

The `load=` column the trainer prints comes from `_expert_load_str`, which feeds
`torch.randint(0, vocab, (1, 8))` — an 8-token random probe. It is far too
noisy to judge MoE health: a single random batch routinely shows several
"zero" experts that are actually alive.

This script measures routing on the real corpus. It streams chunks from a
packed shard dir, accumulates top-1 / top-k routing counts per domain, and
reports the expert x domain matrix, dead experts, load imbalance and routing
entropy. This is the number behind the submission heatmap.

Usage:
    python scripts/router_report.py --ckpt-dir checkpoints/mira/last \
        --data-dir data/packed --chunks 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.shards import ShardReader  # noqa: E402
from src.model.hf_interface import MiraLMForCausalLM  # noqa: E402


def collect(
    model,
    shard_dir: Path,
    n_chunks: int,
    device: torch.device,
    max_seq: int = 256,
) -> tuple[np.ndarray, list[str], dict]:
    """Stream chunks and accumulate routing counts. Returns (matrix, names, stats)."""
    reader = ShardReader(shard_dir)
    names = list(reader.domain_names)
    n_experts = model.model.moe.n_experts
    k = model.model.moe.n_experts_active

    counts = np.zeros((n_experts, len(names)), dtype=np.int64)   # top-1
    k_counts = np.zeros((n_experts, len(names)), dtype=np.int64)  # top-k
    prob_mass = np.zeros(n_experts, dtype=np.float64)             # mean prob per expert
    n_tokens = 0
    entropy_sum = 0.0

    step = max(1, reader.total_chunks // n_chunks)   # spread the sample over the corpus
    sampled = list(range(0, reader.total_chunks, step))
    for c in sampled:
        tokens, domain = reader.chunk(c)
        tokens = np.asarray(tokens)[:max_seq].astype(np.int64)
        ids = torch.from_numpy(tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.model(ids, return_probs=True)
        probs = out.metrics["router_probs"][0]        # (t, E)
        top1 = probs.argmax(dim=-1).cpu().numpy()
        topk = probs.topk(k, dim=-1).indices.cpu().numpy()

        d = int(domain)
        if 0 <= d < len(names):
            np.add.at(counts, (top1, np.full_like(top1, d)), 1)
            np.add.at(k_counts, (topk.reshape(-1), np.full(topk.size, d)), 1)
        n_tokens += probs.shape[0]
        prob_mass += probs.sum(dim=0).cpu().numpy()
        p = probs.mean(dim=0).double()
        entropy_sum += float(-(p * (p + 1e-12).log()).sum())

    colsum = counts.sum(axis=0, keepdims=True)
    matrix = np.divide(counts, colsum, out=np.zeros(counts.shape, dtype=float),
                       where=colsum > 0)

    top1_load = counts.sum(axis=1)
    stats = {
        "tokens_measured": n_tokens,
        "chunks_sampled": len(sampled),
        "top1_load": (top1_load / max(n_tokens, 1)).round(4).tolist(),
        "topk_load": (k_counts.sum(axis=1) / max(n_tokens * k, 1)).round(4).tolist(),
        "mean_prob": (prob_mass / max(n_tokens, 1)).round(4).tolist(),
        "mean_routing_entropy": round(entropy_sum / max(len(sampled), 1), 4),
        "max_entropy": round(float(np.log(n_experts)), 4),
        "dead_experts_top1": [i for i in range(n_experts) if top1_load[i] == 0],
    }
    return matrix, names, stats



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--chunks", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--heatmap", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MiraLMForCausalLM.from_pretrained(str(args.ckpt_dir)).to(device).eval()
    if model.model.moe is None:
        print("checkpoint has no MoE capstone — nothing to report", file=sys.stderr)
        return 1

    matrix, names, stats = collect(model, args.data_dir, args.chunks, device)

    w = 15
    print("\nMoE routing on real corpus — top-1 share per domain")
    print(" " * 10 + "".join(f"{n[:w-1]:<{w}}" for n in names))
    for e, row in enumerate(matrix):
        best = int(row.argmax()) if row.sum() > 0 else -1
        cells = [
            (f"{v:.3f}" + ("*" if d == best else " ")).ljust(w)
            for d, v in enumerate(row)
        ]
        print(f"expert {e:<2} " + "".join(cells))
    print("(* = dominant expert for that domain)")

    print("\nhealth")
    for key, val in stats.items():
        print(f"  {key:>24}: {val}")
    load = np.array(stats["top1_load"])
    gini = float(
        np.abs(np.subtract.outer(load, load)).sum()
        / (2 * len(load) ** 2 * max(load.mean(), 1e-9))
    )
    print(f"  {'load_gini':>24}: {gini:.4f}  (0 = perfectly balanced)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"checkpoint": str(args.ckpt_dir), "matrix": matrix.tolist(),
                        "domains": names, "stats": stats}, indent=2),
            encoding="utf-8",
        )
        print(f"\nreport saved: {args.output}")

    if args.heatmap:
        from src.demo.heatmap import render_heatmap
        render_heatmap(matrix, args.heatmap, names,
                       [f"E{i}" for i in range(matrix.shape[0])])
        print(f"heatmap saved: {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
