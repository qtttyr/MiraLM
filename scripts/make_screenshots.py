"""Generate the ≥3 submission screenshots (PNG).

Screenshots produced:
  1. expert_heatmap.png  — MoE router activation by domain (8×8 grid)
  2. loss_curve.png      — training loss + LR + MoE losses + expert load
  3. demo_outputs.png    — JSON / SQL / CoT example answers rendered as a table

Usage:
    python scripts/make_screenshots.py \\
        --ckpt-dir   checkpoints/mira/last \\
        --sft-ckpt   checkpoints/mira-sft/last \\
        --trace      checkpoints/mira/trace.csv \\
        --out-dir    results/screenshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.heatmap import expert_domain_matrix, render_heatmap, plot_trace  # noqa: E402
from src.demo.prompts import DEMO_PROMPTS                                       # noqa: E402


def _render_demo_outputs(
    ckpt_dir: Path,
    out_path: Path,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> None:
    """Run demo prompts through the SFT checkpoint and render a styled PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.demo.runner import MiraDemoModel
    demo = MiraDemoModel(ckpt_dir)

    rows: list[tuple[str, str, str]] = []
    for domain, prompts in DEMO_PROMPTS.items():
        for p in prompts:
            out = demo.generate(p, max_new_tokens=max_new_tokens, temperature=temperature)
            vis, _ = demo.split_answer(out)
            rows.append((domain, p[:80], vis.strip()[:100]))

    fig, ax = plt.subplots(figsize=(14, max(3, 0.6 * len(rows) + 2)))
    ax.axis("off")
    ax.set_title("MiraLM-47M structured-output examples", fontsize=13, pad=10)

    cell_text = [[r[0], r[1], r[2]] for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["domain", "prompt (truncated)", "answer (truncated)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for i in range(len(rows)):
        color = {"json": "#e8f5e9", "sql": "#e3f2fd", "cot": "#fff3e0"}.get(rows[i][0], "#ffffff")
        for j in range(3):
            table[i + 1, j].set_facecolor(color)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"demo outputs saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="pretrained ckpt (heatmap)")
    ap.add_argument("--sft-ckpt", type=Path, required=True, help="SFT ckpt (demo outputs)")
    ap.add_argument("--trace", type=Path, required=True, help="trainer trace.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("results/screenshots"))
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. expert heatmap
    try:
        from src.demo.runner import MiraDemoModel
        pretrain = MiraDemoModel(args.ckpt_dir)
        mat, erows, dcols = expert_domain_matrix(
            pretrain.model, pretrain.tokenizer, DEMO_PROMPTS,
            max_len=64, device=pretrain.device,
        )
        render_heatmap(mat, out / "expert_heatmap.png", dcols, erows)
    except Exception as exc:
        print(f"WARNING: heatmap failed ({exc}); skipping")

    # 2. loss curve
    plot_trace(args.trace, out / "loss_curve.png")

    # 3. demo outputs
    _render_demo_outputs(args.sft_ckpt, out / "demo_outputs.png")

    print(f"\nall screenshots saved in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())