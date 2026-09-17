"""Training entrypoint for MiraLM.

Loads a pre-packed shard directory (built by scripts/prepare_data.py),
a ModelConfig and a TrainingConfig, then runs the training loop with
AMP, cosine LR schedule, MoE guide-curriculum annealing, and periodic
HF-format checkpointing.

Usage:
    python scripts/train.py \
        --model-config configs/model_sparsemind.yaml \
        --train-config configs/train_sparsemind.yaml \
        --data-dir data/packed \
        --ckpt-dir checkpoints/mira
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, TrainingConfig  # noqa: E402
from src.training import Trainer, PackedDataLoader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", required=True, type=Path)
    ap.add_argument("--train-config", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="packed shard directory (manifest.json)")
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue from a checkpoint dir (e.g. checkpoints/mira/last)")
    args = ap.parse_args()

    if not (args.data_dir / "manifest.json").exists():
        print(f"error: no manifest.json found in {args.data_dir}", file=sys.stderr)
        return 1

    cfg = ModelConfig.from_yaml(args.model_config)
    tc = TrainingConfig.from_yaml(args.train_config)

    loader = PackedDataLoader(
        args.data_dir,
        batch_size=tc.batch_size,
        seq_len=cfg.max_seq_len,
        shuffle=True,
        seed=tc.seed,
    )
    print(f"data loaded: {loader._total_chunks} chunks, {loader.batches_per_epoch()} batches/epoch")

    trainer = Trainer(cfg, tc, loader, args.ckpt_dir)
    if args.resume is not None:
        trainer.load_resume(args.resume)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())