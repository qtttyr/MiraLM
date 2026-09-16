"""SFT entrypoint: finetune a pre-trained checkpoint on structured output.

Wraps the standard training loop with guide_steps=0 so MoE load-balance
losses remain active while the guide curriculum is disabled (the router
specialises based on the SFT domain tags alone).

Usage:
    python scripts/finetune.py \
        --model-config  configs/model_sparsemind.yaml \
        --train-config  configs/train_sparsemind.yaml \
        --data-dir      data/sft \
        --ckpt-dir      checkpoints/mira-sft \
        --resume        checkpoints/mira/last
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, TrainingConfig  # noqa: E402
from src.model.hf_interface import MiraLMForCausalLM  # noqa: E402
from src.training import Trainer, PackedDataLoader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", required=True, type=Path)
    ap.add_argument("--train-config", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                    help="path to pre-trained checkpoint dir (e.g. checkpoints/mira/last)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps (e.g. 2000 for SFT)")
    args = ap.parse_args()

    if not (args.data_dir / "manifest.json").exists():
        print(f"error: no manifest.json in {args.data_dir}", file=sys.stderr)
        return 1

    cfg = ModelConfig.from_yaml(args.model_config)
    tc = TrainingConfig.from_yaml(args.train_config)
    tc.guide_steps = 0   # disable guide curriculum for SFT
    if args.max_steps is not None:
        tc.max_steps = args.max_steps

    loader = PackedDataLoader(args.data_dir, tc.batch_size, cfg.max_seq_len, shuffle=True, seed=tc.seed)
    trainer = Trainer(cfg, tc, loader, args.ckpt_dir)

    if args.resume is not None:
        print(f"loading pre-trained weights from {args.resume}")
        pretrained = MiraLMForCausalLM.from_pretrained(args.resume)
        missing, unexpected = trainer.model.load_state_dict(pretrained.model.state_dict(), strict=False)
        if missing:
            print(f"  warning: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    print(f"SFT data loaded: {loader._total_chunks} chunks")
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())