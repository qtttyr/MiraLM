"""Build SFT shards: `python scripts/build_sft.py --help`.

Modes:
  synthetic  – deterministic offline corpus (JSON/SQL/CoT)
  hf         – real HF dataset via `datasets` (column mapping by dataset name)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import MiraTokenizer  # noqa: E402
from src.finetune.sft_corpus import build_sft_shards, GENERATORS  # noqa: E402


def _synthetic_examples(types: str, n: int, seed: int):
    for t in types.split(","):
        t = t.strip().lower()
        if t not in GENERATORS:
            raise ValueError(f"unknown SFT type {t!r}; choose from {sorted(GENERATORS)}")
        yield from GENERATORS[t](n=n, seed=seed)


def _hf_examples(dataset: str, split: str, instruction_col: str, response_col: str):
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "the `datasets` library is required for hf mode; "
            "install it with: pip install datasets"
        )
    ds = load_dataset(dataset, split=split, streaming=True)
    from src.finetune.sft_format import SFTExample
    for row in ds:
        yield SFTExample(
            domain="cot",  # best-effort default; override per dataset if needed
            instruction=str(row.get(instruction_col, "")),
            response=str(row.get(response_col, "")),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json path")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--min-doc", type=int, default=8)
    ap.add_argument("--mode", choices=["synthetic", "hf"], default="synthetic")
    ap.add_argument("--types", default="json,sql,cot", help="comma-separated (synthetic mode)")
    ap.add_argument("--num-examples", type=int, default=300, help="per-type (synthetic mode)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hf-dataset", type=str, default="tatsu-lab/alpaca")
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--hf-instruction-col", type=str, default="instruction")
    ap.add_argument("--hf-response-col", type=str, default="output")
    args = ap.parse_args()

    tok = MiraTokenizer.load(args.tokenizer)
    if args.mode == "synthetic":
        examples = _synthetic_examples(args.types, args.num_examples, args.seed)
        n = args.num_examples * len(args.types.split(","))
    else:
        examples = _hf_examples(args.hf_dataset, args.hf_split,
                                args.hf_instruction_col, args.hf_response_col)
        n = 0  # unknown for streaming

    stats = build_sft_shards(
        tok, examples, args.out_dir,
        seq_len=args.seq_len, min_doc_tokens=args.min_doc,
    )
    print(f"built SFT shards: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())