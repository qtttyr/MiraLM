#!/usr/bin/env python3
"""MiraLM — corpus preparation: train tokenizer, pack documents, write shards.

Runs offline on plain text inputs (one file per source dataset):
    1) train a from-scratch ByteLevel BPE (vocab 24k) on the corpus sample
    2) pack every document into fixed-length sequences, tagging each chunk
       with its dominant MoE domain
    3) write memmap shards + manifests for fast streaming during training

Domain tagging follows src/data/domains.py, which maps source filenames to
expert seeds (math / code / logic / commonsense / sql / json / general).

Usage:
    python scripts/prepare_data.py \
        --corpus-dir data/raw \
        --out-dir data/packed \
        --vocab-size 24000 \
        --seq-len 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import train_bpe_from_files, pack_documents, ShardWriter  # noqa: E402
from src.data.domains import get_domain_id  # noqa: E402

_SOURCE_KEYS = sorted(
    {
        "fineweb", "fineweb_edu", "gsm8k", "the_pile_math", "code",
        "cwsmse_commonsense", "arc_easy", "piqa", "hellaswag",
        "winogrande", "spider", "sql", "json", "logicqa", "cot",
    },
    key=len,
    reverse=True,
)


def iter_text_files(corpus_dir: Path):
    """Yield (source_name, text) for every .txt / .jsonl file under corpus_dir."""
    for path in sorted(corpus_dir.rglob("*.txt")) + sorted(corpus_dir.rglob("*.jsonl")):
        stem = path.stem
        # longest known source prefix wins — fineweb_edu -> fineweb_edu, not fineweb
        source = next((k for k in _SOURCE_KEYS if stem == k or stem.startswith(f"{k}_")), stem.split("_")[0])
        if path.suffix == ".jsonl":
            import json

            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    yield source, json.loads(line).get("text")
            continue
        yield source, path.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--vocab-size", type=int, default=24000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-chunks-per-shard", type=int, default=200_000)
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="reuse tokenizer.json already in out-dir")
    args = ap.parse_args()

    tok_path = args.out_dir / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_tokenizer and tok_path.exists():
        from src.data import MiraTokenizer
        tok = MiraTokenizer.load(tok_path)
        print(f"reused tokenizer from {tok_path} (vocab {tok.vocab_size})")
    else:
        sample = [p for p in args.corpus_dir.rglob("*.txt")][:32]
        if not sample:
            # tokenizer can train directly on the full corpus if tiny
            sample = [p for p in args.corpus_dir.rglob("*.[tj]xt")]
        if not sample:
            print("no text files found; nothing to do", file=sys.stderr)
            return 1
        print(f"training tokenizer on {len(sample)} files ...")
        tok = train_bpe_from_files(sample, vocab_size=args.vocab_size, out_path=tok_path)
        print(f"tokenizer saved: {tok_path} (vocab {tok.vocab_size})")

    writer = ShardWriter(
        args.out_dir,
        seq_len=args.seq_len,
        vocab_size=tok.vocab_size,
        max_chunks_per_shard=args.max_chunks_per_shard,
    )

    def docs():
        for source, text in iter_text_files(args.corpus_dir):
            if not text:
                continue
            try:
                domain = get_domain_id(source)
            except KeyError:
                domain = 6
            yield domain, tok.encode(text)

    chunks = 0
    for seq in pack_documents(docs(), seq_len=args.seq_len):
        writer.add(seq.tokens, seq.domain)
        chunks += 1
        if chunks % 50000 == 0:
            print(f"  packed {chunks:,} chunks ...")

    writer.close()
    print(f"done: {chunks:,} chunks over {len(writer.shards)} shard(s)")
    print(f"manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())