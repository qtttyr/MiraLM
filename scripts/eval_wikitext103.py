#!/usr/bin/env python3
"""MiraLM — WikiText-103 held-out word-level perplexity.

The rules score "perplexity on a held-out slice of WikiText-103". That is NOT
what lm-eval's stock `wikitext` task measures: it evaluates the standard
`wikitext-2-raw-v1` test split, not a 103 slice, and not a slice this project
held out from training. This script measures what the rules ask for; the four
multiple-choice benchmarks stay in scripts/eval_harness.py so both numbers come
from one reproducible pair of commands.

    python scripts/eval_wikitext103.py --ckpt-dir checkpoints/mira-sft/last \
        --output results/eval_wikitext103.json

PPL is word-level, matching lm-eval's `word_perplexity` convention:
    ppl = exp( sum(token NLL over slice) / number of whitespace words )
A BPE model emits a different number of tokens per word than the reference
GPT-2 tokenizer, so the normaliser must be words, not model tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.hf_interface import MiraLMForCausalLM  # noqa: E402

# Fixed slice (seed 1234 over the test split) so the README number reproduces.
# prepare_data.py trains on FineWeb / task datasets and never reads this file.
WIKITEXT_103_TEST_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/test-00000-of-00001.parquet"
)


def load_heldout_lines(n_lines: int) -> list[str]:
    """Load the WikiText-103 test split and return a deterministic held-out slice."""
    import io

    import numpy as np
    import pyarrow.parquet as pq
    import requests

    print("downloading WikiText-103 test split ...", file=sys.stderr)
    r = requests.get(WIKITEXT_103_TEST_URL, timeout=300)
    r.raise_for_status()
    col = pq.read_table(io.BytesIO(r.content)).column("text").to_pylist()

    # keep prose only: blank lines and " = Heading = " carry no language signal
    # and would deflate perplexity for reasons unrelated to model quality.
    lines = [t for t in col if t and len(t.split()) >= 10]

    rng = np.random.RandomState(1234)
    idx = rng.choice(len(lines), size=min(n_lines, len(lines)), replace=False)
    idx.sort()
    return [lines[i] for i in idx]



@torch.no_grad()
def word_perplexity(
    model,
    tok,
    lines: list[str],
    device: torch.device,
    max_seq: int = 1024,
) -> dict:
    """Token NLL accumulated over the slice, normalised by whitespace-word count.

    Long lines are split into consecutive `max_seq` windows. For each window we
    score every target after its first position, so every token is scored
    exactly once (no double counting) and each gets the longest left context
    its window allows.
    """
    total_nll = 0.0
    total_tokens = 0
    total_words = 0
    windows = 0

    for line in lines:
        ids = tok(line, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        total_words += len(line.split())
        ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        for lo in range(0, ids_t.shape[1], max_seq):
            window = ids_t[:, lo : lo + max_seq]
            if window.shape[1] < 2:
                continue
            logits = model(window).logits
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                window[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_tokens += int(window.shape[1] - 1)
            windows += 1

    if total_words == 0 or total_tokens == 0:
        raise RuntimeError("empty slice: nothing was scored")

    return {
        "word_perplexity": math.exp(total_nll / total_words),
        "token_perplexity": math.exp(total_nll / total_tokens),
        "total_nll": total_nll,
        "tokens_scored": total_tokens,
        "words_scored": total_words,
        "windows_scored": windows,
    }



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--tokenizer-dir", default=None, type=Path,
                    help="defaults to --ckpt-dir")
    ap.add_argument("--lines", type=int, default=2000,
                    help="held-out slice size (lines of >=10 words)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--save-slice", type=Path, default=None,
                    help="write the held-out slice itself, for auditability")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MiraLMForCausalLM.from_pretrained(str(args.ckpt_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir or args.ckpt_dir))

    lines = load_heldout_lines(args.lines)
    print(f"held-out slice: {len(lines)} lines, "
          f"{sum(len(l.split()) for l in lines):,} words", file=sys.stderr)

    if args.save_slice:
        args.save_slice.parent.mkdir(parents=True, exist_ok=True)
        args.save_slice.write_text("\n".join(lines), encoding="utf-8")
        print(f"slice saved: {args.save_slice}", file=sys.stderr)

    res = word_perplexity(model, tok, lines, device)
    out = {
        "model": str(args.ckpt_dir),
        "dataset": "WikiText-103 (raw), test split, held-out slice",
        "slice_lines": len(lines),
        "slice_seed": 1234,
        "metric": "word_perplexity",
        **res,
    }
    print(json.dumps(out, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
