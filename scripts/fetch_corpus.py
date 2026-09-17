#!/usr/bin/env python3
"""MiraLM — fetch the pretrain corpus from public HuggingFace sources.

Builds a plain-text `--out-dir` that scripts/prepare_data.py understands:
one or more files per source dataset, named <source>_NNN.txt where <source>
is a key recognised by src/data/domains.py (fineweb, fineweb_edu, gsm8k,
code, logicqa, hellaswag, arc_easy, piqa, winogrande, spider, json).

Keeps the corpus SMALL by default: the model only needs a few hundred MB
of text; bump --max-docs to scale (2-4B training tokens ≈ 1-2 GB of text).

Usage:
    python scripts/fetch_corpus.py --out-dir data/raw
    python scripts/fetch_corpus.py --out-dir data/raw --max-docs math 20000
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DOCS = {
    # real-DSP sources pulled from the HF hub
    "gsm8k": 8_000,        # MATH    — openai/gsm8k (main)
    "code": 6_000,         # CODE    — codeparrot/tinycodes
    "logicqa": 6_000,      # LOGIC   — google/boolq (Q/A reading-reasoning)
    "hellaswag": 6_000,    # COMMON  — rowanhellus/hellaswag
    "arc_easy": 6_000,     # COMMON  — allenai/ai2_arc (ARC-Easy)
    "piqa": 6_000,         # COMMON  — ybisk/piqa
    "winogrande": 6_000,   # COMMON  — allenai/winogrande (winogrande_xl)
    "spider": 6_000,       # SQL     — xlangai/spider
    "fineweb": 4_000,      # GEN·1   — HuggingFaceFW/fineweb (sample-10BT)
    "fineweb_edu": 4_000,  # GEN·2   — HuggingFaceFW/fineweb-edu (sample-10BT)
    "json": 6_000,         # JSON    — synthetic objects (no download)
}


def _write(rows: list[str], out_dir: Path, source: str, per_file: int = 64):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return 0
    for i in range(0, len(rows), per_file):
        chunk = rows[i : i + per_file]
        (out_dir / f"{source}_{i // per_file:03d}.txt").write_text(
            "\n\n".join(chunk), encoding="utf-8"
        )
    return len(rows)


def _mc(cur: list[str]) -> str:
    key = "text" if "text" in cur else "content"
    vals = cur.get(key)
    return vals if isinstance(vals, str) else "\n".join(map(str, vals or []))


def _http_parquet_texts(url: str, n: int, rng: random.Random) -> list[str] | None:
    """Fetch the first row-groups of one parquet shard over HTTPS.

    Single-threaded (requests + pyarrow) — avoids the `datasets` streaming
    downloader that can crash with 'PyGILState_Release' aborts on some hosts.
    """
    try:
        import pyarrow.parquet as pq
        import requests
    except Exception:
        return None
    try:
        r = requests.get(url, timeout=(30, 300))
        r.raise_for_status()
        pf = pq.ParquetFile(io.BytesIO(r.content))
        col = next((c for c in ("text", "content") if c in pf.schema.names), None)
        if col is None:
            return None
        rows: list[str] = []
        for rg in range(pf.num_row_groups):
            items = pf.read_row_group(rg, columns=[col]).to_pylist()
            rows += [t for t in items if isinstance(t, str) and t.strip()]
            if len(rows) >= n:
                break
        if rows:
            rng.shuffle(rows)
        return rows[:n]
    except Exception:
        return None


def _http_gzip_jsonl_texts(base: str, file_idxs: range, n: int, rng: random.Random) -> list[str] | None:
    """Fetch line-delimited JSON.gz shards (e.g. codeparrot-clean `content`)."""
    try:
        import gzip

        import requests
    except Exception:
        return None
    rows: list[str] = []
    for i in file_idxs:
        try:
            r = requests.get(f"{base}file-{i:012d}.json.gz", timeout=(30, 300))
            if r.status_code != 200:
                continue
            lines = gzip.decompress(r.content).decode("utf-8", "replace").splitlines()
            doc: object
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                if isinstance(doc, dict):
                    text = doc.get("content") or doc.get("text")
                elif isinstance(doc, str):
                    text = doc
                else:
                    text = None
                if isinstance(text, str) and text.strip():
                    rows.append(text)
                    if len(rows) >= n:
                        break
        except Exception:
            continue
        if len(rows) >= n:
            break
    if rows:
        rng.shuffle(rows)
    return rows[:n]


def fetch(name: str, n: int, seed: int) -> list[str] | None:
    from datasets import load_dataset

    rng = random.Random(f"{name}:{seed}")

    if name == "gsm8k":
        rows, ds = [], load_dataset("openai/gsm8k", "main", split="train")
        for ex in ds:
            rows.append(f"Q: {ex['question']}\nA: {ex['answer']}")
            if len(rows) >= n:
                break
        return rows

    if name == "code":
        rows, ds = [], load_dataset("codeparrot/tinycodes", split="train")
        for ex in ds:
            rows.append(ex.get("content") or ex.get("text") or "")
            if len(rows) >= n:
                break
        if not rows:
            # tinycodes was removed from the Hub in 2025 — fall back to the
            # raw codeparrot-clean release (line-delimited gzipped JSON).
            rows = _http_gzip_jsonl_texts(
                "https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/main/",
                range(1, 9),
                n,
                rng,
            )
        return rows

    if name == "logicqa":
        rows, ds = [], load_dataset("google/boolq", split="train")
        for ex in ds:
            ans = "yes" if ex["answer"] else "no"
            rows.append(f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {ans}")
            if len(rows) >= n:
                break
        return rows

    if name == "hellaswag":
        rows, ds = [], load_dataset("Rowan/hellaswag", split="validation")
        for ex in ds:
            ends = "\n".join(ex["endings"])
            rows.append(f"Context: {ex['ctx']}\nContinue:\n{ends}")
            if len(rows) >= n:
                break
        return rows

    if name == "arc_easy":
        rows, ds = [], load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        for ex in ds:
            pairs = "\n".join(
                f"{k}) {v}" for k, v in zip(ex["choices"]["label"], ex["choices"]["text"])
            )
            rows.append(f"Question: {ex['question']}\nOptions:\n{pairs}")
            if len(rows) >= n:
                break
        return rows

    if name == "piqa":
        rows, ds = [], load_dataset("baber/piqa", split="validation")
        for ex in ds:
            rows.append(f"Goal: {ex['goal']}\nSolution 1: {ex['sol1']}\nSolution 2: {ex['sol2']}")
            if len(rows) >= n:
                break
        return rows

    if name == "winogrande":
        rows, ds = [], load_dataset("allenai/winogrande", "winogrande_xl", split="train")
        for ex in ds:
            rows.append(
                f"Sentence: {ex['sentence']}\nOption 1: {ex['option1']}\nOption 2: {ex['option2']}"
            )
            if len(rows) >= n:
                break
        return rows

    if name == "spider":
        rows, ds = [], load_dataset("xlangai/spider", split="train")
        for ex in ds:
            rows.append(f"DB: {ex['db_id']}\nQuestion: {ex['question']}\nSQL: {ex['query']}")
            if len(rows) >= n:
                break
        return rows

    if name == "fineweb":
        rows = _http_parquet_texts(
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet",
            n,
            rng,
        )
        return rows

    if name == "fineweb_edu":
        rows = _http_parquet_texts(
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
            n,
            rng,
        )
        return rows

    return None


def make_json_rows(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    first = ["Alice", "Bob", "Eve", "Mira", "Kyiv", "Claude", "Lena", "Max"]
    verbs = ["prefers", "owns", "is", "wants", "ordered", "built", "studies", "waits"]
    out = []
    for _ in range(n):
        obj = {
            "id": rng.randint(1, 99_999),
            "name": rng.choice(first),
            "age": rng.randint(1, 90),
            "city": rng.choice(["Kyiv", "Lviv", "Odesa", "Kharkiv", "Dnipro"]),
            "active": rng.random() > 0.5,
            "tags": rng.sample(["a", "b", "c", "d"], k=rng.randint(1, 3)),
            "note": f"{rng.choice(first).lower()} {rng.choice(verbs)} {rng.choice(['tea', 'data', 'maps', 'code'])}",
        }
        out.append(json.dumps(obj, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--per-file", type=int, default=64)
    ap.add_argument(
        "--max-docs",
        nargs="+",
        help="per-source overrides, e.g. --max-docs math 20000 fineweb 1000 (repeatable keys)",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="fetch only these sources (useful to fill gaps after a partial run)",
    )
    args = ap.parse_args()

    overrides: dict[str, int] = {}
    m = args.max_docs or []
    for k in range(0, len(m), 2):
        overrides[m[k]] = int(m[k + 1])
    only = set(args.only)

    total = 0
    for name, default in DEFAULT_DOCS.items():
        if only and name not in only:
            continue
        n = overrides.get(name, default)
        try:
            if name == "json":
                rows = make_json_rows(n, args.seed)
            else:
                rows = fetch(name, n, args.seed)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"[skip] {name}: {exc}", file=sys.stderr)
            continue
        nw = _write([r for r in rows if r], args.out_dir, name, args.per_file)
        total += nw
        print(f"{name:>11}: {nw:>7,} docs -> {args.out_dir}/{name}_*.txt")

    print(f"done: {total:,} docs in {args.out_dir}")
    print("next: python scripts/prepare_data.py --corpus-dir", args.out_dir, "--out-dir data/packed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())