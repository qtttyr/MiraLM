"""SFT corpus sources: synthetic generators + shard builder.

The synthetic generators produce deterministic, schematised instructions so
the pack→train pipeline can be exercised end-to-end offline, and so the demo
has guaranteed-valid samples. In the cloud, larger real corpora (GSM8K for
CoT, Spider for SQL, instruction JSONL files) are swapped in via the same
`build_sft_shards` entry point — only the example iterator differs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from ..data.domains import get_domain_id
from ..data.packing import pack_documents
from ..data.shards import ShardWriter
from ..data.tokenizer import MiraTokenizer
from .sft_format import SFTExample, format_sft_tokens


# --------------------------------------------------------------- generators


def gen_json_examples(n: int, seed: int = 0) -> Iterator[SFTExample]:
    """Deterministic JSON-object tasks (user profile / event / product)."""
    rng = random.Random(seed)
    first = ["Alice", "Bob", "Carol", "Dan", "Emma", "Frank", "Grace", "Hugo"]
    cities = ["Kyiv", "Paris", "Berlin", "Tokyo", "Oslo", "Riga", "Lima", "Perth"]
    roles = ["admin", "editor", "viewer"]
    for i in range(n):
        kind = i % 3
        if kind == 0:
            obj = {"id": i, "name": rng.choice(first), "age": rng.randint(18, 80),
                   "city": rng.choice(cities), "role": rng.choice(roles)}
            q = "Return a JSON object describing this user: "
            q += f"id {i}, name {obj['name']}, age {obj['age']}, city {obj['city']}, role {obj['role']}."
        elif kind == 1:
            days = ["Monday", "Tuesday", "Wednesday"]
            obj = {"event": f"event_{i}", "day": rng.choice(days),
                   "starts_at": f"{rng.randint(8, 18)}:00", "capacity": rng.randint(30, 200)}
            q = ("Return a JSON object of the event with the fields event, day, "
                 f"starts_at and capacity: event {obj['event']}, {obj['day']}, "
                 f"{obj['starts_at']}, capacity {obj['capacity']}.")
        else:
            obj = {"product": f"item_{i}", "price": rng.randint(1, 99) * 100,
                   "stock": rng.randint(0, 500), "tags": rng.sample(["sale", "new", "hot", "clearance"], 2)}
            q = "Return a JSON object with product, price, stock and tags: "
            q += f"product {obj['product']}, price {obj['price']}, stock {obj['stock']}."
        yield SFTExample("json", q, json.dumps(obj, sort_keys=True))


def gen_sql_examples(n: int, seed: int = 0) -> Iterator[SFTExample]:
    """Deterministic single-table SQL questions (schema shown in the query)."""
    rng = random.Random(seed)
    tables = {
        "users": "(id INTEGER, name TEXT, age INTEGER, city TEXT)",
        "orders": "(id INTEGER, user_id INTEGER, amount REAL, status TEXT)",
        "products": "(id INTEGER, name TEXT, price REAL, stock INTEGER)",
    }
    agg = rng.choice(["COUNT", "SUM"])
    for i in range(n):
        table = list(tables)[i % len(tables)]
        schema = f"{table}{tables[table]}"
        if agg == "COUNT":
            pred = rng.choice(["> 100", "= 'pending'", "< 5"])
            answer = f"SELECT COUNT(*) FROM {table} WHERE {list(tables[table].split(','))[1].strip().split()[0]} {pred};"
            q = f"Given the schema {schema}, write an SQL query to count rows in {table}."
        else:
            col_ = rng.choice(["amount", "price", "stock", "age"])
            answer = f"SELECT SUM({col_}) FROM {table};"
            q = f"Given the schema {schema}, write an SQL query to sum {col_} in {table}."
        yield SFTExample("sql", q, answer)


def gen_cot_examples(n: int, seed: int = 0) -> Iterator[SFTExample]:
    """Deterministic arithmetic word problems solved with explicit steps."""
    rng = random.Random(seed)
    for i in range(n):
        a, b = rng.randint(3, 97), rng.randint(3, 97)
        op = rng.choice(["+", "*"])
        if op == "+":
            steps = [f"First, add the two numbers: {a} + {b}.",
                     f"Then compute the sum: {a + b}."]
            answer = str(a + b)
        else:
            steps = [f"First, multiply the two numbers: {a} * {b}.",
                     f"Then compute the product: {a * b}."]
            answer = str(a * b)
        q = f"Word problem {i}: a shop starts with {a} items and gets {b} more per day for one day. Total?"
        if op == "*":
            q = f"Word problem {i}: a rectangle is {a} by {b} units. What is its area in square units?"
        reasoning = " ".join(steps)
        yield SFTExample("cot", q, answer, reasoning)


GENERATORS: dict[str, callable] = {
    "json": gen_json_examples,
    "sql": gen_sql_examples,
    "cot": gen_cot_examples,
}


# --------------------------------------------------------------- builder


def build_sft_shards(
    tok: MiraTokenizer,
    examples: Iterable[SFTExample],
    out_dir: Path,
    seq_len: int,
    min_doc_tokens: int = 6,
    max_chunks_per_shard: int = 8192,
) -> dict[str, int]:
    """Encode + pack + persist SFT examples as pretrain-format shards.

    Returns PackingStats-ish counters. The resulting directory is directly
    consumable by src.training.loader.PackedDataLoader.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: list[tuple[int, list[int]]] = []
    n = 0
    for ex in examples:
        tokens = format_sft_tokens(ex, tok)
        domain = get_domain_id(ex.domain)   # json->JSON, sql->SQL, cot->LOGIC
        docs.append((domain, tokens))
        n += 1

    encoded = pack_documents(docs, seq_len=seq_len, min_doc_tokens=min_doc_tokens)
    writer = ShardWriter(out_dir, seq_len=seq_len, vocab_size=tok.vocab_size,
                         max_chunks_per_shard=max_chunks_per_shard)
    for chunk in encoded:
        writer.add(chunk.tokens, domain=chunk.domain)
    total_chunks = writer.total_chunks
    seq_len_actual = seq_len
    writer.close()

    return {"examples": n, "chunks": total_chunks, "tokens": total_chunks * seq_len_actual}