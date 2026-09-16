"""Token packing: documents -> fixed-length sequences (with MoE domain labels).

Training on packed sequences instead of per-doc batches extracts ~10-30% more
tokens per GPU peek and keeps compute constant per step. Every chunk carries
the *dominant* token-domain of its constituent documents, which the trainer
feeds to the MoE guide-loss as the router's semantic seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, NamedTuple, Optional


class PackedSequence(NamedTuple):
    tokens: List[int]        # exactly seq_len ids
    domain: int              # dominant domain id of this chunk


@dataclass
class PackingStats:
    seq_len: int
    documents_seen: int = 0
    documents_dropped: int = 0   # shorter than min_doc_tokens
    chunks: int = 0                # full packed sequences produced
    tokens_packed: int = 0
    waste_tokens: int = 0          # truncated remainder at end of stream

    @property
    def utilization(self) -> float:
        total = self.tokens_packed + self.waste_tokens
        return self.tokens_packed / total if total else 0.0


def pack_documents(
    docs: Iterable[tuple[Optional[int], Iterable[int]]],
    seq_len: int,
    min_doc_tokens: int = 8,
) -> Iterator[PackedSequence]:
    """Yield fixed-length packed sequences from a stream of (domain, token).

    `docs` yields `(domain_id | None, token_ids)`. None domain falls back to
    the dominant domain of the current buffer. Chunks never mix across the
    packing boundary — the buffer is flushed at seq_len, not per domain, and
    the chunk domain is whichever source contributed the most tokens.
    """
    if seq_len < min_doc_tokens:
        raise ValueError(f"seq_len ({seq_len}) must be >= min_doc_tokens ({min_doc_tokens})")

    stats = PackingStats(seq_len=seq_len)
    buf: List[int] = []
    buf_len = 0
    counts: dict[int, int] = {}

    def flush() -> PackedSequence:
        domain = max(counts, key=counts.get)
        seq = PackedSequence(tokens=buf, domain=domain)
        return seq

    for domain, tokens in docs:
        tokens = list(tokens)
        stats.documents_seen += 1
        if len(tokens) < min_doc_tokens:
            stats.documents_dropped += 1
            continue

        i = 0
        n = len(tokens)
        while i < n:
            need = seq_len - buf_len
            take = tokens[i : i + need]
            buf.extend(take)
            buf_len += len(take)
            counts[domain] = counts.get(domain, 0) + len(take) if domain is not None else counts
            i += len(take)

            if buf_len == seq_len:
                stats.chunks += 1
                stats.tokens_packed += seq_len
                yield flush()
                buf = []
                buf_len = 0
                counts = {}

    if buf_len:
        # a full chunk wasn't reached at stream end -> count truncated waste
        stats.waste_tokens += buf_len