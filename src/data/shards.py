"""Memmap shard I/O for fast, low-memory training iteration.

Layout per shard (a directory):
    tokens.bin     raw uint16 ids, shape (n_chunks, seq_len), row-major
    domains.bin    raw int8 domain ids, shape (n_chunks,)
    meta.json      {n_chunks, seq_len, vocab_size, domain_names}

A small root `manifest.json` lists every shard path. Shards are written once
at prepare time; the trainer memory-maps them and streams chunks in order /
shuffle-per-epoch without ever holding the corpus in RAM.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .domains import DOMAIN_NAMES


@dataclass
class ShardMeta:
    n_chunks: int
    seq_len: int
    vocab_size: int

    @classmethod
    def load(cls, shard_dir: Path) -> "ShardMeta":
        d = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def save(self, shard_dir: Path) -> None:
        (Path(shard_dir) / "meta.json").write_text(
            json.dumps(asdict(self)), encoding="utf-8"
        )


class ShardWriter:
    """Sequentially builds a single shard (or several capped shards)."""

    def __init__(
        self,
        out_dir: Path,
        seq_len: int,
        vocab_size: int,
        max_chunks_per_shard: int = 200_000,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_chunks_per_shard = max_chunks_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._tokens_f: Any = None
        self._domains_f: Any = None
        self._current_dir: Path | None = None
        self._chunks_in_shard = 0
        self.shards: List[Path] = []
        self.total_chunks = 0

        self._open_shard()

    def _open_shard(self) -> None:
        if self._tokens_f is not None:
            self._close_shard()
        idx = len(self.shards)
        shard_dir = self.out_dir / f"shard_{idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        self._current_dir = shard_dir
        self._tokens_f = open(shard_dir / "tokens.bin", "wb")
        self._domains_f = open(shard_dir / "domains.bin", "wb")
        self._chunks_in_shard = 0

    def add(self, tokens: List[int], domain: int) -> None:
        if self._chunks_in_shard >= self.max_chunks_per_shard:
            self._open_shard()
        self._tokens_f.write(np.asarray(tokens, dtype=np.uint16).tobytes())
        self._domains_f.write(np.asarray([domain], dtype=np.int8).tobytes())
        self._chunks_in_shard += 1
        self.total_chunks += 1

    def _close_shard(self) -> None:
        if self._tokens_f is None:
            return
        self._tokens_f.close()
        self._domains_f.close()
        ShardMeta(
            n_chunks=self._chunks_in_shard,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
        ).save(self._current_dir)
        self.shards.append(self._current_dir)
        self._tokens_f = None
        self._domains_f = None

    def close(self) -> List[Path]:
        self._close_shard()
        manifest = {
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "n_shards": len(self.shards),
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_chunks * self.seq_len,
            "domain_names": list(DOMAIN_NAMES),
            "shards": [str(s) for s in self.shards],
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return self.shards


class ShardReader:
    """Memory-maps a packed-shard directory (root manifest.json)."""

    def __init__(self, out_dir: Path) -> None:
        manifest = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.seq_len: int = manifest["seq_len"]
        self.vocab_size: int = manifest["vocab_size"]
        self.total_chunks: int = manifest["total_chunks"]
        self.domain_names: List[str] = manifest["domain_names"]
        self._shards = [self._map(Path(root) / p) for root, p in
                        [(out_dir, s) for s in manifest["shards"]]]

    @staticmethod
    def _map(shard_dir: Path):
        meta = ShardMeta.load(shard_dir)
        tokens = np.memmap(shard_dir / "tokens.bin", dtype=np.uint16, mode="r")
        tokens = tokens.reshape(meta.n_chunks, meta.seq_len)
        domains = np.memmap(shard_dir / "domains.bin", dtype=np.int8, mode="r")
        return {"tokens": tokens, "domains": domains, "n_chunks": meta.n_chunks}

    def chunk(self, global_id: int):
        """Fetch a (tokens, domain) chunk by its global index."""
        for shard in self._shards:
            n = shard["n_chunks"]
            if global_id < n:
                return shard["tokens"][global_id], int(shard["domains"][global_id])
            global_id -= n
        raise IndexError("global chunk index out of range")