"""Streaming data loader over packed memmap shards.

`PackedDataLoader` yields `(input_ids, domain)` batches for the trainer.
It reads from a ShardWriter output directory (manifest.json + shard_*/),
shuffles per-epoch and never holds the corpus in RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from ..data.shards import ShardReader


class PackedDataLoader:
    """Iterates packed chunks in shuffled order, yielding mini-batches."""

    def __init__(
        self,
        shard_dir: Path,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.reader = ShardReader(shard_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

        n = self.reader.total_chunks
        if n < 1:
            raise RuntimeError(f"no chunks found in {shard_dir}")
        self._total_chunks = n
        self._batches_per_epoch = n // batch_size
        self._orders: list[np.ndarray] = []
        self._epoch = 0
        self._pos = 0

    # ------------------------------------------------------------------ API
    def epoch_count(self) -> int:
        return self._epoch

    def batches_per_epoch(self) -> int:
        return self._batches_per_epoch

    def batches_total(self) -> int:
        return self._batches_per_epoch * (self._epoch + 1)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the next (batch_size, seq_len) ids + (batch_size,) domain tensors."""
        if not self._orders or self._pos >= self._batches_per_epoch:
            self._reshuffle()
        order = self._orders[self._epoch % len(self._orders)]
        start = self._pos * self.batch_size
        ids, doms = [], []
        for i in range(start, start + self.batch_size):
            tok, dom = self.reader.chunk(int(order[i]))
            ids.append(tok)
            doms.append(dom)
        self._pos += 1
        ids_t = torch.from_numpy(np.stack(ids, axis=0))      # (B, seq) uint16 → long
        doms_t = torch.tensor(doms, dtype=torch.long)         # (B,)
        return ids_t.to(torch.long), doms_t

    def reset_epoch(self) -> None:
        self._reshuffle()

    # --------------------------------------------------------------- private
    def _reshuffle(self) -> None:
        rng = np.random.RandomState(self.seed + self._epoch)
        order = np.arange(self._total_chunks, dtype=np.int64)
        rng.shuffle(order)
        self._orders.append(order)
        self._pos = 0
        self._epoch += 1