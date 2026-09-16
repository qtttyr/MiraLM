"""MiraLM data pipeline.

Modules:
    domains.py    domain taxonomy -> MoE expert ids (semantic router seeds)
    tokenizer.py  from-scratch ByteLevel BPE tokenizer (vocab 24k) + HF compat
    packing.py    token-packing into fixed-length sequences with domain labels
    shards.py     memmap shard I/O + manifests (fast training iteration)
"""

from .domains import DomainTagger, DOMAIN_NAMES, Domain, get_domain_id
from .tokenizer import MiraTokenizer, train_bpe_from_files
from .packing import pack_documents, PackedSequence, PackingStats
from .shards import ShardWriter, ShardReader

__all__ = [
    "DomainTagger",
    "DOMAIN_NAMES",
    "Domain",
    "get_domain_id",
    "MiraTokenizer",
    "train_bpe_from_files",
    "pack_documents",
    "PackedSequence",
    "PackingStats",
    "ShardWriter",
    "ShardReader",
]