"""Packing and shard I/O: domain tagging, chunk boundaries, memmap round-trips."""

import numpy as np
import pytest

from src.data import pack_documents, ShardWriter, ShardReader, PackedSequence
from src.data.domains import get_domain_id


def test_get_domain_id_maps_sources_and_names():
    assert get_domain_id("gsm8k") == 0
    assert get_domain_id("fineweb") == 6
    assert get_domain_id("fineweb_edu") == 7
    assert get_domain_id("spider") == 4
    assert get_domain_id("math") == 0
    assert get_domain_id("commonsense") == 3
    with pytest.raises(KeyError):
        get_domain_id("unknown_source_xyz")


def test_pack_respects_sequence_length_and_domains():
    seq_len = 8
    docs = [
        (0, list(range(20))),            # math doc, spans multiple chunks
        (3, list(range(100, 112))),      # commonsense doc (12 tokens)
    ]
    out = list(pack_documents(docs, seq_len=seq_len))
    assert all(len(s.tokens) == seq_len for s in out)
    assert len(out) == 4  # 20 + 12 = 32 tokens -> 4 full chunks
    assert out[0].tokens == list(range(8))
    assert out[1].tokens == list(range(8, 16))
    assert out[2].tokens == list(range(16, 20)) + list(range(100, 104))
    assert out[3].tokens == list(range(104, 112))
    assert out[0].domain == 0
    assert out[1].domain == 0
    assert out[2].domain == 0  # math tail (4) ties with commonsense head (4); first wins
    assert out[3].domain == 3


def test_pack_drops_short_documents():
    docs = [(4, list(range(5))), (1, list(range(100)))]
    out = list(pack_documents(docs, seq_len=16, min_doc_tokens=8))
    assert len(out) == (100 + 5 - 5) // 16  # short doc dropped outright
    assert out[0].tokens == list(range(16))


def test_pack_errors_on_tiny_sequence_len():
    with pytest.raises(ValueError):
        list(pack_documents([(0, [1, 2, 3])], seq_len=4, min_doc_tokens=8))


def test_pack_waste_tracking_at_eof():
    out = list(pack_documents([(5, list(range(13))), (6, list(range(13)))],
                              seq_len=8, min_doc_tokens=1))
    assert len(out) == 3  # 26 tokens -> 3 full chunks
    assert all(len(s.tokens) == 8 for s in out)


def test_domain_tagger_stream():
    from src.data import DomainTagger
    tagger = DomainTagger()
    stream = [("gsm8k", "text a"), ("spider", "text b")]
    tagged = list(tagger.tag_stream(iter(stream)))
    assert tagged[0] == (0, "text a")
    assert tagged[1] == (4, "text b")


def test_shard_writer_reader_roundtrip(tmp_path):
    seq_len, vocab = 8, 24000
    writer = ShardWriter(tmp_path, seq_len=seq_len, vocab_size=vocab,
                         max_chunks_per_shard=3)
    for i in range(7):
        writer.add([(i * 8 + j) % vocab for j in range(seq_len)], domain=i % 8)
    shards = writer.close()

    assert len(shards) == 3  # 7 chunks / cap 3 -> shards of 3,3,1
    meta = (shards[0] / "meta.json").read_text()
    assert '"n_chunks": 3' in meta

    reader = ShardReader(tmp_path)
    assert reader.total_chunks == 7
    assert reader.seq_len == seq_len
    tokens, domain = reader.chunk(6)
    assert domain == 6
    assert list(tokens) == [(48 + j) % vocab for j in range(seq_len)]
    with pytest.raises(IndexError):
        reader.chunk(7)


def test_shard_files_raw_layout(tmp_path):
    writer = ShardWriter(tmp_path, seq_len=4, vocab_size=24000, max_chunks_per_shard=2)
    writer.add([1, 2, 3, 4], 0)
    writer.add([5, 6, 7, 8], 1)
    writer.close()
    raw = np.memmap(tmp_path / "shard_00000" / "tokens.bin", dtype=np.uint16, mode="r")
    raw = raw.reshape(2, 4)
    np.testing.assert_array_equal(raw, [[1, 2, 3, 4], [5, 6, 7, 8]])