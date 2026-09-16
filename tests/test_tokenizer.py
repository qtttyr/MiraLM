"""Tokenizer: from-scratch ByteLevel BPE training, round-trips, HF compat."""

import json

import pytest
from transformers import PreTrainedTokenizerFast

from src.data import MiraTokenizer, train_bpe_from_files
from src.data.tokenizer import SPECIAL_TOKENS


def _write_corpus(tmp_path, n_docs=40):
    files = []
    for i in range(n_docs):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(
            "The quick brown fox jumps over the lazy dog. "
            "hello world the fox runs "
            f"Sentence number {i} about tokenization, subword unit and "
            "structured reasoning for large language models, json sql.\n",
            encoding="utf-8",
        )
        files.append(p)
    return files


def _train(tmp_path, vocab_size=4096):
    files = _write_corpus(tmp_path)
    return train_bpe_from_files(files, vocab_size=vocab_size, min_frequency=1,
                                out_path=tmp_path / "tok.json")


def test_train_saves_loads_roundtrip(tmp_path):
    tok = _train(tmp_path)
    loaded = MiraTokenizer.load(tmp_path / "tok.json")
    assert loaded.vocab_size == tok.vocab_size
    text = "quick brown fox"
    assert loaded.decode(loaded.encode(text)) == text

    # special token slots are fixed at the head of the vocab
    assert loaded.pad_id == 0
    assert loaded.eos_id == 1
    assert loaded.token_to_id["<|pad|>"] == 0
    assert loaded.token_to_id["<|endoftext|>"] == 1
    for i, s in enumerate(SPECIAL_TOKENS[2:], start=2):
        assert loaded.token_to_id[s] == i


def test_pad_and_eos_ids_stable():
    # id 0 = pad, id 1 = eos must be guaranteed for the demos & eval scripts
    assert SPECIAL_TOKENS[0] == "<|pad|>"
    assert SPECIAL_TOKENS[1] == "<|endoftext|>"


def test_encode_decode_batch(tmp_path):
    tok = _train(tmp_path)
    texts = ["a b c", "the fox runs"]
    batches = tok(texts)
    assert len(batches["input_ids"]) == 2
    for ids, mask in zip(batches["input_ids"], batches["attention_mask"]):
        assert len(ids) == len(mask)
        assert all(m == 1 for m in mask)


def test_special_markers_present_in_vocab(tmp_path):
    tok = _train(tmp_path)
    for marker in ("<|json|>", "<|sql|>", "<|cot|>", "<|think|>", "<|answer|>"):
        assert marker in tok.token_to_id
        # the marker is a SINGLE vocab entry, not a token sequence
        ids = tok.tok.encode(marker, add_special_tokens=False).ids
        assert ids == [tok.token_to_id[marker]]


def test_hf_tokenizer_builds(tmp_path):
    tok = _train(tmp_path)
    hf = tok.get_hf_tokenizer()
    assert isinstance(hf, PreTrainedTokenizerFast)
    assert hf.pad_token_id == 0
    assert hf.eos_token_id == 1
    ids = hf("hello world", add_special_tokens=False)["input_ids"]
    assert isinstance(ids, list) and len(ids) > 0
    assert hf.decode(ids) == "hello world"


def test_tokenizer_json_is_standard_format(tmp_path):
    out = tmp_path / "tok.json"
    train_bpe_from_files(_write_corpus(tmp_path), vocab_size=4096, min_frequency=1, out_path=out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "model" in doc and "type" in doc["model"] and doc["model"]["type"] == "BPE"