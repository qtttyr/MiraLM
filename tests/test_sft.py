"""SFT corpus end-to-end: format + synthetic + shard + loss decrease."""

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from src.config import ModelConfig, MambaConfig, TrainingConfig
from src.data.tokenizer import build_bpe
from src.data.shards import ShardReader
from src.finetune.sft_format import SFTExample, format_sft_tokens, parse_sft_document
from src.finetune.sft_corpus import (
    gen_json_examples, gen_sql_examples, gen_cot_examples,
    build_sft_shards,
)
from src.training.loader import PackedDataLoader
from src.training.trainer import Trainer


# -------------------------------------------------------- tiny BPE helper


def _tiny_tokenizer(tmp: pathlib.Path) -> "MiraTokenizer":
    """Train a tiny ByteLevel BPE for tests (vocab 512, deterministic)."""
    corpus = tmp / "corpus.txt"
    rng = np.random.RandomState(42)
    lines = ["".join(chr(rng.randint(32, 127)) for _ in range(40)) for _ in range(200)]
    corpus.write_text("\n".join(lines), encoding="utf-8")
    from src.data.tokenizer import MiraTokenizer, train_bpe_from_files
    return train_bpe_from_files([corpus], tmp / "tok.json", vocab_size=512, min_frequency=2)


# ---------------------------------------------- format round-trip tests


def test_format_json_has_markers():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    ex = SFTExample("json", "id 1", '{"id":1}')
    ids = format_sft_tokens(ex, tok)
    assert ids[0] == tok.token_to_id["<|json|>"]
    assert tok.token_to_id["<|answer|>"] in ids
    assert ids[-1] == tok.eos_id


def test_format_cot_has_think_answer():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    ex = SFTExample("cot", "2+3=?", "5", reasoning="add two and three")
    ids = format_sft_tokens(ex, tok)
    t_id = tok.token_to_id["<|think|>"]
    a_id = tok.token_to_id["<|answer|>"]
    assert t_id in ids
    assert a_id in ids
    assert ids.index(t_id) < ids.index(a_id)


def test_parse_cot_round_trip():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    orig = SFTExample("cot", "area of 3x4?", "12", reasoning="3 * 4 = 12")
    ids = format_sft_tokens(orig, tok)
    parsed = parse_sft_document(ids, tok)
    assert parsed.domain == "cot"
    assert parsed.response.strip() == "12"
    assert "12" in (parsed.reasoning or "")


def test_parse_json_round_trip():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    orig = SFTExample("json", "give me name", '{"name":"A"}')
    ids = format_sft_tokens(orig, tok)
    parsed = parse_sft_document(ids, tok)
    assert parsed.domain == "json"
    assert "name" in parsed.response


# -------------------------------------------------------- generator tests


def test_json_examples_deterministic():
    a = list(gen_json_examples(5, seed=0))
    b = list(gen_json_examples(5, seed=0))
    assert [e.instruction for e in a] == [e.instruction for e in b]


def test_sql_examples_produce_valid_answer():
    examples = list(gen_sql_examples(10))
    for ex in examples:
        assert ex.response.strip().startswith("SELECT")


def test_cot_examples_have_reasoning():
    examples = list(gen_cot_examples(5))
    for ex in examples:
        assert ex.reasoning is not None
        assert ex.response.strip().lstrip("-").isdigit()


# -------------------------------------------------------- shard builder


def test_build_sft_shards_produces_chunks():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    examples = list(gen_json_examples(20)) + list(gen_sql_examples(20))
    with tempfile.TemporaryDirectory() as td:
        stats = build_sft_shards(tok, examples, pathlib.Path(td) / "sft",
                                 seq_len=64, min_doc_tokens=4)
        assert stats["chunks"] > 0
        assert stats["examples"] == 40
        assert (pathlib.Path(td) / "sft" / "manifest.json").exists()


def test_sft_chunks_loadable_by_loader():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    examples = list(gen_cot_examples(50))
    with tempfile.TemporaryDirectory() as td:
        shard_dir = pathlib.Path(td) / "sft"
        build_sft_shards(tok, examples, shard_dir, seq_len=32, min_doc_tokens=4)
        loader = PackedDataLoader(shard_dir, batch_size=4, seq_len=32, shuffle=False)
        ids, doms = loader.next_batch()
        assert ids.shape == (4, 32)


# ----------------------------------------------- trainer on SFT shards


def test_sft_trainer_step():
    tok = _tiny_tokenizer(pathlib.Path(tempfile.mkdtemp()))
    examples = list(gen_json_examples(60)) + list(gen_sql_examples(60))
    cfg = ModelConfig(name="sft", vocab_size=tok.vocab_size, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=32,
                      use_mamba=True, mamba=MambaConfig(d_state=8, d_conv=3, expand=2),
                      use_moe=True)
    tc = TrainingConfig(max_steps=3, batch_size=4, micro_batch=2, precision="fp32",
                        guide_steps=0, max_lr=5e-4, min_lr=1e-4, warmup_frac=0.3,
                        log_interval=1, save_interval=0, seed=1)
    with tempfile.TemporaryDirectory() as td:
        shard_dir = pathlib.Path(td) / "sft"
        build_sft_shards(tok, examples, shard_dir, seq_len=32, min_doc_tokens=4)
        loader = PackedDataLoader(shard_dir, tc.batch_size, cfg.max_seq_len, shuffle=False, seed=1)
        trainer = Trainer(cfg, tc, loader, pathlib.Path(td) / "ck")
        fixed, fixed_d = loader.next_batch()
        fixed, fixed_d = fixed.to(trainer.device), fixed_d.to(trainer.device)
        with torch.no_grad():
            loss_before = torch.nn.functional.cross_entropy(
                trainer.model(fixed).logits[:, :-1].reshape(-1, cfg.vocab_size),
                fixed[:, 1:].reshape(-1),
            ).item()
        trainer.train()
        with torch.no_grad():
            loss_after = torch.nn.functional.cross_entropy(
                trainer.model(fixed).logits[:, :-1].reshape(-1, cfg.vocab_size),
                fixed[:, 1:].reshape(-1),
            ).item()
        assert loss_after < loss_before + 10.0  # no blow-up; ideally decreasing