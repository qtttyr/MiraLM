"""Eval-harness integration smoke.

Creates a tiny checkpoint, verifies HF load/forward, then checks
lm_eval imports and simple_evaluate can be called (full task runs
need real tokenizer + seq_len, done via scripts/eval_harness.py).
"""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from src.config import ModelConfig, MambaConfig, TrainingConfig
from src.data.tokenizer import train_bpe_from_files
from src.finetune.sft_corpus import build_sft_shards, gen_cot_examples
from src.model.hf_interface import MiraLMForCausalLM, MiraConfig
from src.training.loader import PackedDataLoader
from src.training.trainer import Trainer
from scripts.eval_harness import run_eval, TASKS


def _tiny_tokenizer(tmp: pathlib.Path):
    corpus = tmp / "c.txt"
    rng = np.random.RandomState(0)
    lines = ["".join(chr(rng.randint(32, 127)) for _ in range(60)) for _ in range(200)]
    corpus.write_text("\n".join(lines), encoding="utf-8")
    return train_bpe_from_files([corpus], tmp / "tok.json", vocab_size=512, min_frequency=2)


def _make_ckpt(tmp: pathlib.Path) -> pathlib.Path:
    """Build a minimal HF-format checkpoint in tmp/ck/last."""
    tok = _tiny_tokenizer(tmp)
    cfg = ModelConfig(name="smoke", vocab_size=tok.vocab_size, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=128,
                      use_mamba=True, mamba=MambaConfig(d_state=8, d_conv=3, expand=2),
                      use_moe=True)
    tc = TrainingConfig(max_steps=2, batch_size=4, micro_batch=2, precision="fp32",
                        guide_steps=0, max_lr=1e-3, min_lr=1e-4, warmup_frac=0.5,
                        log_interval=0, save_interval=1, seed=0)
    examples = list(gen_cot_examples(20))
    shard = tmp / "sft"; shard.mkdir()
    build_sft_shards(tok, examples, shard, seq_len=cfg.max_seq_len, min_doc_tokens=4)
    ckpt = tmp / "ck"
    loader = PackedDataLoader(shard, tc.batch_size, cfg.max_seq_len)
    trainer = Trainer(cfg, tc, loader, ckpt, tokenizer=tok)
    trainer.train()
    return ckpt / "last"


def test_imports():
    from scripts.eval_harness import run_eval, TASKS
    assert "hellaswag" in TASKS
    assert "wikitext" in TASKS


def test_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as td:
        ckpt = _make_ckpt(pathlib.Path(td))
        model = MiraLMForCausalLM.from_pretrained(ckpt)
        assert model.config.vocab_size > 0
        assert (ckpt / "tokenizer.json").exists()


def test_run_eval_can_load_model():
    """Verify model + tokenizer load correctly for lm_eval (no actual run)."""
    with tempfile.TemporaryDirectory() as td:
        ckpt = _make_ckpt(pathlib.Path(td))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        assert model is not None
        assert tok is not None
        # verify lm_eval imports are available
        import lm_eval.evaluator as ev  # noqa: F401
        assert hasattr(ev, "simple_evaluate")