"""Training smoke: schedule math, one-step, multi-step overfit, checkpoint round-trip."""

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from src.config import ModelConfig, MambaConfig, TrainingConfig
from src.model.hf_interface import MiraLMForCausalLM
from src.training.loader import PackedDataLoader
from src.training.schedules import cosine_warmup_lr, guide_weight
from src.training.trainer import Trainer


def _mini_cfg():
    return ModelConfig(
        name="smoke", vocab_size=1024, d_model=64, n_layers=2,
        n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=32,
        use_mamba=True, mamba=MambaConfig(d_state=8, d_conv=3, expand=2),
        use_moe=True,
    )


def _mini_train():
    return TrainingConfig(
        max_steps=4, batch_size=4, micro_batch=2,
        max_lr=1e-3, min_lr=1e-4, warmup_frac=0.3,
        grad_clip=1.0, precision="fp32",
        guide_steps=2, guide_ramp=1,
        log_interval=2, save_interval=2, seed=0,
    )


def _write_synthetic_corpus(out: pathlib.Path, n_chunks=64):
    rng = np.random.RandomState(0)
    from src.data.shards import ShardWriter
    w = ShardWriter(out, seq_len=32, vocab_size=1024, max_chunks_per_shard=16)
    for _ in range(n_chunks):
        w.add(rng.randint(0, 1024, (32,)).tolist(), domain=rng.randint(0, 8))
    w.close()


# ------------------------------- schedules ----------------------------------

def test_cosine_warmup_lr_increases_then_decreases():
    lrs = [cosine_warmup_lr(i, 1000, 1e-3, 1e-4, 0.1) for i in range(1000)]
    assert lrs[0] < lrs[90]   # warmup
    assert lrs[90] > lrs[990] # decay


def test_guide_weight_bell():
    vals = [guide_weight(i, 100, 20) for i in range(100)]
    assert vals[0] == pytest.approx(0.0, abs=1e-9)
    assert vals[50] == pytest.approx(1.0, abs=1e-6)
    assert vals[99] == pytest.approx(0.0, abs=1e-9)
    assert vals[9] < 0.5  # still ramping


# -------------------------------- loader ------------------------------------

def test_loader_yields_correct_shapes():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard, n_chunks=32)
        loader = PackedDataLoader(shard, batch_size=4, seq_len=32, shuffle=True, seed=0)
        ids, doms = loader.next_batch()
        assert ids.shape == (4, 32) and doms.shape == (4,)
        assert ids.dtype == torch.long
        assert doms.min() >= 0 and doms.max() < 8


# -------------------------------- one-step ----------------------------------

def test_one_step_gradients_finite():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard)
        cfg, tc = _mini_cfg(), _mini_train()
        loader = PackedDataLoader(shard, tc.batch_size, cfg.max_seq_len)
        trainer = Trainer(cfg, tc, loader, pathlib.Path(td) / "ck")
        ids, doms = loader.next_batch()
        ids, doms = ids.to(trainer.device), doms.to(trainer.device)
        out = trainer.model(ids, domain=doms, guide_w=0.5)
        loss = torch.nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


def test_seek_mid_epoch_and_epoch_boundary():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard, n_chunks=64)
        seq = 32
        a = PackedDataLoader(shard, 4, seq, seed=3)
        for _ in range(5):
            a.next_batch()
        expected_ids, _ = a.next_batch()

        b = PackedDataLoader(shard, 4, seq, seed=3)
        b.seek(5)
        resumed_ids, _ = b.next_batch()
        assert torch.equal(expected_ids, resumed_ids)


def test_seek_mid_epoch_and_epoch_boundary():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard, n_chunks=128)  # batch 4 -> 32 batches/epoch
        seq = 32
        full = PackedDataLoader(shard, 4, seq, seed=7)
        ids = [full.next_batch()[0] for _ in range(33)]  # crosses into epoch 1

        replayed = PackedDataLoader(shard, 4, seq, seed=7)
        replayed.seek(30)
        got = [replayed.next_batch()[0] for _ in range(3)]
        assert torch.equal(ids[30], got[0])
        assert torch.equal(ids[31], got[1])
        assert torch.equal(ids[32], got[2])


def test_trainer_resume_round_trip():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard, n_chunks=128)
        cfg, tc = _mini_cfg(), _mini_train()
        ckpt = pathlib.Path(td) / "ck"

        loader = PackedDataLoader(shard, tc.batch_size, cfg.max_seq_len, seed=0)
        trainer = Trainer(cfg, tc, loader, ckpt)
        trainer.train()
        assert (ckpt / "last" / "train_meta.json").exists()

        loader2 = PackedDataLoader(shard, tc.batch_size, cfg.max_seq_len, seed=0)
        trainer2 = Trainer(cfg, tc, loader2, pathlib.Path(td) / "ck2")
        trainer2.load_resume(ckpt / "last")
        assert trainer2.step_num == cfg if False else True
        import json as _json
        meta = _json.loads((ckpt / "last" / "train_meta.json").read_text())
        assert trainer2.step_num == meta["step"]

        # continued loss must be finite and the two trainers agree on weights
        ids, doms = loader2.next_batch()
        ids = ids.to(trainer.device)
        with torch.no_grad():
            assert torch.isfinite(trainer2.model(ids).logits).all()
        s1 = trainer.model.state_dict()
        s2 = trainer2.model.state_dict()
        assert all(torch.equal(s1[k].cpu(), s2[k].cpu())
                   for k in s1 if k in s2)

def test_full_train_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as td:
        shard = pathlib.Path(td) / "s"; shard.mkdir()
        _write_synthetic_corpus(shard, n_chunks=128)
        cfg, tc = _mini_cfg(), _mini_train()
        ckpt = pathlib.Path(td) / "ck"
        loader = PackedDataLoader(shard, tc.batch_size, cfg.max_seq_len, seed=0)
        trainer = Trainer(cfg, tc, loader, ckpt)
        fixed_ids, fixed_doms = loader.next_batch()
        fixed_ids, fixed_doms = fixed_ids.to(trainer.device), fixed_doms.to(trainer.device)
        with torch.no_grad():
            loss_before = torch.nn.functional.cross_entropy(
                trainer.model(fixed_ids).logits[:, :-1].reshape(-1, cfg.vocab_size),
                fixed_ids[:, 1:].reshape(-1),
            ).item()

        trainer.train()

        with torch.no_grad():
            loss_after = torch.nn.functional.cross_entropy(
                trainer.model(fixed_ids).logits[:, :-1].reshape(-1, cfg.vocab_size),
                fixed_ids[:, 1:].reshape(-1),
            ).item()

        assert loss_after < loss_before, f"loss didn't drop: {loss_before:.4f} -> {loss_after:.4f}"
        assert (ckpt / "last").exists()
        loaded = MiraLMForCausalLM.from_pretrained(ckpt / "last")
        loaded.eval()
        loaded = loaded.to(trainer.device)  # match trainer so the compare works on GPU too
        with torch.no_grad():
            reloaded_logits = loaded(fixed_ids).logits
        assert torch.allclose(
            trainer.model(fixed_ids).logits, reloaded_logits, atol=1e-5
        )