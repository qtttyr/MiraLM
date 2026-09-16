"""Demo visualisation smoke tests: heatmap, trace plot, generation."""

import csv
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E401

import numpy as np
import pytest
import torch

from src.config import ModelConfig, MambaConfig, TrainingConfig
from src.data.tokenizer import train_bpe_from_files
from src.demo.heatmap import expert_domain_matrix, render_heatmap, plot_trace
from src.demo.prompts import DEMO_PROMPTS, DEMO_INTRO
from src.demo.runner import MiraDemoModel
from src.finetune.sft_corpus import build_sft_shards, gen_cot_examples, gen_json_examples, gen_sql_examples
from src.model.architecture import MiraLM
from src.training.loader import PackedDataLoader
from src.training.trainer import Trainer


def _tiny_tokenizer(tmp: pathlib.Path):
    rng = np.random.RandomState(0)
    lines = ["".join(chr(rng.randint(32, 127)) for _ in range(60)) for _ in range(200)]
    (tmp / "c.txt").write_text("\n".join(lines), encoding="utf-8")
    return train_bpe_from_files([tmp / "c.txt"], tmp / "tok.json", vocab_size=512, min_frequency=2)


def _make_cfg_and_ckpt(tmp: pathlib.Path):
    tok = _tiny_tokenizer(tmp)
    cfg = ModelConfig(name="smoke", vocab_size=tok.vocab_size, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=64,
                      use_mamba=True, mamba=MambaConfig(d_state=8, d_conv=3, expand=2),
                      use_moe=True)
    tc = TrainingConfig(max_steps=3, batch_size=4, micro_batch=2, precision="fp32",
                        guide_steps=0, max_lr=1e-3, min_lr=1e-4, warmup_frac=0.3,
                        log_interval=0, save_interval=1, seed=0)
    examples = (list(gen_json_examples(10)) + list(gen_sql_examples(10)) + list(gen_cot_examples(10)))
    shard = tmp / "sft"; shard.mkdir()
    build_sft_shards(tok, examples, shard, seq_len=64, min_doc_tokens=4)
    ckpt = tmp / "ck"
    loader = PackedDataLoader(shard, tc.batch_size, 64)
    trainer = Trainer(cfg, tc, loader, ckpt, tokenizer=tok)
    trainer.train()
    return tok, cfg, ckpt / "last"


def test_prompts_import():
    assert "json" in DEMO_PROMPTS
    assert "sql" in DEMO_PROMPTS
    assert "cot" in DEMO_PROMPTS
    assert len(DEMO_PROMPTS["json"]) == 3


def test_expert_domain_matrix_runs():
    with tempfile.TemporaryDirectory() as td:
        tok, cfg, ckpt = _make_cfg_and_ckpt(pathlib.Path(td))
        model = MiraLM(cfg)
        texts = {"json": ["test json prompt"], "sql": ["test sql prompt"]}
        mat, erows, dcols = expert_domain_matrix(model, tok, texts, max_len=32)
        assert mat.shape == (cfg.moe.n_experts, 2)
        assert abs(mat.sum(axis=0).max() - 1.0) < 1e-6


def test_render_heatmap_saves_png():
    with tempfile.TemporaryDirectory() as td:
        mat = np.eye(8)
        out = pathlib.Path(td) / "h.png"
        render_heatmap(mat, out, [f"E{i}" for i in range(8)], [f"D{i}" for i in range(8)])
        assert out.exists() and out.stat().st_size > 0


def test_plot_trace_saves_png():
    with tempfile.TemporaryDirectory() as td:
        csv_path = pathlib.Path(td) / "trace.csv"
        rows = [{"step": str(i), "lr": f"{0.001 * (1 - i/100):.6f}",
                 "loss": f"{5.0 - i*0.1:.4f}", "z_loss": "0.5",
                 "aux_loss": "0.3", "guide_loss": "0.1",
                 "load": " ".join(f"{1/8:.4f}" for _ in range(8))} for i in range(100)]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        out = pathlib.Path(td) / "loss.png"
        plot_trace(csv_path, out)
        assert out.exists() and out.stat().st_size > 0


def test_mira_demo_model_generation():
    with tempfile.TemporaryDirectory() as td:
        tok, cfg, ckpt = _make_cfg_and_ckpt(pathlib.Path(td))
        demo = MiraDemoModel(ckpt)
        prompt = "<|json|> Return a JSON object with id=1."
        out = demo.generate(prompt, max_new_tokens=30, do_sample=False, temperature=0.0)
        vis, hid = demo.split_answer(out)
        assert len(vis) >= 0  # may be empty with tiny model but shouldn't crash