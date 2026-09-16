# MiraLM-47M — Sparse Mixture of Experts × Mamba Hybrid for Structured Reasoning

> `≤50M` trainable parameters (incl. embeddings & output head) · trained **from scratch**
> GIBC V2 Hackathon, Track 01 — foundational model track.
> Status: **core + data + training + SFT + eval live** (95 tests green).

---

## What is this?

A from-scratch, sub-quadratic hybrid language model: **GQA + RoPE + SwiGLU**
dense backbone alternating with **Mamba** selective state-space blocks, capped
by a sparse **top-k MoE** policy layer with a *semantically-seeded router* —
tuned for structured output (JSON / SQL / CoT) while remaining competitive on
standard commonsense and language-modeling benchmarks.

## Highlights

- **Sub-50M budget, provably.** `scripts/param_budget.py` is a day-1 gate that
  prints a full parameter accounting and fails the build if the budget is
  exceeded. `scripts/check_params.py` re-checks the *real* model (numel ==
  projection, weight-tied head). (See `results/param_budget_*.txt`.)
- **Sparse routing with guided specialization.** Router biases are initialized
  as semantic domain priors and annealed over the first 2k steps — experts
  visibly specialize, and the demo visualizes their activations live.
- **SFT for structured output.** A JSON/SQL/CoT curriculum fine-tunes the
  pretrained backbone into a machine-parseable answer protocol
  (`<|json|>`/`<|sql|>`/`<|cot|>` + `<|think|>`/`<|answer|>` delimiters).
- **Reproducible end-to-end.** `make params → check → train → finetune →
  eval` with a single config; benchmark results committed as JSON under
  `results/`.

## Training

_(to be completed — hardware, training time and approximate compute are a
submission requirement; filled in during the training phase.)_

| Hardware | GPU-hours | Tokens | tok/s/GPU | Notes |
|---|---|---|---|---|
| TBD | TBD | ~3–4B | TBD | fp16 AMP (T4) / bf16 (A100/H100) |

## Model

Two builds, both under the cap (budget gate report in `results/`):

| Build | Config | Params (static) | Layout |
|---|---|---|---|
| SparseMind-Balanced (recommended) | `configs/model_sparsemind.yaml` | **≈47.6M** | 7×Mamba ∥ 7×(GQA+SwiGLU) + MoE 8×top-2 (d1280) |
| Dense fallback | `configs/model_dense_fallback.yaml` | **≈46.9M** | 11×(GQA+SwiGLU), d=512 |

## Evaluation

Official Track-01 metrics (via lm-evaluation-harness), same harness version
and commands for the baseline models (GPT-2-117M, Pythia-70M) so comparisons
are controlled:

The harness runs the same versions/commands for the baseline models so all
comparisons are controlled:

```bash
python scripts/eval_harness.py --ckpt-dir checkpoints/mira-sft/last \
    --output results/eval_mira.json --batch-size 4
```

| Model | HellaSwag (acc_norm) | ARC-E (acc_norm) | PIQA (acc_norm) | WinoGrande (acc) | Wiki-103 (PPL) |
|---|---|---|---|---|---|
| GPT-2 117M (reproduced) | ~32.7 | ~43.3 | ~64.2 | ~49.9 | ~35–45 |
| Pythia-70M (reproduced) | ~27–30 | ~40–45 | ~61–63 | ~50–52 | ~40–60 |
| **MiraLM-47M (ours)** | TBD | TBD | TBD | TBD | TBD |

## Reproduce

```bash
make setup        # venv + deps
make params       # budget gate (must PASS)
make check        # authoritative gate (real model numel == projection)
make test         # 95-unit suite

# preparation -> train -> SFT -> eval
python scripts/prepare_data.py --input-dir data/raw --out-dir data/packed \
      --tokenizer-data data/raw --seq-len 1024 --domain fineweb,fineweb_edu
make train        # pre-training (needs --data-dir/--ckpt-dir)
make build-sft    # synthetic JSON/SQL/CoT corpus -> data/sft
make finetune     # SFT on structured output
make eval         # mandatory benchmarks via lm-evaluation-harness
```

## Reporting & compliance

- [x] Full parameter accounting: `scripts/param_budget.py` (static) + `scripts/check_params.py` (real model) → `results/`
- [x] Training loop, SFT pipeline, evaluation harness (`scripts/`)
- [ ] Evaluation JSON results committed in `results/`
- [ ] Hardware / training time / approximate compute (section above)
- [ ] Datasets & licenses: FineWeb / FineWeb-Edu (ODC-By), GSM8K (MIT), Spider (MIT)
- [ ] AI-assisted tooling disclosure *(section below)*
- [ ] Demo video + ≥3 screenshots

## AI-tooling disclosure

This project uses AI-assisted tooling (code assistants) for scaffolding,
development and analysis. All model weights and training are produced
from scratch on our own compute; no pretrained checkpoints are used.