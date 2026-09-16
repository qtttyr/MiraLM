# MiraLM-47M — Sparse Mixture of Experts × Mamba Hybrid for Structured Reasoning

> `≤50M` trainable parameters (incl. embeddings & output head) · trained **from scratch**
> GIBC V2 Hackathon, Track 01 — foundational model track.
> Status: **work in progress** — scaffold + parameter-budget gate live.

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
  exceeded. (See `results/param_budget_*.txt`.)
- **Sparse routing with guided specialization.** Router biases are initialized
  as semantic domain priors and annealed over the first 2k steps — experts
  visibly specialize, and the demo visualizes their activations live.
- **Reproducible end-to-end.** `make params → checkout → eval` with a single
  config; benchmark results committed as JSON under `results/`.

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

```bash
lm_eval --model hf \
  --model_args pretrained=./checkpoints/best,dtype=fp16 \
  --tasks hellaswag,arc_easy,piqa,winogrande,wikitext \
  --batch_size 32 --output_path results/final.json --seed 1234
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
# …training + eval scripts added as they land
```

## Reporting & compliance

- [ ] Full parameter accounting: `scripts/param_budget.py` → `results/`
- [ ] Evaluation script + JSON results committed in `results/`
- [ ] Hardware / training time / approximate compute (section above)
- [ ] Datasets & licenses: FineWeb / FineWeb-Edu (ODC-By), GSM8K (MIT), Spider (MIT)
- [ ] AI-assisted tooling disclosure *(section below)*
- [ ] Demo video + ≥3 screenshots

## AI-tooling disclosure

This project uses AI-assisted tooling (code assistants) for scaffolding,
development and analysis. All model weights and training are produced
from scratch on our own compute; no pretrained checkpoints are used.