# Prefer the local venv; fall back to system python (e.g. Kaggle) when absent.
# Override with: make PY=python3.11 ...
ifeq ($(origin PY), undefined)
  PY := $(shell if test -x .venv/bin/python; then echo .venv/bin/python; else command -v python3 || echo python3; fi)
endif
SAMPLE ?= configs/model_sparsemind.yaml

.PHONY: setup params check test prepare-data train build-sft finetune eval eval-wiki router report demo screenshots

## bootstrap: create venv and install requirements
setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## run the parameter-budget gate (static projection)
params:
	$(PY) scripts/param_budget.py

## run the authoritative gate on the REAL model (projection must equal numel)
check:
	$(PY) scripts/check_params.py

## run the test suite
test:
	$(PY) -m pytest -q

## pre-tokenize + pack a corpus into shards (see --help)
prepare-data:
	$(PY) scripts/prepare_data.py --help

## launch pre-training (needs --model-config/--data-dir/--ckpt-dir)
train:
	$(PY) scripts/train.py --model-config $(SAMPLE) \
		--train-config configs/train_sparsemind.yaml \
		--data-dir data/packed --ckpt-dir checkpoints/mira

## build the synthetic SFT corpus
build-sft:
	$(PY) scripts/build_sft.py --tokenizer checkpoints/mira/last/tokenizer.json \
		--out-dir data/sft --seq-len 1024 --types json,sql,cot --num-examples 300

## fine-tune a pre-trained checkpoint on structured output
finetune:
	$(PY) scripts/finetune.py --model-config $(SAMPLE) \
		--train-config configs/train_sparsemind.yaml \
		--data-dir data/sft --ckpt-dir checkpoints/mira-sft \
		--resume checkpoints/mira/last --max-steps 2000

## run the four mandatory GIBC V2 multiple-choice benchmarks (lm-evaluation-harness)
eval:
	$(PY) scripts/eval_harness.py --ckpt-dir checkpoints/mira-sft/last \
		--output results/eval_mira.json --batch-size 4

## held-out WikiText-103 word-level perplexity (the fifth scored metric)
eval-wiki:
	$(PY) scripts/eval_wikitext103.py --ckpt-dir checkpoints/mira-sft/last \
		--output results/eval_wikitext103.json --lines 2000

## MoE router health on the real corpus (expert x domain routing matrix)
router:
	$(PY) scripts/router_report.py --ckpt-dir checkpoints/mira/last \
		--data-dir data/packed --chunks 512 \
		--output results/router_report.json

## render the README's measured numbers from the real artifacts (never invents)
report:
	$(PY) scripts/fill_readme.py --write

## interactive structured-output demo
demo:
	$(PY) scripts/demo.py --ckpt-dir checkpoints/mira-sft/last --all

## render submission screenshots (heatmap + curves + demo table)
screenshots:
	$(PY) scripts/make_screenshots.py --ckpt-dir checkpoints/mira/last \
		--sft-ckpt checkpoints/mira-sft/last \
		--trace checkpoints/mira/trace.csv \
		--out-dir results/screenshots