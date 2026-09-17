#!/usr/bin/env bash
#
# MiraLM cloud runbook (Kaggle T4 / any Crusher+ box).
# End-to-end: tokenize+pack corpus -> pre-train -> SFT -> eval -> screenshots.
#
# Expects environment variables:
#   DATA_RAW_DIR   directory containing raw .txt/, subtask/ source folders
#   DATA_PACKED    output shards (default: data/packed)
#   CKPT_DIR       where checkpoints land (default: checkpoints/mira)
#   MAX_STEPS      override training steps (default: 20000)
#   SFT_STEPS      override SFT steps (default: 2000)
#   MODEL_CONFIGS  comma list of configs to gate+train (default: sparsemind)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
# fall back to system python when no local venv (e.g. Kaggle)
[ -x "$PY" ] || PY="$(command -v python3 || echo python3)"
cd "$ROOT"

DATA_RAW_DIR="${DATA_RAW_DIR:?set DATA_RAW_DIR to the raw corpus root}"
DATA_PACKED="${DATA_PACKED:-data/packed}"
CKPT_DIR="${CKPT_DIR:-checkpoints/mira}"
MAX_STEPS="${MAX_STEPS:-20000}"
SFT_STEPS="${SFT_STEPS:-2000}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_sparsemind.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_sparsemind.yaml}"

mira_log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# 0. gates --------------------------------------------------------------------
mira_log "parameter-budget gates"
"$PY" scripts/param_budget.py
"$PY" scripts/check_params.py

# 1. tokenizer + shards ------------------------------------------------------
mira_log "tokenize + pack corpus"
mkdir -p "$DATA_PACKED"
"$PY" scripts/prepare_data.py \
    --input-dir "$DATA_RAW_DIR" \
    --out-dir "$DATA_PACKED" \
    --tokenizer-out "$DATA_PACKED/tokenizer.json" \
    --seq-len 1024 \
    --vocab-size 24000 \
    --domain fineweb,fineweb_edu

# 2. pre-train ---------------------------------------------------------------
mira_log "pre-train (${MAX_STEPS} steps)"
"$PY" scripts/train.py \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --data-dir "$DATA_PACKED" \
    --ckpt-dir "$CKPT_DIR"

# 3. SFT (structured output) -------------------------------------------------
if [ -d "${DATA_PACKED}/sft" ]; then
    SFT_DATA="${DATA_PACKED}/sft"
else
    mira_log "building SFT corpus (JSON/SQL/CoT) + running SFT"
    "$PY" scripts/build_sft.py \
        --tokenizer "$CKPT_DIR/last/tokenizer.json" \
        --out-dir "$CKPT_DIR/sft-shards" \
        --seq-len 1024 --types json,sql,cot --num-examples 1500
    SFT_DATA="$CKPT_DIR/sft-shards"
fi
mira_log "fine-tune on structured output (${SFT_STEPS} steps)"
"$PY" scripts/finetune.py \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --data-dir "$SFT_DATA" \
    --ckpt-dir "$CKPT_DIR-sft" \
    --resume "$CKPT_DIR/last" \
    --max-steps "$SFT_STEPS"

# 4. eval + screenshots ------------------------------------------------------
mira_log "mandatory benchmarks via lm-evaluation-harness"
"$PY" scripts/eval_harness.py \
    --ckpt-dir "$CKPT_DIR-sft/last" \
    --output "results/eval_mira.json" \
    --batch-size 4

mira_log "submission screenshots (heatmap + curves + demo outputs)"
"$PY" scripts/make_screenshots.py \
    --ckpt-dir "$CKPT_DIR/last" \
    --sft-ckpt "$CKPT_DIR-sft/last" \
    --trace "$CKPT_DIR/trace.csv" \
    --out-dir results/screenshots

mira_log "ALL DONE — artifacts in results/, checkpoints in ${CKPT_DIR}*"