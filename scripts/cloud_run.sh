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
#   HF_REPO        (optional) repo id to push the final SFT weights to Hub
#   KAGGLE_OUT     (optional) /kaggle/work equivalent that survives the session
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

# Persistent mirror for every fresh checkpoint. Kaggle wipes /kaggle/working /
# /kaggle/working but keeps /kaggle/output; the trainer copies each save there,
# so the last N steps survive the 12h session kill — no manual snapshot needed.
# This is pure insurance: if the target is unwritable, training just continues.
if [ -z "${MIRALM_PERSIST_DIR:-}" ] && [ -d "/kaggle/output" ]; then
    MIRALM_PERSIST_DIR="/kaggle/output/persist-${CKPT_DIR##*/}"
    export MIRALM_PERSIST_DIR
    mira_log "mirroring every checkpoint save to ${MIRALM_PERSIST_DIR} (survives session)"
fi

mira_log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# 0. gates --------------------------------------------------------------------
mira_log "parameter-budget gates"
"$PY" scripts/param_budget.py
"$PY" scripts/check_params.py

# 1. tokenizer + shards ------------------------------------------------------
if [ -f "$DATA_PACKED/manifest.json" ]; then
    mira_log "shards already present at $DATA_PACKED — skipping prepare"
else
    mira_log "tokenize + pack corpus"
    mkdir -p "$DATA_PACKED"
    "$PY" scripts/prepare_data.py \
        --corpus-dir "$DATA_RAW_DIR" \
        --out-dir "$DATA_PACKED" \
        --seq-len 1024 \
        --vocab-size 24000
fi

# 2. pre-train ---------------------------------------------------------------
RESUME_ARGS=""
if [ -d "$CKPT_DIR/last" ]; then
    RESUME_ARGS="--resume $CKPT_DIR/last"
    mira_log "pre-train checkpoint found — resuming from $CKPT_DIR/last"
fi
mira_log "pre-train (${MAX_STEPS} steps)"
"$PY" scripts/train.py \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --data-dir "$DATA_PACKED" \
    --ckpt-dir "$CKPT_DIR" \
    $RESUME_ARGS

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

mira_log "optional: push final SFT weights to HuggingFace Hub (only if HF_REPO set)"
if [ -n "${HF_REPO:-}" ]; then
    if [ -d "$CKPT_DIR-sft/last" ]; then
        "$PY" scripts/push_to_hub.py "$CKPT_DIR-sft/last" \
            --repo "$HF_REPO" \
            --token "${HF_TOKEN:-}" \
            ${HF_PRIVATE:---private} \
            --commit-message "MiraLM: SFT-${SFT_STEPS} @ ${MAX_STEPS} pretrain steps"
    else
        mira_log "no final SFT checkpoint yet — skipping hub push"
    fi
else
    mira_log "HF_REPO not set — skipping hub push (final weights stay local)"
fi

mira_log "ALL DONE — artifacts in results/, checkpoints in ${CKPT_DIR}*"

if [ -n "${KAGGLE_OUT:-}" ] && [ -d "$KAGGLE_OUT" ]; then
    mira_log "copying artifacts to persistent $KAGGLE_OUT"
    cp -r "$CKPT_DIR" "$DATA_PACKED" results "$KAGGLE_OUT/" 2>/dev/null || true
    # dated full snapshot for the jury / rollback — one per run, timestamped.
    SNAP="$(date +%Y-%m-%d_%H%M)"
    SNAP_DIR="$KAGGLE_OUT/snapshots/$SNAP"
    mkdir -p "$SNAP_DIR"
    cp -r "$CKPT_DIR" "$CKPT_DIR-sft" "$DATA_PACKED" results "$SNAP_DIR/" 2>/dev/null || true
    mira_log "dated snapshot -> $SNAP_DIR (weights + corpus + results survive the 12h kill)"
fi