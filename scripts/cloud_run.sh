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

mira_log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

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

# Persistent mirror for every fresh checkpoint. Kaggle wipes /kaggle/working on
# every session restart (including "Save & Run All") but keeps /kaggle/output.
# The trainer copies each save into MIRALM_PERSIST_DIR; the restore block below
# copies them BACK so a re-run resumes instead of retraining from step 0.
if [ -z "${MIRALM_PERSIST_DIR:-}" ] && [ -d "/kaggle/output" ]; then
    MIRALM_PERSIST_DIR="/kaggle/output/persist-${CKPT_DIR##*/}"
    export MIRALM_PERSIST_DIR
    mira_log "mirroring every checkpoint save to ${MIRALM_PERSIST_DIR} (survives session)"
fi

# Directory-level mirror for everything a fresh session would otherwise lose:
# the packed corpus, the SFT weights, results and the loss trace.
PERSIST_ROOT="${PERSIST_ROOT:-/kaggle/output/miralm-persist}"
if [ -d "/kaggle/output" ] && [ -z "${MIRALM_PERSIST_DIR:-}" ]; then
    PERSIST_ROOT="/kaggle/output/miralm-persist"
fi

# _mirror_tree <src> <dst-in-persist-root>   best effort, never fatal
_mirror_tree() {
    [ -e "$1" ] || return 0
    mkdir -p "$2" 2>/dev/null || return 0
    cp -r "$1/." "$2/" 2>/dev/null || true
}

# _restore_tree <src-in-persist-root> <dst>   only if dst is missing
_restore_tree() {
    [ -e "$1" ] || return 0
    if [ -e "$2" ]; then
        mira_log "restore: ${2} already exists — keeping the live copy"
        return 0
    fi
    mkdir -p "$(dirname "$2")" 2>/dev/null || return 0
    cp -r "$1" "$2" 2>/dev/null || true
    [ -e "$2" ] && mira_log "restored ${2} from ${PERSIST_ROOT}"
}

# ---------------------------------------------------------------------------
# 0a. RESTORE — must run before anything else touches the filesystem.
#     After a "Save & Run All" commit, /kaggle/working is empty. Without this
#     the run would silently restart from step 0 and re-pack the whole corpus.
# ---------------------------------------------------------------------------
if [ -d "$PERSIST_ROOT" ]; then
    mira_log "found persistent store at ${PERSIST_ROOT} — restoring"
    _restore_tree "$PERSIST_ROOT/checkpoints"    "$ROOT/checkpoints"
    _restore_tree "$PERSIST_ROOT/data/packed"    "$ROOT/$DATA_PACKED"
    _restore_tree "$PERSIST_ROOT/results"        "$ROOT/results"
    # the trainer's per-save mirror is the authoritative copy of the newest
    # weights; prefer it when the working-dir checkpoint is absent
    if [ -n "${MIRALM_PERSIST_DIR:-}" ] && [ -d "$MIRALM_PERSIST_DIR" ]; then
        _restore_tree "$MIRALM_PERSIST_DIR" "$ROOT/${CKPT_DIR}/last"
    fi
else
    mira_log "no persistent store yet at ${PERSIST_ROOT} — starting fresh"
fi

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

# The packed corpus lives in /kaggle/working, which a "Save & Run All" commit
# wipes. Re-packing it means re-tokenising the whole corpus (hours), so mirror it
# into the persistent store the moment it exists.
_mirror_tree "$DATA_PACKED" "$PERSIST_ROOT/$DATA_PACKED"
mira_log "corpus shards mirrored to $PERSIST_ROOT/$DATA_PACKED"

# 2. pre-train ---------------------------------------------------------------
# Require train_meta.json, not just the directory: a checkpoint dir that exists
# but is empty (e.g. a half-finished restore) would make --resume crash on a
# missing metadata file instead of training from scratch.
RESUME_ARGS=""
if [ -f "$CKPT_DIR/last/train_meta.json" ]; then
    RESUME_ARGS="--resume $CKPT_DIR/last"
    mira_log "pre-train checkpoint found — resuming from $CKPT_DIR/last"
elif [ -d "$CKPT_DIR/last" ]; then
    mira_log "WARNING: $CKPT_DIR/last exists but has no train_meta.json — ignoring it and training from scratch"
else
    mira_log "no pre-train checkpoint — training from scratch"
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
mira_log "mandatory multiple-choice benchmarks via lm-evaluation-harness"
"$PY" scripts/eval_harness.py \
    --ckpt-dir "$CKPT_DIR-sft/last" \
    --output "results/eval_mira.json" \
    --batch-size 4

# The rules score "perplexity on a held-out slice of WikiText-103" — lm-eval's
# stock `wikitext` task is wikitext-2 and is not a held-out slice, so the fifth
# metric is measured here instead.
mira_log "held-out WikiText-103 word-level perplexity"
"$PY" scripts/eval_wikitext103.py \
    --ckpt-dir "$CKPT_DIR-sft/last" \
    --output "results/eval_wikitext103.json" \
    --lines 2000 \
    --save-slice "results/wikitext103_heldout_slice.txt"

# Router health on the real corpus — the number behind the expert heatmap, and
# the evidence that the semantic seeding did something.
mira_log "MoE router report on real corpus"
"$PY" scripts/router_report.py \
    --ckpt-dir "$CKPT_DIR/last" \
    --data-dir "$DATA_PACKED" \
    --chunks 512 \
    --output "results/router_report.json" \
    --heatmap "results/screenshots/expert_heatmap.png"

mira_log "recording real demo generations (the site replays these verbatim)"
"$PY" scripts/record_demo.py \
    --ckpt-dir "$CKPT_DIR-sft/last" \
    --output "results/demo_outputs.json"

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

# ---------------------------------------------------------------------------
# 5. final mirror — everything the next session needs to resume or resubmit.
# ---------------------------------------------------------------------------
mira_log "mirroring artifacts into the persistent store: $PERSIST_ROOT"
_mirror_tree "$CKPT_DIR"     "$PERSIST_ROOT/checkpoints/${CKPT_DIR##*/}"
_mirror_tree "${CKPT_DIR}-sft" "$PERSIST_ROOT/checkpoints/${CKPT_DIR##*/}-sft"
_mirror_tree "$DATA_PACKED"  "$PERSIST_ROOT/$DATA_PACKED"
_mirror_tree "results"       "$PERSIST_ROOT/results"
[ -f "$CKPT_DIR/trace.csv" ] && _mirror_tree "$CKPT_DIR" "$PERSIST_ROOT/checkpoints/${CKPT_DIR##*/}"
mira_log "persistent store contents:"
find "$PERSIST_ROOT" -maxdepth 3 2>/dev/null | head -30 || true

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