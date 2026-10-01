#!/usr/bin/env bash
# =============================================================================
# RECOVERY CELL — run this when a previous launch died with
#   "cannot fork() for git-remote-https: Resource temporarily unavailable"
#
# That error means the process table is full: a launch cell looped thousands of
# times and each iteration forked git. This cell (1) reaps the strays, (2) shows
# what state we are actually in, and (3) launches WITHOUT git at all.
#
# No git on the happy path: once the repo is cloned its HEAD is already correct,
# and every `git` call is another fork we do not need.
#
#   bash scripts/recover_and_launch.sh
# =============================================================================
set +e
sep() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
REPO="${REPO:-/kaggle/working/MiraLM}"
OUT="${OUT:-/kaggle/output}"
MAX_STEPS="${MAX_STEPS:-15000}"
SFT_STEPS="${SFT_STEPS:-1000}"

sep "1. reap stray processes left by the fork storm"
echo "   processes before: $(ps -e 2>/dev/null | wc -l)"
pkill -f 'git-remote-https' 2>/dev/null
pkill -f 'git fetch' 2>/dev/null
pkill -f 'kaggle_cell.sh' 2>/dev/null
# do NOT kill a training run that is genuinely making progress
if pgrep -f 'scripts/train.py' >/dev/null 2>&1; then
    echo "   WARNING: a training process is still alive — leaving it alone."
else
    echo "   no live trainer; safe to continue"
fi
sleep 2
echo "   processes after:  $(ps -e 2>/dev/null | wc -l)"

sep "2. what do we actually have"
cd "$REPO" 2>/dev/null || { echo "FATAL: $REPO missing — clone it once, manually"; exit 1; }
echo "   repo HEAD : $(cat .git/HEAD 2>/dev/null | head -c 60)"
for f in data/packed/manifest.json checkpoints/mira/last/train_meta.json; do
    if [ -f "$f" ]; then echo "   OK      $f"; else echo "   MISSING $f"; fi
done
if [ -f checkpoints/mira/last/train_meta.json ]; then
    echo "   current step: $(grep -o '"step"[^,]*' checkpoints/mira/last/train_meta.json | head -1)"
fi

sep "3. restore anything missing from the persistent store (no git)"
PERSIST="${PERSIST_ROOT:-$OUT/miralm-persist}"
if [ -d "$PERSIST" ]; then
    [ -f "$REPO/data/packed/manifest.json" ] || {
        mkdir -p "$REPO/data"
        cp -r "$PERSIST/data/packed" "$REPO/data/" 2>/dev/null \
            && echo "   restored data/packed from $PERSIST"
    }
    [ -f "$REPO/checkpoints/mira/last/train_meta.json" ] || {
        mkdir -p "$REPO/checkpoints/mira"
        cp -r "$OUT/persist-mira" "$REPO/checkpoints/mira/last" 2>/dev/null \
            && echo "   restored checkpoints/mira/last from $OUT/persist-mira"
    }
else
    echo "   no persistent store at $PERSIST"
fi

sep "4. launch (no git, single invocation)"
export PERSIST_ROOT="${PERSIST_ROOT:-$OUT/miralm-persist}"
export MIRALM_PERSIST_DIR="${MIRALM_PERSIST_DIR:-$OUT/persist-mira}"
export DATA_PACKED="${DATA_PACKED:-data/packed}"
export CKPT_DIR="${CKPT_DIR:-checkpoints/mira}"
export DATA_RAW_DIR="${DATA_RAW_DIR:-$REPO/data/raw}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/model_sparsemind.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_sparsemind.yaml}"
export MAX_STEPS SFT_STEPS
echo "   MAX_STEPS=$MAX_STEPS  SFT_STEPS=$SFT_STEPS"
echo "   starting — this takes hours; do not re-run this cell"
exec bash scripts/cloud_run.sh 2>&1 | tee "$OUT/run.log"
