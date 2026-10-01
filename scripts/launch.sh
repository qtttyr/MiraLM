#!/usr/bin/env bash
# =============================================================================
#  MIRA-LM — ONE-SHOT LAUNCHER.  Paste the whole thing into a %%bash cell.
# =============================================================================
set +e
REPO_URL="https://github.com/qtttyr/MiraLM.git"
WORK=/kaggle/working
OUT=/kaggle/output
REPO=$WORK/MiraLM
PERSIST=$OUT/miralm-persist
MAX_STEPS=15000
SFT_STEPS=1000

echo "############ STEP 1  directories"
mkdir -p "$WORK" "$OUT" "$PERSIST" "$OUT/persist-mira"
echo "  $WORK  -> $([ -d "$WORK" ] && echo ok || echo FAIL)"
echo "  $OUT   -> $([ -d "$OUT" ]  && echo ok || echo FAIL)"

echo "############ STEP 2  repository"
if [ -f "$REPO/scripts/cloud_run.sh" ]; then
    echo "  already present, no clone"
else
    echo "  cloning..."
    rm -rf "$REPO"
    git clone -q --depth 1 "$REPO_URL" "$REPO"
fi
cd "$REPO" || { echo "FATAL: cannot enter $REPO"; exit 1; }
echo "  HEAD $(git rev-parse --short HEAD 2>/dev/null)"

echo "############ STEP 3  state"
for f in checkpoints/mira/last/train_meta.json data/packed/manifest.json; do
    if [ -f "$REPO/$f" ]; then echo "  ok      $f"; else echo "  MISSING $f"; fi
done

echo "############ STEP 4  backup to the volume that survives a wipe"
mkdir -p "$PERSIST/data" "$PERSIST/checkpoints"
[ -d "$REPO/checkpoints/mira/last" ] && cp -r "$REPO/checkpoints/mira/last" "$OUT/persist-mira/" 2>/dev/null && echo "  weights -> $OUT/persist-mira"
[ -d "$REPO/checkpoints/mira/best" ] && cp -r "$REPO/checkpoints/mira/best"  "$PERSIST/checkpoints/" 2>/dev/null && echo "  best    -> $PERSIST/checkpoints"
[ -d "$REPO/data/packed" ] && cp -r "$REPO/data/packed" "$PERSIST/data/" 2>/dev/null && echo "  corpus  -> $PERSIST/data"
du -sh "$OUT" 2>/dev/null

echo "############ STEP 5  run"
export DATA_RAW_DIR="$REPO/data/raw"
export DATA_PACKED=data/packed
export CKPT_DIR=checkpoints/mira
export MODEL_CONFIG=configs/model_sparsemind.yaml
export TRAIN_CONFIG=configs/train_sparsemind.yaml
export PERSIST_ROOT="$PERSIST"
export MIRALM_PERSIST_DIR="$OUT/persist-mira"
export MAX_STEPS SFT_STEPS
bash scripts/cloud_run.sh 2>&1 | tee "$OUT/run.log"
echo "############ DONE rc=${PIPESTATUS[0]}"
