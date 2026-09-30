# =============================================================================
# KAGGLE CELL #1  —  put this as the FIRST cell of the notebook, once.
# =============================================================================
# Why this exact form. "Save & Run All" wipes /kaggle/working on EVERY run, so
# a notebook whose first cell starts with `cd /kaggle/working/MiraLM` dies
# immediately on the second run with:
#     cd: /kaggle/working/MiraLM: No such file or directory
#     bash: scripts/cloud_run.sh: No such file or directory
#     tee: /kaggle/output/run.log: No such file or directory
# All three are the same failure cascading from the missing directory.
#
# This cell depends on nothing but the network: it creates the dirs, clones or
# resets the repo, and only then enters it. /kaggle/output survives the wipe,
# so the weights and the packed corpus come back from there.
# =============================================================================

REPO_URL="https://github.com/qtttyr/MiraLM.git"
MAX_STEPS="${MAX_STEPS:-15000}"
SFT_STEPS="${SFT_STEPS:-1000}"

mkdir -p /kaggle/working /kaggle/output
cd /kaggle/working || exit 1

if [ -d MiraLM/.git ]; then
    echo ">>> repo present, resetting to origin/main"
    git -C MiraLM fetch -q --depth 1 origin
    git -C MiraLM reset -q --hard origin/main
else
    echo ">>> cloning $REPO_URL"
    git clone -q --depth 1 "$REPO_URL" MiraLM
fi

cd MiraLM || exit 1
echo ">>> repo HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
test -f scripts/kaggle_cell.sh || { echo "FATAL: bootstrap missing"; exit 1; }

MAX_STEPS="$MAX_STEPS" SFT_STEPS="$SFT_STEPS" \
    bash scripts/kaggle_cell.sh 2>&1 | tee /kaggle/output/run.log
exit ${PIPESTATUS[0]}

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/qtttyr/MiraLM.git}"
REPO_DIR="${REPO_DIR:-/kaggle/working/MiraLM}"
WORK="${WORK:-/kaggle/working}"
OUT="${OUT:-/kaggle/output}"
LOG="$OUT/run.log"

# Kaggle provides these paths, but not always on the very first cell of a
# re-run — create them before anything tries to use them.
mkdir -p "$WORK" "$OUT" 2>/dev/null || true

log() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }

log "working dir : $(pwd)"
log "repo target : $REPO_DIR"
log "output dir  : $OUT"

# --------------------------------------------------------------- repo ------
if [ -d "$REPO_DIR/.git" ]; then
    log "repo exists — pulling latest"
    git -C "$REPO_DIR" fetch --depth 1 origin 2>/dev/null || true
    git -C "$REPO_DIR" reset --hard origin/main 2>/dev/null \
        || git -C "$REPO_DIR" pull --ff-only 2>/dev/null \
        || log "WARNING: git pull failed — continuing with the on-disk copy"
else
    log "cloning $REPO_URL"
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
log "repo ready at $(pwd) @ $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --------------------------------------------------------------- python ----
# The repo ships no venv on Kaggle; prefer one, else system python.
PY="$REPO_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || echo python3)"
fi
log "python      : $PY"
"$PY" -c 'import torch' 2>/dev/null || log "WARNING: torch import failed"

# --------------------------------------------------------------- persist ---
# Mirror target lives on the volume that survives a session restart.
export PERSIST_ROOT="${PERSIST_ROOT:-$OUT/miralm-persist}"
export MIRALM_PERSIST_DIR="${MIRALM_PERSIST_DIR:-$OUT/persist-mira}"
export DATA_PACKED="${DATA_PACKED:-data/packed}"
export CKPT_DIR="${CKPT_DIR:-checkpoints/mira}"
export DATA_RAW_DIR="${DATA_RAW_DIR:-$REPO_DIR/data/raw}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export SFT_STEPS="${SFT_STEPS:-2000}"
log "persist root: $PERSIST_ROOT"

mkdir -p "$PERSIST_ROOT" "$MIRALM_PERSIST_DIR"

# --------------------------------------------------------------- run -------
# pipefail so a pipeline failure is not hidden by tee.
set -o pipefail
bash scripts/cloud_run.sh 2>&1 | tee "$LOG"
status=$?
log "cloud_run exit status: $status"
exit $status
