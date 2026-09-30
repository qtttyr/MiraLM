#!/usr/bin/env bash
# Rescue: verify training state, then populate the persistent store.
#
# Two things this corrects, both learned the hard way:
#   1. training runs with --data-dir data/packed RELATIVE TO THE REPO ROOT, so
#      the shards live in <repo>/data/packed, not /kaggle/working/data/packed;
#   2. /kaggle/output does not exist until something creates it, so the persist
#      store was silently never written on this session.
#
# Read-only until the copy step. Safe to run.
set +e
sep() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
REPO="${REPO:-/kaggle/working/MiraLM}"
OUT="${OUT:-/kaggle/output}"
PERSIST="${PERSIST_ROOT:-$OUT/miralm-persist}"
MIRA_PERSIST="${MIRALM_PERSIST_DIR:-$OUT/persist-mira}"

sep "current step"
for k in last best; do
  f="$REPO/checkpoints/mira/$k/train_meta.json"
  [ -f "$f" ] && printf '  %-5s %s\n' "$k" "$(cat "$f" | tr -d '\n')"
done

sep "corpus, at the paths that actually matter"
for d in "$REPO/data/packed" /kaggle/working/data/packed; do
  if [ -f "$d/manifest.json" ]; then
    printf '  FOUND %s\n' "$d"
    head -c 400 "$d/manifest.json"; printf '\n'
  else
    printf '  none  %s\n' "$d"
  fi
done
du -sh "$REPO/data/packed" 2>/dev/null

sep "creating the persistent store (it does not exist on a fresh session)"
mkdir -p "$PERSIST" "$MIRA_PERSIST" "$PERSIST/checkpoints" || true
ls -ld "$OUT" "$PERSIST" "$MIRA_PERSIST"

sep "mirroring the trainer's own save target (newest weights)"
if [ -d "$REPO/checkpoints/mira/last" ]; then
  cp -r "$REPO/checkpoints/mira/last/." "$MIRA_PERSIST/" && echo "  -> $MIRA_PERSIST"
fi
if [ -d "$REPO/checkpoints/mira/best" ]; then
  cp -r "$REPO/checkpoints/mira/best/." "$PERSIST/checkpoints/mira-best/" && echo "  -> $PERSIST/checkpoints/mira-best"
fi

sep "mirroring the packed corpus (re-packing costs hours)"
if [ -f "$REPO/data/packed/manifest.json" ]; then
  mkdir -p "$PERSIST/data"
  cp -r "$REPO/data/packed" "$PERSIST/data/" && echo "  -> $PERSIST/data/packed"
fi
[ -f "$REPO/checkpoints/trace.csv" ] && cp "$REPO/checkpoints/trace.csv" "$PERSIST/" 2>/dev/null

sep "persistent store contents"
find "$OUT" -maxdepth 4 2>/dev/null | head -40
du -sh "$OUT" 2>/dev/null

sep "verdict"
n=$(find "$OUT" -name 'train_meta.json' 2>/dev/null | wc -l | tr -d ' ')
m=$([ -f "$PERSIST/data/packed/manifest.json" ] && echo yes || echo no)
echo "  checkpoints in persist: $n"
echo "  corpus in persist     : $m"
if [ "$n" -ge 1 ] && [ "$m" = yes ]; then
  echo "  SAFE — a Save & Run All can no longer lose this run."
else
  echo "  INCOMPLETE — see the warnings above before committing."
fi
