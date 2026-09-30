#!/usr/bin/env bash
# Read-only triage: where does training state actually live right now?
# Safe to run repeatedly, touches nothing.
set +e
sep() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }

sep "paths exist?"
for p in /kaggle/working /kaggle/output /kaggle/working/MiraLM \
         /kaggle/working/checkpoints /kaggle/working/data/packed \
         /kaggle/output/miralm-persist /kaggle/output/persist-mira; do
  if [ -e "$p" ]; then printf '  OK      %s\n' "$p"; else printf '  MISSING %s\n' "$p"; fi
done

sep "any checkpoint dirs ANYWHERE under /kaggle (maxdepth 6)"
find /kaggle -maxdepth 6 -type d -name 'last' 2>/dev/null | head -20
[ -z "$(find /kaggle -maxdepth 6 -type d -name 'last' 2>/dev/null)" ] && echo "  none found"

sep "any train_meta.json ANYWHERE (this is the real marker of a checkpoint)"
find /kaggle -maxdepth 7 -name 'train_meta.json' 2>/dev/null | head -20
[ -z "$(find /kaggle -maxdepth 7 -name 'train_meta.json' 2>/dev/null)" ] && echo "  none found"

sep "weights (*.safetensors) ANYWHERE"
find /kaggle -maxdepth 8 -name '*.safetensors' 2>/dev/null | head -20
[ -z "$(find /kaggle -maxdepth 8 -name '*.safetensors' 2>/dev/null)" ] && echo "  none found"

sep "corpus: manifest + shard size"
for m in /kaggle/working/data/packed/manifest.json \
         /kaggle/output/miralm-persist/data/packed/manifest.json; do
  if [ -f "$m" ]; then printf '  FOUND %s\n' "$m"; cat "$m" | head -12; else printf '  none at %s\n' "$m"; fi
done
du -sh /kaggle/working/data 2>/dev/null
du -sh /kaggle/working/data/packed 2>/dev/null

sep "/kaggle/output listing"
ls -la /kaggle/output 2>/dev/null | head -30

sep "tokenizer (needed for SFT; if absent, the corpus must be re-packed)"
find /kaggle -maxdepth 7 -name 'tokenizer.json' 2>/dev/null | head -10
[ -z "$(find /kaggle -maxdepth 7 -name 'tokenizer.json' 2>/dev/null)" ] && echo "  none found"

sep "verdict"
if find /kaggle -maxdepth 7 -name 'train_meta.json' 2>/dev/null | grep -q .; then
  echo "  A checkpoint EXISTS -> cloud_run.sh will resume, training is safe."
else
  echo "  NO checkpoint anywhere -> pre-training starts from zero."
  echo "  Do NOT waste 14h: lower MAX_STEPS, or re-run fetch_corpus + prepare_data first."
fi
