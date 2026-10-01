#!/usr/bin/env bash
# Read-only: show where the repo, the corpus and the weights actually are.
set +e
echo "=== /kaggle/working ==="; ls -la /kaggle/working/ 2>&1 | head -20
echo; echo "=== /kaggle/output ==="; ls -la /kaggle/output/ 2>&1 | head -20
echo; echo "=== any cloud_run.sh ==="; find /kaggle -maxdepth 4 -name cloud_run.sh 2>/dev/null | head
echo; echo "=== any train_meta.json ==="; find /kaggle -maxdepth 6 -name train_meta.json 2>/dev/null | head
echo; echo "=== any manifest.json (corpus) ==="; find /kaggle -maxdepth 6 -name manifest.json 2>/dev/null | head
echo; echo "=== processes ==="; echo "count: $(ps -e | wc -l)"; pgrep -af 'train.py|cloud_run' 2>/dev/null | head
