#!/usr/bin/env bash
# =============================================================================
# KAGGLE CELL — publish the step-11,000 "best" checkpoint to the HuggingFace Hub.
# =============================================================================
# Paste into a NEW cell in the SAME Kaggle session that still has
# /kaggle/working/MiraLM on disk, and run it BEFORE the session is closed.
#
# Two traps this cell exists to avoid:
#   1. "last" (step 17,000, loss 2.587) DIVERGED. Publishing it would ship the
#      broken model. The measured results belong to "best" (step 11,000,
#      loss 2.2113, ppl 9.13), so this cell refuses "last" by name.
#   2. The token must never be pasted in plaintext. It is read from a Kaggle
#      Secret (Add-ons > Secrets, name: HF_TOKEN) or from the environment.
#
# Idempotent: re-running only re-uploads. The repo is created PUBLIC so the
# jury can load the weights directly.
#
# Overrides:
#   REPO=/kaggle/working/MiraLM    checkout that holds checkpoints/
#   HF_REPO=vaprooll/MiraLM-47M    target hub repo id
#   HF_PRIVATE=1                   create the repo private (default: public)
# =============================================================================
set -uo pipefail

REPO="${REPO:-/kaggle/working/MiraLM}"
HF_REPO="${HF_REPO:-vaprooll/MiraLM-47M}"
SECRET_NAME="${SECRET_NAME:-HF_TOKEN}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- locate ------
[ -d "$REPO" ] || die "no checkout at $REPO — is this the same session?"
BEST="$REPO/checkpoints/mira/best"
LAST="$REPO/checkpoints/mira/last"
[ -d "$BEST" ] || die "missing $BEST — nothing to publish"

for f in config.json model.safetensors tokenizer.json; do
    [ -f "$BEST/$f" ] || die "$BEST is not HF-loadable (no $f)"
done

log "source : $BEST"
cat "$BEST/train_meta.json" 2>/dev/null || true
du -sh "$BEST"

if [ -f "$LAST/train_meta.json" ]; then
    log "refusing 'last' — the diverged run, kept on disk as evidence only:"
    cat "$LAST/train_meta.json"
fi

# -------------------------------------------------------------- token ------
if [ -z "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN unset — reading Kaggle Secret '$SECRET_NAME'"
    HF_TOKEN="$(SECRET_NAME="$SECRET_NAME" python3 - <<'PY' 2>/dev/null || true
import os
try:
    from kaggle_secrets import UserSecretsClient
    print(UserSecretsClient().get_secret(os.environ["SECRET_NAME"]), end="")
except Exception:
    pass
PY
)"
fi
[ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN — Add-ons > Secrets > add '$SECRET_NAME', then re-run"
log "token  : ${HF_TOKEN:0:4}…${HF_TOKEN: -4}  (len ${#HF_TOKEN})"

# ------------------------------------------------------------ hub lib ------
python3 -c 'import huggingface_hub' 2>/dev/null || pip install -q "huggingface_hub>=0.23"
[ -f "$REPO/scripts/push_to_hub.py" ] \
    || die "scripts/push_to_hub.py missing — run: git -C $REPO pull --ff-only"

# --------------------------------------------------------- model card ------
# Written into the checkpoint dir so upload_folder ships it as the repo card.
# Numbers are the measured step-11,000 results, never targets. '__HF_REPO__' is
# substituted below (the heredoc stays quoted so nothing in it is executed).
cat > "$BEST/README.md" <<'MD'
---
library_name: transformers
pipeline_tag: text-generation
tags:
  - from-scratch
  - mixture-of-experts
  - mamba
  - gibc-v2
---

# MiraLM-47M

A **47,640,968-parameter** hybrid language model trained **from scratch** — no
pretrained weights, no distillation — under a hard 50,000,000 cap, on a single
Tesla T4. Interleaved Mamba ∥ attention blocks (Jamba-style) with a sparse
**top-2-of-8** mixture of experts whose router is domain-seeded by a guide loss.

This repository holds the **step-11,000 best checkpoint** (train loss 2.2113,
perplexity 9.13) — the run that carries the measured results below. The
step-17,000 "last" checkpoint **diverged** (loss 2.587) and is deliberately not
published.

## Measured results

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | WikiText-103 PPL |
|---|---|---|---|---|---|
| **MiraLM-47M** | 33.0 | 27.0 | 50.0 | 50.2 | 2837.6 |

Read honestly: at 47.6M parameters and 45,056,000 tokens — roughly 4.7% of the
Chinchilla-optimal budget for this size — the model is undertrained by
construction. Multiple-choice commonsense sits near its random floor, and the
WikiText-103 perplexity reflects a domain shift, since the training corpus is
code / math / web text, not prose. The claim is not "we win on HellaSwag". It is
that the whole pipeline — architecture, budget enforcement, data, routing,
structured SFT, evaluation — runs end to end, reproduces from one config, and
fits the ceiling.

## Load it

```bash
git clone https://github.com/qtttyr/MiraLM && cd MiraLM
```

```python
from src.model.hf_interface import MiraLMForCausalLM  # registers model_type "mira"
from transformers import AutoTokenizer

repo = "__HF_REPO__"
model = MiraLMForCausalLM.from_pretrained(repo).eval()
tok   = AutoTokenizer.from_pretrained(repo)

prompt = "<|json|> Return a JSON object with product, price and stock: product mug, price 500, stock 23."
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:]))
```

No `trust_remote_code` needed: importing `hf_interface` registers the `mira`
model type locally, so `from_pretrained` resolves the checkpoint.
MD
# Substitute the repo id with python, not `sed -i`: BSD/macOS sed needs an
# explicit backup suffix (`sed -i '' ...`) and silently fails otherwise, which
# would publish the card with the placeholder still in it.
HF_REPO="$HF_REPO" python3 - "$BEST/README.md" <<'PY'
import os
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
if "__HF_REPO__" not in text:
    sys.exit("model card has no __HF_REPO__ placeholder — refusing to guess")
p.write_text(text.replace("__HF_REPO__", os.environ["HF_REPO"]), encoding="utf-8")
PY
[ $? -eq 0 ] || die "model card substitution failed"
grep -q "$HF_REPO" "$BEST/README.md" || die "model card does not name $HF_REPO"
log "model card: $BEST/README.md -> $HF_REPO"

# ---------------------------------------------------------------- push -----
cd "$REPO" || die "cannot cd $REPO"
PRIVFLAG=""
[ "${HF_PRIVATE:-0}" = "1" ] && PRIVFLAG="--private"

HF_TOKEN="$HF_TOKEN" python3 scripts/push_to_hub.py "$BEST" \
    --repo "$HF_REPO" \
    $PRIVFLAG \
    --commit-message "MiraLM-47M step-11,000 best (loss 2.2113, ppl 9.13)"
status=$?
[ "$status" -eq 0 ] || die "push failed (exit $status) — see the log above"
log "pushed → https://huggingface.co/$HF_REPO"

# -------------------------------------------------------------- verify -----
# Prove the repo is loadable the way a jury member loads it: pull config.json
# back from the hub, then build the model straight from the repo id.
log "round-tripping the weights back from the hub"
HF_TOKEN="$HF_TOKEN" HF_REPO="$HF_REPO" REPO="$REPO" python3 - <<'PY' \
    || log "WARNING: round-trip check failed — verify the hub page before submitting"
import json
import os
import sys

from huggingface_hub import hf_hub_download

repo = os.environ["HF_REPO"]
cfg = json.loads(open(hf_hub_download(repo, "config.json")).read())
print(f"  hub config.json : model_type={cfg.get('model_type')} "
      f"d_model={cfg.get('d_model')} n_layers={cfg.get('n_layers')} "
      f"vocab={cfg.get('vocab_size')}")

sys.path.insert(0, os.environ["REPO"])
from src.model.hf_interface import MiraLMForCausalLM
from transformers import AutoTokenizer

model = MiraLMForCausalLM.from_pretrained(repo)
tok = AutoTokenizer.from_pretrained(repo)
print(f"  loaded from hub : {model.num_unique_params():,} unique trainable params")
print(f"  tokenizer vocab : {tok.vocab_size:,}")
print("  ROUND-TRIP OK — this is exactly what the jury's snippet does.")
PY

log "DONE — https://huggingface.co/$HF_REPO"

