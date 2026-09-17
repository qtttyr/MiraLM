#!/usr/bin/env python3
"""Push a HuggingFace-format checkpoint (config.json + model.safetensors + tokenizer)
to the HuggingFace Hub — the free, permanent home for trained weights.

The Hub has no 12-h session limit, no per-session disk wipe, and survives a dead
Kaggle kernel, a lost laptop, whatever. It is also the artifact the jury can load
and try directly (MiraLMForCausalLM.from_pretrained("<repo>")).

Used by scripts/cloud_run.sh when HF_REPO is set (optional; no-op otherwise).

Example:
    HF_TOKEN=hf_... python scripts/push_to_hub.py \\
        results/mira-sft --repo vemimo/MiraLM-47M --private
"""

import argparse
import json
import os
import pathlib

from huggingface_hub import create_repo, upload_folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", type=pathlib.Path, help="HF checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--repo", type=str, required=True, help="hub repo id, e.g. vemimo/MiraLM-47M")
    ap.add_argument("--token", type=str, default=None, help="HF token (default: HF_TOKEN env)")
    ap.add_argument("--private", action="store_true", help="create repo as private")
    ap.add_argument("--commit-message", type=str, default="update from MiraLM training run")
    args = ap.parse_args()

    ckpt = args.ckpt_dir
    if not (ckpt / "config.json").exists() or not (ckpt / "model.safetensors").exists():
        ap.error(f"not an HF checkpoint: {ckpt} (missing config.json/model.safetensors)")

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("no HF_TOKEN — skipping hub push (set HF_TOKEN to enable)", file=os.stderr)
        return 2

    print(f"ensuring repo {args.repo} (private={args.private})")
    create_repo(
        args.repo,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )
    print(f"uploading {ckpt} -> hub:{args.repo}")
    upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        token=token,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"uploaded — https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
