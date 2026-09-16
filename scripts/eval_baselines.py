"""Evaluate baseline models for controlled comparison (submission table).

Runs the SAME harness + task set on GPT-2-117M / Pythia-70M so that the
README comparison table reports apples-to-apples numbers.

Usage:
    python scripts/eval_baselines.py --output results/eval_baselines.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES = {
    "gpt2_117m": "gpt2",
    "pythia_70m": "EleutherAI/pythia-70m-deduped",
}

TASKS = "hellaswag,arc_easy,piqa,winogrande,wikitext"


def run_baseline(name: str, model: str, batch_size: int, output_path: Path) -> dict:
    cmd = (
        f"{sys.executable} -m lm_eval --model hf "
        f"--model_args pretrained={model},trust_remote_code=True "
        f"--tasks {TASKS} --batch_size {batch_size} --output_path {output_path}"
    )
    print(f"=== {name} ({model}) ===")
    print(cmd)
    subprocess.run(cmd, check=True, shell=True, cwd=str(ROOT))

    # lm_eval writes results into a subdirectory, mirror the path it uses
    found = sorted(output_path.parent.glob("**/*.json")) if output_path.exists() else []
    if not found:
        return {"error": "no results file produced", "task": TASKS}
    latest = found[-1]
    with open(latest, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "model": model,
        "results": {
            t: {k: v for k, v in data["results"].get(t, {}).items() if k.startswith(("acc", "word_perplexity"))}
            for t in [t.strip() for t in TASKS.split(",")]
        },
        "source_file": str(latest),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results/eval_baselines.json"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of {gpt2_117m,pythia_70m}")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else list(BASELINES)
    output: dict[str, dict] = {}
    for n in names:
        n = n.strip()
        if n not in BASELINES:
            print(f"unknown baseline {n!r}; choose from {list(BASELINES)}", file=sys.stderr)
            continue
        output[n] = run_baseline(n, BASELINES[n], args.batch_size, args.output.parent / n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"baseline eval report saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())