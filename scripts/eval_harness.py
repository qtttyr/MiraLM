"""lm-evaluation-harness integration for MiraLM.

Runs the five mandatory GIBC V2 benchmarks and writes a single JSON report.

Benchmarks:
  - hellaswag   (4-way MC accuracy)
  - arc_easy    (4-way MC accuracy)
  - piqa        (4-way MC accuracy)
  - winogrande  (4-way MC accuracy)
  - wikitext    (word-level perplexity)

Usage:
    python scripts/eval_harness.py \
        --ckpt-dir checkpoints/mira/last \
        --tokenizer-dir checkpoints/mira/last \
        --output results/eval_mira.json \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext"]


def run_eval(
    ckpt_dir: str,
    tokenizer_dir: str,
    output: str | None,
    tasks: list[str] | None = None,
    batch_size: int = 8,
    device: str | None = None,
    trust_remote_code: bool = True,
) -> dict:
    import lm_eval.evaluator as evaluator
    import lm_eval.tasks as tasks_mod

    # ensure our config registration is live
    from src.model.hf_interface import MiraLMForCausalLM, MiraConfig  # noqa: F401

    tasks = tasks or TASKS

    model_args = (
        f"pretrained={ckpt_dir}"
        f",tokenizer={tokenizer_dir}"
        f",trust_remote_code={trust_remote_code}"
    )
    if device:
        model_args += f",device={device}"

    print(f"evaluating: tasks={tasks}  ckpt={ckpt_dir}  batch_size={batch_size}")
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        fewshot_as_multiturn=False,
        apply_chat_template=False,
    )

    summary: dict = {}
    per_task: dict = {}
    for t in tasks:
        r = results["results"].get(t, {})
        per_task[t] = {k: v for k, v in r.items() if not k.startswith("alias")}
        if "acc,none" in r:
            summary[t] = {"accuracy": r["acc,none"]}
        elif "word_perplexity,none" in r:
            summary[t] = {"word_perplexity": r["word_perplexity,none"]}
        else:
            summary[t] = r

    out = {
        "model": str(ckpt_dir),
        "tasks": per_task,
        "summary": summary,
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"results saved: {output}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--tokenizer-dir", default=None, type=Path,
                    help="defaults to --ckpt-dir")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.ckpt_dir
    if args.output is None:
        args.output = Path("results") / f"eval_{args.ckpt_dir.parent.name}.json"

    out = run_eval(
        str(args.ckpt_dir),
        str(args.tokenizer_dir),
        str(args.output),
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(out.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())