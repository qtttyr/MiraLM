#!/usr/bin/env python3
"""Read-only training status: which step, what result, and is it resumable?

Touches nothing. Works locally and inside a Kaggle cell.

    python scripts/status.py                      # auto-discover checkpoints/
    python scripts/status.py --ckpt /kaggle/output/persist-mira/checkpoints

State lives in <ckpt>/{last,best}/train_meta.json, written by
Trainer.save_checkpoint on every save, plus a rolling trace.csv of the curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, EMBER, CYAN = (
    "\033[32m", "\033[33m", "\033[31m", "\033[38;5;208m", "\033[36m")


def head(s: str) -> None:
    print(f"\n{BOLD}{EMBER}=== {s} ==={RESET}")


def bar(frac: float, width: int = 42) -> str:
    filled = max(0, min(width, round(frac * width)))
    return (f"{EMBER}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}")


def find_roots(explicit: str | None) -> list[Path]:
    if explicit:
        return [Path(explicit)]
    seen: list[Path] = []
    for base in (Path("checkpoints"),
                 Path("/kaggle/working/checkpoints"),
                 Path("/kaggle/output/persist-mira/checkpoints"),
                 Path("/kaggle/output/miralm-persist/checkpoints")):
        if base.is_dir():
            seen.append(base)
    return seen


def read_meta(d: Path) -> dict | None:
    f = d / "train_meta.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  {RED}не читается: {f} ({exc}){RESET}")
        return None


def show_ckpt(name: str, d: Path, target: int) -> None:
    m = read_meta(d)
    if m is None:
        print(f"  {DIM}{name:<6} нет train_meta.json{RESET}")
        return
    step = int(m.get("step", 0))
    loss = float(m.get("loss", float("nan")))
    ppl = float(m.get("perplexity", float("nan")))
    toks = int(m.get("tokens", 0))
    hrs = float(m.get("elapsed_h", 0.0))
    tps = float(m.get("tokens_per_s", 0.0))
    sz = next((f.stat().st_size for f in sorted(d.glob("*.safetensors"))), 0)

    colour = GREEN if name == "best" else CYAN
    print(f"  {colour}{BOLD}{name:<6}{RESET} step {BOLD}{step:,}{RESET}"
          f"  loss {BOLD}{loss:.4f}{RESET}  ppl {BOLD}{ppl:.2f}{RESET}")
    print(f"         {DIM}{toks:,} токенов · {hrs:.2f} ч · {tps:,.0f} ток/с · "
          f"веса {sz/1e6:.0f} МБ{RESET}")
    if target > 0:
        frac = step / target
        print(f"         {bar(frac)} {frac*100:5.1f}%  "
              f"{DIM}из {target:,}{RESET}")


def show_curve(root: Path) -> None:
    csvs = sorted(root.glob("*/trace.csv")) or sorted(root.glob("trace.csv"))
    for f in csvs:
        rows = [l.split(",") for l in f.read_text().splitlines() if l.strip()]
        if len(rows) < 2:
            continue
        head_, *body = rows
        try:
            si, li = head_.index("step"), head_.index("loss")
        except ValueError:
            return
        pts = [(int(r[si]), float(r[li])) for r in body if r[li]]
        first, last = pts[0], pts[-1]
        best = min(pts, key=lambda p: p[1])
        print(f"  {DIM}{f.parent.name}/trace.csv — {len(pts)} точек{RESET}")
        print(f"     {first[0]:,} → {last[0]:,}   loss {first[1]:.4f} → "
              f"{last[1]:.4f}   {GREEN}минимум {best[1]:.4f} @ {best[0]:,}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="каталог checkpoints (auto, если не указан)")
    ap.add_argument("--target", type=int, default=0,
                    help="ожидаемый MAX_STEPS для полосы прогресса")
    a = ap.parse_args()

    target = a.target or 20000
    roots = find_roots(a.ckpt)
    if not roots:
        print(f"{RED}Ни одного каталога checkpoints не найдено.{RESET}")
        print(f"{DIM}Проверь, что training шёл в Kaggle и вес сохранились "
              f"в /kaggle/output/persist-mira/checkpoints{RESET}")
        return 1

    for root in roots:
        head(str(root))
        for name in ("last", "best"):
            d = root / name
            if d.is_dir():
                show_ckpt(name, d, target)
            else:
                print(f"  {DIM}{name:<6} отсутствует{RESET}")
        if (root / "trace.csv").is_file() or any(root.glob("*/trace.csv")):
            head("кривая обучения")
            show_curve(root)

        # сводка: сколько ещё осталось
        m = read_meta(root / "last") or read_meta(root / "best")
        if m and target > 0:
            step = int(m.get("step", 0))
            left = target - step
            head("оценка остатка")
            if left <= 0:
                print(f"  {GREEN}Цель {target:,} шагов достигнута.{RESET}")
            else:
                tps = float(m.get("tokens_per_s", 0)) or 1.0
                secs = left * 4096 / tps          # 4096 ток/шаг (4 x 1024)
                print(f"  осталось {BOLD}{left:,}{RESET} шагов ≈ "
                      f"{secs/3600:.1f} ч при {tps:,.0f} ток/с")

    head("resume")
    m = read_meta(roots[0] / "last")
    if m:
        print(f"  {GREEN}resume возможен{RESET} — cloud_run.sh продолжит с шага "
              f"{int(m.get('step',0)):,}")
        print(f"  {DIM}SFT запускать только после этого{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
