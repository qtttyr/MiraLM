"""Every budget gate must pass on every supported config."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["configs/model_sparsemind.yaml", "configs/model_dense_fallback.yaml"]


def run_budget(config: str) -> str:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "param_budget.py"), "--config", config],
        capture_output=True, text=True, cwd=ROOT,
    )
    return r.stdout + r.stderr


def test_budget_passes_both_configs():
    for cfg in CONFIGS:
        assert "✓ PASS" in run_budget(cfg), f"budget gate failed for {cfg}"


def test_budget_rejects_over_limit():
    over = "configs/model_sparsemind.yaml"
    out = run_budget(over)
    assert "Trainable parameters" in out
    assert "PASS" in out