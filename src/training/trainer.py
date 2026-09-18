"""Core training loop: AMP, loss composition, checkpointing, metrics.

Designed for T4 fp16 training (GIBC V2). The trainer operates on raw MiraLM
(not the HF wrapper) and converts to HuggingFace format only at checkpoint
time so that lm-evaluation-harness can load weights directly.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from ..config import ModelConfig, TrainingConfig
from ..model.architecture import MiraLM
from ..model.hf_interface import MiraConfig, MiraLMForCausalLM
from .loader import PackedDataLoader
from .schedules import cosine_warmup_lr, guide_weight


def _autocast(device: torch.device, precision: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    if dtype == torch.float32 or device.type != "cuda":
        return autocast(device.type, enabled=False)
    return autocast(device.type, dtype=dtype)


class Trainer:
    def __init__(
        self,
        cfg: ModelConfig,
        train_cfg: TrainingConfig,
        loader: PackedDataLoader,
        ckpt_dir: Path,
        tokenizer=None,
    ):
        self.cfg = cfg
        self.tc = train_cfg
        self.loader = loader
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

        torch.manual_seed(train_cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiraLM(cfg).to(self.device)
        self._wandb = self._try_init_wandb()
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler(enabled=(train_cfg.precision == "fp16" and self.device.type == "cuda"))
        )
        self.optimizer = self._make_optimizer()
        self.step_num = 0
        self.best_loss: float = math.inf
        self._last_loss: float = math.inf
        self._csv_file: Optional[Any] = None
        self._csv_writer = None

    def _make_optimizer(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.tc.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.tc.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def train(self):
        micro_steps = max(1, self.tc.batch_size // self.tc.micro_batch)
        total = self.tc.max_steps
        start = self.step_num
        print(f"training {total} steps, effective batch {self.tc.batch_size} * {self.loader.seq_len} tokens")
        print(f"  precision={self.tc.precision}  device={self.device}  ckpt_dir={self.ckpt_dir}")
        print(f"  resume at step {start}/{total}")

        for self.step_num in range(start, total):
            lr = cosine_warmup_lr(self.step_num, total, self.tc.max_lr, self.tc.min_lr, self.tc.warmup_frac)
            gw = guide_weight(self.step_num, self.tc.guide_steps, self.tc.guide_ramp)
            self.optimizer.zero_grad()

            accum_metrics: dict[str, float] = {}
            for _ in range(micro_steps):
                ids, domains = self.loader.next_batch()
                ids, domains = ids.to(self.device), domains.to(self.device)
                with _autocast(self.device, self.tc.precision):
                    out = self.model(ids, domain=domains, guide_w=gw)
                    logits = out.logits[:, :-1].reshape(-1, self.cfg.vocab_size)
                    labels = ids[:, 1:].reshape(-1)
                    loss = F.cross_entropy(logits, labels)
                    m = out.metrics or {}
                    loss = loss + self.cfg.moe.z_loss_coef * m.get("z_loss", 0.0)
                    loss = loss + self.cfg.moe.aux_loss_coef * m.get("aux_loss", 0.0)
                    loss = loss + m.get("guide_loss", 0.0)
                    loss_scaled = loss / micro_steps

                if self.scaler is not None:
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                # aggregate metrics
                for k in ("z_loss", "aux_loss", "guide_loss"):
                    mv = m.get(k, 0.0)
                    mv = mv.detach().item() if torch.is_tensor(mv) else float(mv)
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + mv / micro_steps

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
                self.optimizer.step()

            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr
            self._last_loss = float(loss.detach())

            if self.tc.log_interval > 0 and (self.step_num + 1) % self.tc.log_interval == 0:
                self._log(self.step_num + 1, lr, float(loss.detach()), accum_metrics)
            if self.tc.save_interval > 0 and (self.step_num + 1) % self.tc.save_interval == 0:
                avg = float(loss.detach())
                self._last_loss = avg
                self.save_checkpoint(self.ckpt_dir / "last", avg)
                if avg < self.best_loss:
                    self.best_loss = avg
                    self.save_checkpoint(self.ckpt_dir / "best", avg)

        self.save_checkpoint(self.ckpt_dir / "last", self._last_loss)
        if self._wandb is not None:
            self._wandb.log({"train/final_perplexity": math.exp(self._last_loss)})
        if self._csv_file:
            self._csv_file.close()
        print("training complete")

    def _try_init_wandb(self) -> Any:
        """Optional W&B logging (via WANDB_API_KEY or `wandb login`).

        Enabled only when credentials exist and not WANDB_DISABLED=1;
        never fails the run if wandb is unavailable.
        """
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001 — logging is optional
            print(f"[wandb] not installed ({exc}) — skipping logging")
            return None
        if os.environ.get("WANDB_DISABLED") == "1":
            return None
        key = os.environ.get("WANDB_API_KEY")
        logged_in = os.path.exists(os.path.join(os.path.expanduser("~"), ".netrc"))
        if not (key or logged_in):
            print("[wandb] no credentials — skip logging (set WANDB_API_KEY or run `wandb login`)")
            return None
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "MiraLM"),
                config={"model": self.cfg.dict(), "train": self.tc.dict()} if hasattr(self.cfg, "dict") else {},
                reinit=True,
            )
            return wandb if run is not None else None
        except Exception as exc:  # noqa: BLE001 — never block training
            print(f"[wandb] init failed ({exc}) — continuing without logging")
            return None

    def _log(self, step, lr, loss, metrics):
        vals = {"step": step, "lr": lr, "loss": loss, **metrics, "load": self._expert_load_str()}
        line = "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items())
        print(f"[{step:>6d}] {line}")
        self._write_csv(vals)
        if self._wandb is not None:
            self._wandb.log(
                {"step": step, "train/loss": loss, "train/perplexity": math.exp(loss), "train/lr": lr}
            )

    def _write_csv(self, vals):
        if self._csv_file is None:
            self._csv_file = open(self.ckpt_dir / "trace.csv", "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(vals.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in vals.items()})
        self._csv_file.flush()

    def _expert_load_str(self):
        if not hasattr(self.model, "moe") or self.model.moe is None:
            return ""
        with torch.no_grad():
            out = self.model(torch.randint(0, self.cfg.vocab_size, (1, 8), device=self.device))
            if out.metrics and "expert_load" in out.metrics:
                loads = out.metrics["expert_load"].cpu().tolist()
                return " ".join(f"{v:.2f}" for v in loads)
        return ""

    def load_resume(self, path: Path) -> None:
        """Load a saved checkpoint and continue training from its step.

        Weights come from the HF-format checkpoint; optimizer/AMP states are
        rebuilt fresh (acceptable for this scale). Data iteration re-seeks so
        the resumed run sees exactly the sequence the interrupted one would.
        """
        import json as _json

        path = Path(path)
        if not (path / "train_meta.json").exists():
            raise FileNotFoundError(f"not a trainer checkpoint: {path}")
        wrapper = MiraLMForCausalLM.from_pretrained(str(path))
        self.model.load_state_dict(wrapper.model.state_dict(), strict=True)
        meta = _json.loads((path / "train_meta.json").read_text(encoding="utf-8"))
        step = int(meta.get("step", 0))
        loss = float(meta.get("loss", math.inf))
        self.step_num = step
        self._last_loss = loss
        self.best_loss = loss
        self.loader.seek(step)
        print(f"resumed from {path} at step={step} loss={loss:.4f}")

    def save_checkpoint(self, path: Path, loss: float):
        import json as _json

        path.mkdir(parents=True, exist_ok=True)
        wrapper = MiraLMForCausalLM(MiraConfig.from_model_config(self.cfg))
        wrapper.model.load_state_dict(self.model.state_dict(), strict=True)
        wrapper.save_pretrained(str(path), safe_serialization=True)
        if self.tokenizer is not None:
            tok_path = path / "tokenizer.json"
            if not tok_path.exists():
                self.tokenizer.save(tok_path)
            # also write tokenizer_config.json for AutoTokenizer compatibility
            tc_path = path / "tokenizer_config.json"
            if not tc_path.exists():
                import json as _json
                tc_path.write_text(_json.dumps({
                    "bos_token": "eos",
                    "eos_token": "eos",
                    "pad_token": "<|pad|>",
                    "unk_token": " unk",
                    "model_max_length": 1024,
                }, indent=2), encoding="utf-8")
        meta = {"step": self.step_num, "loss": float(loss), "perplexity": math.exp(loss)}
        (path / "train_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
        self._mirror_persist(path, loss)
        print(f"  saved checkpoint: {path} (step={self.step_num}, loss={loss:.4f})")

    def _mirror_persist(self, path: Path, loss: float) -> None:
        """Best-effort copy of a freshly saved checkpoint into a persistent dir.

        Kaggle keeps `/kaggle/output` alive after the 12h session dies, and
        `/kaggle/working` is wiped. Point MIRALM_PERSIST_DIR at
        `/kaggle/output/persist-<name>` (or any folder of your choosing) and the
        latest weights land there at every save — no manual snapshot cell needed.
        Nothing fails if the target is unwritable; this is pure insurance.
        """
        import shutil as _shutil

        persist = os.environ.get("MIRALM_PERSIST_DIR")
        if not persist:
            return
        target = Path(persist)
        try:
            target.mkdir(parents=True, exist_ok=True)
            for entry in path.iterdir():
                _shutil.copy2(entry, target / entry.name)
            print(f"  mirrored checkpoint -> {target} (persistent)")
        except Exception as e:  # never break training over a mirror failure
            print(f"  [warn] mirror to {target} failed ({e}) — continuing", file=sys.stderr)
        meta = {"step": self.step_num, "loss": loss}
        import json as _json2
        (path / "train_meta.json").write_text(_json2.dumps(meta), encoding="utf-8")
        print(f"  saved checkpoint: {path} (step={self.step_num}, loss={loss:.4f})")
