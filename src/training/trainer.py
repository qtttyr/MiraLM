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
        print(f"training {total} steps, effective batch {self.tc.batch_size} * {self.loader.seq_len} tokens")
        print(f"  precision={self.tc.precision}  device={self.device}  ckpt_dir={self.ckpt_dir}")

        for self.step_num in range(total):
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
        if self._csv_file:
            self._csv_file.close()
        print("training complete")

    def _log(self, step, lr, loss, metrics):
        vals = {"step": step, "lr": lr, "loss": loss, **metrics, "load": self._expert_load_str()}
        line = "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items())
        print(f"[{step:>6d}] {line}")
        self._write_csv(vals)

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

    def save_checkpoint(self, path: Path, loss: float):
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
        meta = {"step": self.step_num, "loss": loss}
        import json as _json2
        (path / "train_meta.json").write_text(_json2.dumps(meta), encoding="utf-8")
        print(f"  saved checkpoint: {path} (step={self.step_num}, loss={loss:.4f})")
