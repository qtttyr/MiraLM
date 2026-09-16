"""Demo utilities: model generation with the structured-output protocol."""

from __future__ import annotations

from pathlib import Path

import torch

from ..model.hf_interface import MiraLMForCausalLM


class MiraDemoModel:
    """Loads an SFT checkpoint and generates parseable structured answers."""

    def __init__(self, ckpt_dir: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MiraLMForCausalLM.from_pretrained(str(ckpt_dir)).to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        except Exception:
            self.tokenizer = None
        if self.tokenizer is None:
            raise RuntimeError("checkpoint has no tokenizer; re-train with tokenizer saved")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1,
        )
        new_ids = out[0, ids["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=False)

    def split_answer(self, generated: str) -> tuple[str, str]:
        """Return (visible, hidden) parts of the structured response."""
        if "<|answer|>" in generated:
            visible = generated.split("<|answer|>", 1)[1]
        else:
            visible = generated
        hidden = ""
        if "<|think|>" in generated:
            mid = generated.split("<|think|>", 1)[1]
            hidden = mid.split("<|answer|>", 1)[0]
        return visible, hidden