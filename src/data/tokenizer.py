"""From-scratch ByteLevel BPE tokenizer for MiraLM (vocab 24k).

ByteLevel BPE (GPT-2 recipe) is robust to any UTF-8 text and needs no
pretokenizer heuristics — a good match for the sub-token roughness a ≤50M
model expects. Special tokens cover the structured-output protocol used
during SFT and the domain markers exposed to the trainer.

The tokenizer is stored as a standard `tokenizer.json`, so
transformers.AutoTokenizer can load it natively for lm-evaluation-harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from tokenizers import Tokenizer as _TTokenizer
from tokenizers import decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|endoftext|>",
    "<|json|>",
    "<|sql|>",
    "<|cot|>",
    "<|think|>",
    "<|answer|>",
]


def build_bpe(unk_token: str = "<|endoftext|>") -> _TTokenizer:
    """A fresh, untrained ByteLevel BPE tokenizer."""
    tok = _TTokenizer(BPE(unk_token=unk_token, byte_fallback=True))
    tok.normalizer = normalizers.Sequence([normalizers.NFC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel(add_prefix_space=False)
    return tok


def train_bpe_from_files(
    text_files: Iterable[Union[str, Path]],
    out_path: Union[str, Path],
    vocab_size: int = 24000,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
) -> "MiraTokenizer":
    """Train a ByteLevel BPE tokenizer on plain-text files and save it.

    Special tokens keep their slots at the head of the vocab (ids 0..k-1),
    so `<|pad|>/<|endoftext|>` are `0/1` by construction.
    """
    spec = list(special_tokens) if special_tokens is not None else list(SPECIAL_TOKENS)

    tok = build_bpe()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=spec,
        min_frequency=min_frequency,
        show_progress=True,
    )
    tok.train(files=[str(f) for f in text_files], trainer=trainer)

    # Attach the EOS-template only now that ids exist.
    eos_id = tok.token_to_id("<|endoftext|>")
    tok.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    tok.save(str(out_path))
    return MiraTokenizer.load(out_path)


class MiraTokenizer:
    """Thin, type-safe wrapper around a tokenizers.Tokenizer."""

    def __init__(self, tok: _TTokenizer) -> None:
        self.tok = tok
        ids = tok.get_vocab()
        self.vocab_size: int = len(ids)
        self.token_to_id: dict[str, int] = ids
        self.id_to_token: dict[int, str] = {v: k for k, v in ids.items()}
        self.pad_id = ids.get("<|pad|>", 0)
        self.eos_id = ids.get("<|endoftext|>", 1)

    @property
    def bos_id(self) -> int:
        return self.eos_id  # reuse EOS as BOS (no dedicated token)

    # ------------------------------------------------------------ codec
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if add_special_tokens:
            return self.tok.encode(text).ids
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(ids))

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def __call__(self, text: Union[str, list]) -> dict:
        """Minimal batched-encode API mirroring a transformers tokenizer."""
        if isinstance(text, str):
            enc = self.tok.encode(text)
            return {"input_ids": enc.ids, "attention_mask": [1] * len(enc.ids)}
        encoded = self.tok.encode_batch(list(text))
        ids = [e.ids for e in encoded]
        return {"input_ids": ids, "attention_mask": [[1] * len(r) for r in ids]}

    # ---------------------------------------------------------------- I/O
    def save(self, path: Union[str, Path]) -> None:
        self.tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MiraTokenizer":
        return cls(_TTokenizer.from_file(str(path)))

    def get_hf_tokenizer(self):
        """A transformers tokenizer (for lm-evaluation-harness)."""
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_object=self.tok,
            pad_token="<|pad|>",
            eos_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
        )