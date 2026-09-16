"""SFT formatting: instruction examples -> token sequences.

The structured-output protocol wraps every example in the model's special
tokens so answers stay machine-parseable. Three variants are supported:

    json  <|json|> <instruction> <|answer|> <json-doc> <|endoftext|>
    sql   <|sql|>  <instruction> <|answer|> <sql-query> <|endoftext|>
    cot   <|cot|>  <instruction> <|think|> <reasoning> <|answer|> <final> <|endoftext|>

The `<|think|>` block is the hidden chain-of-thought; only text after
`<|answer|>` is surfaced to the end user (used by the demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..data.tokenizer import MiraTokenizer

DOMAIN_MARKERS: dict[str, str] = {
    "json": "<|json|>",
    "sql": "<|sql|>",
    "cot": "<|cot|>",
}

_DELIM_ANSWER = "<|answer|>"
_DELIM_THINK = "<|think|>"


@dataclass
class SFTExample:
    domain: str                       # "json" | "sql" | "cot"
    instruction: str
    response: str                     # final/surfaced answer
    reasoning: Optional[str] = None   # hidden chain-of-thought (cot only)


def format_sft_tokens(ex: SFTExample, tok: MiraTokenizer) -> list[int]:
    """Encode one instruction example into the structured-output protocol."""
    if ex.domain not in DOMAIN_MARKERS:
        raise ValueError(f"unsupported SFT domain {ex.domain!r}, want one of {sorted(DOMAIN_MARKERS)}")

    marker_id = tok.token_to_id[DOMAIN_MARKERS[ex.domain]]
    answer_id = tok.token_to_id[_DELIM_ANSWER]
    think_id = tok.token_to_id.get(_DELIM_THINK)

    ids = [marker_id]
    ids += tok.encode(ex.instruction, add_special_tokens=False)
    if ex.reasoning:
        if think_id is None:
            raise ValueError("tokenizer lacks <|think|>; needed for cot reasoning")
        ids += [think_id]
        ids += tok.encode(ex.reasoning, add_special_tokens=False)
    ids += [answer_id]
    ids += tok.encode(ex.response, add_special_tokens=False)
    ids += [tok.eos_id]
    return ids


def parse_sft_document(ids: list[int], tok: MiraTokenizer) -> SFTExample:
    """Inverse of `format_sft_tokens` — used by tests and the demo."""
    answer_id = tok.token_to_id[_DELIM_ANSWER]
    think_id = tok.token_to_id.get(_DELIM_THINK)

    marker = None
    for dom, m in DOMAIN_MARKERS.items():
        mid = tok.token_to_id.get(m)
        if mid is not None and ids and ids[0] == mid:
            marker = dom
            break
    if marker is None:
        raise ValueError("no domain marker at sequence head")

    end = tok.eos_id
    end_at = ids.index(end) if end in ids else len(ids)
    a = ids.index(answer_id)

    instruction = tok.decode(ids[1:a])
    if marker == "cot":
        t = ids.index(think_id)
        reasoning = tok.decode(ids[t + 1 : a])
        response = tok.decode(ids[a + 1 : end_at])
        return SFTExample(marker, instruction, response, reasoning)

    response = tok.decode(ids[a + 1 : end_at])
    return SFTExample(marker, instruction, response)