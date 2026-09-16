"""Domain taxonomy — the semantic seeds for the MoE router.

Every training document carries a domain tag. The router guide-loss (see
src/model/moe.py) uses these to teach each expert its specialty during the
annealed curriculum, and the demo heatmap renders per-domain activation.

The eight constants MUST stay aligned with EXPERT_DOMAINS in src/model/moe.py
(an assert below enforces it on first import).
"""

from __future__ import annotations

from enum import IntEnum

from src.model.moe import EXPERT_DOMAINS


class Domain(IntEnum):
    MATH = 0
    CODE = 1
    LOGIC = 2
    COMMONSENSE = 3
    SQL = 4
    JSON = 5
    GENERAL_1 = 6
    GENERAL_2 = 7


DOMAIN_NAMES: tuple[str, ...] = tuple(str(d.name).lower() for d in Domain)

assert list(DOMAIN_NAMES) == EXPERT_DOMAINS, (
    "src/data/domains.py and src/model/moe.py disagree on the domain taxonomy"
)

# Source dataset -> domain. Pretrain corpora are mapped through here so the
# same pipeline decides both what the model reads and which expert it should
# specialize toward.
_SOURCE_DOMAIN: dict[str, int] = {
    "fineweb": Domain.GENERAL_1,
    "fineweb_edu": Domain.GENERAL_2,
    "gsm8k": Domain.MATH,
    "the_pile_math": Domain.MATH,
    "code": Domain.CODE,
    "cwsmse_commonsense": Domain.COMMONSENSE,
    "arc_easy": Domain.COMMONSENSE,
    "piqa": Domain.COMMONSENSE,
    "hellaswag": Domain.COMMONSENSE,
    "winogrande": Domain.COMMONSENSE,
    "spider": Domain.SQL,
    "sql": Domain.SQL,
    "json": Domain.JSON,
    "logicqa": Domain.LOGIC,
    "cot": Domain.LOGIC,
}


def get_domain_id(name: str) -> int:
    """Resolve a domain label or source name to a Domain id."""
    key = str(name).lower().strip()
    if key in _SOURCE_DOMAIN:
        return int(_SOURCE_DOMAIN[key])
    if key in DOMAIN_NAMES:
        return int(Domain[key.upper()])
    fallback = int(Domain.GENERAL_2)
    raise KeyError(
        f"unknown domain/source {name!r}; known: {sorted(set(_SOURCE_DOMAIN) | set(DOMAIN_NAMES))}"
    )


class DomainTagger:
    """Wraps an iterator of (source, text) into (domain_id, text)."""

    def __init__(self) -> None:
        pass

    def tag(self, source: str, text: str) -> tuple[int, str]:
        return get_domain_id(source), text

    def tag_stream(self, stream) -> iter:
        for source, text in stream:
            yield self.tag(source, text)

    @staticmethod
    def reset() -> None:
        pass