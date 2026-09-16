"""SFT (Supervised Fine-Tuning) structured-output corpus and formatting."""
from .sft_format import SFTExample, format_sft_tokens, parse_sft_document, DOMAIN_MARKERS
from .sft_corpus import (
    gen_json_examples, gen_sql_examples, gen_cot_examples,
    build_sft_shards, GENERATORS,
)
