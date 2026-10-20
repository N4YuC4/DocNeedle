"""Unit tests for speculative decoding (Prompt Lookup Decoding) integration."""

import numpy as np
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from docneedle.config import settings


def test_prompt_lookup_decoding_ngram_match():
    """Verify that Prompt Lookup Decoding identifies matching n-gram continuation."""
    lookup = LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=3)

    # Sequence where [10, 20] occurs earlier, followed by [30, 40, 50]
    input_ids = np.array([10, 20, 30, 40, 50, 99, 10, 20], dtype=np.intc)
    candidates = lookup(input_ids)

    assert len(candidates) == 3
    assert candidates.tolist() == [30, 40, 50]


def test_prompt_lookup_decoding_no_match():
    """Verify that Prompt Lookup Decoding returns empty array when no n-gram match exists."""
    lookup = LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=3)

    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.intc)
    candidates = lookup(input_ids)

    assert len(candidates) == 0


def test_speculative_settings():
    """Verify default speculative settings in configuration."""
    assert settings.llm_speculative_enabled is True
    assert settings.llm_speculative_num_pred_tokens >= 2
    assert settings.llm_speculative_max_ngram_size >= 2
