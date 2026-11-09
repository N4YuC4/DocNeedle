"""Prompt Lookup Decoding speculative model setup."""

import logging
from pathlib import Path
from typing import Any

from docneedle.config import settings

logger = logging.getLogger(__name__)


def _create_speculative_draft_model(model_path: Path, llama_cls: Any) -> Any:
    """Inspect GGUF vocabulary size and build a Prompt Lookup Decoding draft model when eligible."""
    use_speculative = settings.llm_speculative_enabled and (
        settings.gpu_enabled or settings.llm_speculative_cpu_enabled
    )
    if not use_speculative:
        return None

    probe_llm = llama_cls(
        model_path=str(model_path),
        vocab_only=True,
        verbose=False,
    )
    vocab_size = probe_llm.n_vocab()
    del probe_llm

    if vocab_size > settings.llm_speculative_max_vocab_size:
        logger.info(
            "Skipping speculative decoding because model vocab_size (%d) exceeds threshold (%d).",
            vocab_size,
            settings.llm_speculative_max_vocab_size,
        )
        return None

    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

    num_pred = (
        settings.llm_speculative_num_pred_tokens
        if not settings.gpu_enabled
        else max(10, settings.llm_speculative_num_pred_tokens)
    )
    logger.info(
        "Prompt Lookup Decoding enabled: max_ngram_size=%d, num_pred_tokens=%d",
        settings.llm_speculative_max_ngram_size,
        num_pred,
    )
    return LlamaPromptLookupDecoding(
        max_ngram_size=settings.llm_speculative_max_ngram_size,
        num_pred_tokens=num_pred,
    )
