import inspect
import logging
import re
import time
from typing import Any

from docneedle.config import settings
from docneedle.extraction.schemas import (
    ExtractionEvidence,
    ExtractionItemQuery,
    ExtractionItemResult,
    ExtractionResponse,
)
from docneedle.extraction.verifier import ConfidenceCalculator, normalize_for_match
from docneedle.llm.client import LLMClient
from docneedle.retrieval.bm25 import normalize_search_text
from docneedle.retrieval.models import SearchResult
from docneedle.retrieval.store import SessionVectorStore

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SNIPPET_MAX_CHARS: int = 350
SHORT_QUERY_TOKEN_THRESHOLD: int = 3
RAG_LOG_HEADER_SEP_LEN: int = 25
RAG_LOG_TOTAL_SEP_LEN: int = 75

_PLACEHOLDER_KEY_PATTERN = re.compile(r"^(madde|item|field|key|q)\s*\d+$", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"\w+")

# Alias for backward compatibility with private imports and tests
_normalize_for_match = normalize_for_match


def _build_retrieval_query(key: str, description: str | None) -> str:
    """Construct a clean, natural search query from the item key and description."""
    clean_key = key.replace("_", " ").strip()
    if description and description.strip():
        desc = description.strip()
        is_placeholder = bool(_PLACEHOLDER_KEY_PATTERN.match(clean_key))
        if not is_placeholder and clean_key.lower() not in desc.lower():
            return f"{desc} ({clean_key})"
        return desc
    return clean_key


def _log_rag_context(
    item_key: str,
    item_desc: str | None,
    search_query: str,
    matches: list[SearchResult],
) -> None:
    """Log retrieved RAG context blocks for auditing and transparency."""
    desc_str = f" ({item_desc})" if item_desc else ""
    sep = "=" * RAG_LOG_HEADER_SEP_LEN
    lines = [
        f"\n{sep} [RAG CHUNKS FOR: '{item_key}'{desc_str}] {sep}",
        f"Retrieval Query: {search_query}",
    ]
    for idx, match in enumerate(matches, 1):
        meta_info = f"Doc: {match.chunk.doc_name}"
        if match.chunk.page_number is not None:
            meta_info += f" | Page: {match.chunk.page_number}"
        score_lbl = (
            f"Rerank: {match.rerank_score:.4f}"
            if match.rerank_score is not None
            else f"Score: {match.score:.4f}"
        )
        lines.append(f"--- Chunk #{idx} ({score_lbl} | {meta_info}) ---")
        lines.append(match.chunk.text.strip())
    lines.append(f"{'=' * RAG_LOG_TOTAL_SEP_LEN}\n")
    logger.info("\n".join(lines))


def _truncate_snippet(text: str, max_chars: int = DEFAULT_SNIPPET_MAX_CHARS) -> str | None:
    """Format and truncate snippet text cleanly on word boundaries."""
    snippet = text.strip()
    if not snippet:
        return None
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    return snippet or None


def _match_exact_substring(norm_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with exact Turkish case-aware match, enforcing word boundary for short tokens."""
    short_pattern = (
        re.compile(r"\b" + re.escape(norm_val) + r"\b")
        if len(norm_val) <= SHORT_QUERY_TOKEN_THRESHOLD
        else None
    )
    for m in matches:
        chunk_norm = normalize_for_match(m.chunk.text)
        if short_pattern:
            if short_pattern.search(chunk_norm):
                return m
        elif norm_val in chunk_norm:
            return m
    return None


def _match_folded_substring(folded_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with accent-folded text match, enforcing word boundary for short tokens."""
    if not folded_val:
        return None
    short_pattern = (
        re.compile(r"\b" + re.escape(folded_val) + r"\b")
        if len(folded_val) <= SHORT_QUERY_TOKEN_THRESHOLD
        else None
    )
    for m in matches:
        chunk_folded = normalize_search_text(m.chunk.text)
        if short_pattern:
            if short_pattern.search(chunk_folded):
                return m
        elif folded_val in chunk_folded:
            return m
    return None


def _match_token_overlap(folded_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with highest keyword token overlap."""
    words = [w for w in _WORD_PATTERN.findall(folded_val) if len(w) > 1]
    if not words:
        return None

    best_match: SearchResult | None = None
    max_overlap = 0
    for m in matches:
        chunk_folded = normalize_search_text(m.chunk.text)
        overlap = sum(1 for w in words if w in chunk_folded)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = m
    return best_match if max_overlap > 0 else None


def _find_best_supporting_match(raw_value: str, matches: list[SearchResult]) -> SearchResult:
    """Find the specific chunk that actually contains or best supports the extracted value."""
    if not matches:
        raise ValueError("Matches list cannot be empty")

    clean_val = raw_value.strip()
    norm_val = normalize_for_match(clean_val)
    if not norm_val:
        return matches[0]

    folded_val = normalize_search_text(clean_val)

    return (
        _match_exact_substring(norm_val, matches)
        or _match_folded_substring(folded_val, matches)
        or _match_token_overlap(folded_val, matches)
        or matches[0]
    )


class ExtractionService:
    """Orchestrates information extraction across documents stored in SessionVectorStore."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()

    @staticmethod
    def _build_empty_item_result(item_key: str) -> ExtractionItemResult:
        """Create a default result payload when an item is not found or has no RAG matches."""
        return ExtractionItemResult(
            key=item_key,
            found=False,
            value=None,
            confidence=0.0,
            evidence=None,
            thinking=None,
        )

    @staticmethod
    def _create_evidence_and_confidence(
        raw_val: Any,
        matches: list[SearchResult],
        expected_type: str | None,
    ) -> tuple[ExtractionEvidence, float]:
        """Resolve the best supporting chunk, compute confidence score, and build evidence."""
        best_match = _find_best_supporting_match(str(raw_val), matches)
        confidence = ConfidenceCalculator.calculate(
            found=True,
            value=raw_val,
            retrieval_score=best_match.score,
            target_chunk_text=best_match.chunk.text,
            expected_type=expected_type,
        )
        if best_match.chunk.image_bytes is not None and (
            "[Görsel Doküman" in best_match.chunk.text
            or "[Görsel Belge Sayfası" in best_match.chunk.text
        ):
            doc_ref = best_match.chunk.doc_name
            pg = f" (Sayfa {best_match.chunk.page_number})" if best_match.chunk.page_number else ""
            snippet_text = f"Görsel Doküman {doc_ref}{pg} doğrudan incelendi (Değer: '{raw_val}')"
        else:
            snippet_text = _truncate_snippet(best_match.chunk.text)
        evidence = ExtractionEvidence(
            document=best_match.chunk.doc_name,
            page=best_match.chunk.page_number,
            snippet=snippet_text,
        )
        return evidence, confidence

    async def _extract_single_item(
        self,
        item: ExtractionItemQuery,
        store: SessionVectorStore,
        top_k: int | None = None,
        enable_reranker: bool | None = None,
    ) -> ExtractionItemResult:
        """Retrieve relevant context and extract a single schema item using LLM inference."""
        k = settings.extraction_top_k if top_k is None else top_k
        search_query = _build_retrieval_query(item.key, item.description)
        matches = store.search(search_query, top_k=k, enable_reranker=enable_reranker)

        if not matches:
            return self._build_empty_item_result(item.key)

        # Sort retrieved chunks canonically for deterministic prompt prefixes
        # (maximizes KV-cache reuse across subsequent queries in LLM)
        canonical_matches = sorted(
            matches,
            key=lambda m: (m.chunk.doc_name, m.chunk.page_number or 0, m.chunk.chunk_index),
        )
        context_blocks = [m.chunk.text.strip() for m in canonical_matches]
        context_str = settings.extraction_context_delimiter.join(context_blocks)

        _log_rag_context(item.key, item.description, search_query, matches)

        # Select at most 2 most relevant visual page chunks to prevent LLM context flooding
        top_visual_ids = {
            m.chunk.chunk_id for m in [m for m in matches if m.chunk.image_bytes is not None][:2]
        }
        image_bytes_list: list[bytes] = [
            m.chunk.image_bytes
            for m in canonical_matches
            if m.chunk.image_bytes is not None and m.chunk.chunk_id in top_visual_ids
        ]

        llm_result = await self.llm_client.extract_item(
            query_key=item.key,
            query_description=item.description,
            context=context_str,
            image_bytes_list=image_bytes_list if image_bytes_list else None,
            expected_type=item.type,
        )

        is_found = bool(llm_result.get("found", False))
        raw_val = llm_result.get("value")
        thinking_trace = llm_result.get("thinking")

        evidence: ExtractionEvidence | None = None
        confidence: float = 0.0

        if is_found and raw_val is not None:
            evidence, confidence = self._create_evidence_and_confidence(
                raw_val=raw_val,
                matches=matches,
                expected_type=item.type,
            )

        return ExtractionItemResult(
            key=item.key,
            found=is_found and raw_val is not None,
            value=str(raw_val) if raw_val is not None else None,
            confidence=confidence,
            evidence=evidence,
            thinking=thinking_trace,
        )

    async def extract_items(
        self,
        items: list[ExtractionItemQuery],
        store: SessionVectorStore,
        top_k: int | None = None,
        enable_reranker: bool | None = None,
        on_item_extracted: Any | None = None,
    ) -> ExtractionResponse:
        """Process extraction queries by retrieving RAG chunks and querying the LLM."""
        start_time = time.perf_counter()
        results: list[ExtractionItemResult] = []
        k = settings.extraction_top_k if top_k is None else top_k

        logger.info(f"===> [EXTRACTION SESSION] Processing {len(items)} target field query(s)...")

        for idx, item in enumerate(items, 1):
            t_item_start = time.perf_counter()
            logger.info(
                f"--> [{idx}/{len(items)}] Query: key='{item.key}' | "
                f"desc='{item.description or item.key}' | type='{item.type or 'auto'}'"
            )
            result = await self._extract_single_item(
                item, store, top_k=k, enable_reranker=enable_reranker
            )
            results.append(result)
            elapsed_item = (time.perf_counter() - t_item_start) * 1000.0
            val_display = f"'{result.value}'" if result.found else "[NOT FOUND]"
            doc_display = f"from '{result.evidence.document}'" if result.evidence else ""
            logger.info(
                f"    [{idx}/{len(items)} COMPLETE in {elapsed_item:.0f}ms] "
                f"{item.key} = {val_display} (conf: {result.confidence:.2f}) {doc_display}"
            )
            if on_item_extracted is not None:
                try:
                    cb_res = on_item_extracted(result, idx, len(items))
                    if inspect.isawaitable(cb_res):
                        await cb_res
                except Exception as cb_err:
                    logger.debug(f"on_item_extracted callback error: {cb_err}")

        found_count = sum(1 for r in results if r.found)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"===> [SESSION COMPLETE] Found {found_count}/{len(items)} items in {elapsed_ms:.1f}ms"
        )

        return ExtractionResponse(
            status="completed",
            processing_time_ms=round(elapsed_ms, 2),
            total_items=len(items),
            found_items=found_count,
            results=results,
        )
