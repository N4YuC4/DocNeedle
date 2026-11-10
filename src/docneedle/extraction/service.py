import inspect
import logging
import re
import time
from typing import Any

from docneedle.config import settings
from docneedle.extraction.collision import resolve_cross_field_collisions
from docneedle.extraction.matcher import (
    DEFAULT_SNIPPET_MAX_CHARS,
    SHORT_QUERY_TOKEN_THRESHOLD,
    _find_best_supporting_match,
    _is_visual_chunk,
    _match_exact_substring,
    _match_folded_substring,
    _match_token_overlap,
    _truncate_snippet,
)
from docneedle.extraction.schemas import (
    ExtractionEvidence,
    ExtractionItemQuery,
    ExtractionItemResult,
    ExtractionResponse,
)
from docneedle.extraction.verifier import ConfidenceCalculator, normalize_for_match
from docneedle.llm.client import LLMClient
from docneedle.retrieval.models import SearchResult
from docneedle.retrieval.store import SessionVectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SNIPPET_MAX_CHARS",
    "ExtractionService",
    "SHORT_QUERY_TOKEN_THRESHOLD",
    "_build_retrieval_query",
    "_clean_field_description",
    "_find_best_supporting_match",
    "_log_rag_context",
    "_match_exact_substring",
    "_match_folded_substring",
    "_match_token_overlap",
    "_normalize_for_match",
    "_truncate_snippet",
]


# Constants
RAG_LOG_HEADER_SEP_LEN: int = 25
RAG_LOG_TOTAL_SEP_LEN: int = 75

_PLACEHOLDER_KEY_PATTERN = re.compile(r"^(madde|item|field|key|q)\s*\d+$", re.IGNORECASE)
_COVER_PAGE_PATTERN = re.compile(
    r"\b(?:cover[\s_-]*page|cover[\s_-]*sheet|first[\s_-]*page|document[\s_-]*header|kapak\s*sayfas[ıi]|ilk\s*sayfa)\b",
    re.IGNORECASE,
)

# Alias for backward compatibility with private imports and tests
_normalize_for_match = normalize_for_match


def _clean_field_description(desc: str | None) -> str | None:
    """Strip benchmark meta-rules, null-patches, and examples from field description."""
    if not desc or not desc.strip():
        return None
    primary = desc.strip()
    for marker in (
        "Examples:",
        "Example:",
        "Do NOT",
        "Do not",
        "[PATCH",
        "Format requirement:",
        "NULL semantics:",
        "Location hint:",
    ):
        if marker in primary:
            primary = primary.split(marker)[0].strip()
    return primary or desc[:200]


def _format_chunk_context(chunk: Any) -> str:
    """Format chunk text with clear, compact document and page boundary markers."""
    doc_header = f"[Doküman: {chunk.doc_name}" + (
        f" (Sayfa {chunk.page_number})]" if chunk.page_number else "]"
    )
    raw_text = chunk.text.strip()
    if raw_text.startswith("[Doküman:"):
        return raw_text
    return f"{doc_header}\n{raw_text}"


def _build_retrieval_query(key: str, description: str | None) -> str:
    """Construct a clean, natural search query from the item key and description."""
    clean_key = key.replace("_", " ").strip()
    if description and description.strip():
        primary_desc = _clean_field_description(description) or description.strip()
        is_placeholder = bool(_PLACEHOLDER_KEY_PATTERN.match(clean_key))
        if not is_placeholder and clean_key.lower() not in primary_desc.lower():
            return f"{primary_desc} ({clean_key})"
        return primary_desc
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
        is_cover_query: bool = False,
        candidate_chunks: list[Any] | None = None,
    ) -> tuple[ExtractionEvidence, float]:
        """Resolve the best supporting chunk, compute confidence score, and build evidence."""
        best_match = _find_best_supporting_match(
            raw_val,
            matches,
            is_cover_query=is_cover_query,
            candidate_chunks=candidate_chunks,
        )
        confidence = ConfidenceCalculator.calculate(
            found=True,
            value=raw_val,
            retrieval_score=best_match.score,
            target_chunk_text=best_match.chunk.text,
            expected_type=expected_type,
            has_image=bool(best_match.chunk.image_bytes),
        )
        is_visual_only = _is_visual_chunk(best_match.chunk)
        if is_visual_only:
            doc_ref = best_match.chunk.doc_name
            pg = f" (Sayfa {best_match.chunk.page_number})" if best_match.chunk.page_number else ""
            val_repr = str(raw_val)[:60]
            snippet_text = f"Görsel Doküman {doc_ref}{pg} doğrudan incelendi (Değer: '{val_repr}')"
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
        response = await self.extract_items(
            [item],
            store,
            top_k=top_k,
            enable_reranker=enable_reranker,
        )
        return response.results[0] if response.results else self._build_empty_item_result(item.key)

    @staticmethod
    def _collect_field_matches(
        items: list[ExtractionItemQuery],
        store: SessionVectorStore,
        top_k: int,
        enable_reranker: bool | None,
    ) -> tuple[dict[str, list[SearchResult]], list[Any], str]:
        """Retrieve candidate RAG chunks across all requested schema fields and build context."""
        item_matches: dict[str, list[SearchResult]] = {}
        all_candidate_chunks: dict[str, Any] = {}

        for item in items:
            is_table = (item.type or "").lower() in ("list", "array") or (
                item.items_schema is not None
            )
            item_k = max(top_k, 8) if is_table else top_k
            search_query = _build_retrieval_query(item.key, item.description)
            matches = store.search(search_query, top_k=item_k, enable_reranker=enable_reranker)

            is_cover = bool(
                _COVER_PAGE_PATTERN.search(item.description or "")
                or _COVER_PAGE_PATTERN.search(item.key.replace("_", " "))
            )
            if is_cover:
                p1_match = next((m for m in matches if m.chunk.page_number == 1), None)
                if p1_match:
                    matches.remove(p1_match)
                    matches.insert(0, p1_match)
                else:
                    page_1_chunks = [c for c in store.chunks if c.page_number == 1]
                    if page_1_chunks:
                        matches.insert(
                            0,
                            SearchResult(chunk=page_1_chunks[0], score=1.0, rrf_score=1.0),
                        )

            item_matches[item.key] = matches
            for m in matches:
                all_candidate_chunks[m.chunk.chunk_id] = m.chunk

            _log_rag_context(item.key, item.description, search_query, matches)

        if not all_candidate_chunks and store.chunks:
            for c in store.chunks[:top_k]:
                all_candidate_chunks[c.chunk_id] = c

        sorted_chunks = sorted(
            all_candidate_chunks.values(),
            key=lambda c: (c.doc_name, c.page_number or 0, c.chunk_index),
        )
        context_str = settings.extraction_context_delimiter.join(
            _format_chunk_context(c) for c in sorted_chunks
        )
        return item_matches, sorted_chunks, context_str

    @staticmethod
    def _collect_visual_page_images(
        store: SessionVectorStore,
        sorted_chunks: list[Any],
    ) -> list[bytes]:
        """Extract high-resolution visual page images referenced by retrieved context chunks."""
        all_store_visual_pages: dict[tuple[str, int], bytes] = {}
        for c in store.chunks:
            if c.page_number is not None and c.image_bytes is not None:
                doc_key = c.doc_name or c.doc_id
                key = (doc_key, c.page_number)
                if key not in all_store_visual_pages:
                    all_store_visual_pages[key] = c.image_bytes

        image_bytes_list: list[bytes] = []
        if not all_store_visual_pages:
            return image_bytes_list

        referenced_keys: list[tuple[str, int]] = []
        scanned_keys: list[tuple[str, int]] = []
        seen_keys: set[tuple[str, int]] = set()

        for c in sorted_chunks:
            if c.page_number is not None:
                doc_key = c.doc_name or c.doc_id
                k_tuple = (doc_key, c.page_number)
                if k_tuple in all_store_visual_pages and k_tuple not in seen_keys:
                    seen_keys.add(k_tuple)
                    referenced_keys.append(k_tuple)
                    is_scanned = bool(
                        "[Görsel" in c.text
                        or len(c.text.strip()) < settings.pdf_embedded_text_min_chars
                    )
                    if is_scanned:
                        scanned_keys.append(k_tuple)

        if not referenced_keys:
            first_k = next(iter(all_store_visual_pages.keys()))
            referenced_keys.append(first_k)

        # Preserve visual page images referenced by context chunks (up to 5 pages)
        selected_keys: list[tuple[str, int]] = []
        if scanned_keys:
            selected_keys = scanned_keys[:5]
            for k in referenced_keys:
                if len(selected_keys) >= 5:
                    break
                if k not in selected_keys:
                    selected_keys.append(k)
        else:
            selected_keys = referenced_keys[:5]

        for k_tuple in selected_keys:
            image_bytes_list.append(all_store_visual_pages[k_tuple])

        return image_bytes_list

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

        if not items:
            return ExtractionResponse(
                status="completed",
                processing_time_ms=0.0,
                total_items=0,
                found_items=0,
                results=[],
            )

        if not store.chunks:
            return ExtractionResponse(
                status="completed",
                processing_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                total_items=len(items),
                found_items=0,
                results=[self._build_empty_item_result(it.key) for it in items],
            )

        t_ret_start = time.perf_counter()
        item_matches, all_candidate_chunks, _ = self._collect_field_matches(
            items=items,
            store=store,
            top_k=k,
            enable_reranker=enable_reranker,
        )
        t_ret_elapsed = time.perf_counter() - t_ret_start

        batch_size = max(1, settings.extraction_schema_batch_size)
        total_items_count = len(items)
        processed_idx = 0

        t_llm_start = time.perf_counter()
        for b_start in range(0, total_items_count, batch_size):
            batch_items = items[b_start : b_start + batch_size]
            batch_chunks_map: dict[str, Any] = {}
            for it in batch_items:
                for m in item_matches.get(it.key, []):
                    batch_chunks_map[m.chunk.chunk_id] = m.chunk

            if not batch_chunks_map and store.chunks:
                for c in store.chunks[:k]:
                    batch_chunks_map[c.chunk_id] = c

            sorted_batch_chunks = sorted(
                batch_chunks_map.values(),
                key=lambda c: (c.doc_name, c.page_number or 0, c.chunk_index),
            )
            batch_context_str = settings.extraction_context_delimiter.join(
                _format_chunk_context(c) for c in sorted_batch_chunks
            )
            batch_images = self._collect_visual_page_images(store, sorted_batch_chunks)

            fields_payload = [
                {
                    "key": it.key,
                    "description": it.description,
                    "type": it.type,
                    "items_schema": it.items_schema,
                }
                for it in batch_items
            ]

            extracted_dict = await self.llm_client.extract_schema(
                fields=fields_payload,
                context=batch_context_str,
                image_bytes_list=batch_images if batch_images else None,
            )

            for item in batch_items:
                processed_idx += 1
                it_res = extracted_dict.get(item.key, {"found": False, "value": None})
                is_found = bool(it_res.get("found", False))
                raw_val = it_res.get("value")
                thinking = it_res.get("thinking")

                evidence: ExtractionEvidence | None = None
                confidence: float = 0.0

                if is_found and raw_val is not None:
                    matches = item_matches.get(item.key, [])
                    if not matches and sorted_batch_chunks:
                        matches = [
                            SearchResult(chunk=c, score=1.0) for c in sorted_batch_chunks[:k]
                        ]
                    is_cover = bool(
                        _COVER_PAGE_PATTERN.search(item.description or "")
                        or _COVER_PAGE_PATTERN.search(item.key.replace("_", " "))
                    )
                    if matches:
                        evidence, confidence = self._create_evidence_and_confidence(
                            raw_val=raw_val,
                            matches=matches,
                            expected_type=item.type,
                            is_cover_query=is_cover,
                            candidate_chunks=sorted_batch_chunks,
                        )
                    else:
                        confidence = settings.confidence_found_score

                result = ExtractionItemResult(
                    key=item.key,
                    found=is_found and raw_val is not None,
                    value=raw_val if (is_found and raw_val is not None) else None,
                    confidence=confidence,
                    evidence=evidence,
                    thinking=thinking,
                )
                results.append(result)

                if on_item_extracted is not None:
                    try:
                        cb_res = on_item_extracted(result, processed_idx, total_items_count)
                        if inspect.isawaitable(cb_res):
                            await cb_res
                    except Exception as cb_err:
                        logger.debug(f"on_item_extracted callback error: {cb_err}")
        t_llm_elapsed = time.perf_counter() - t_llm_start

        # Deterministic cross-field collision resolution & label echo rejection
        t_col_start = time.perf_counter()
        results = resolve_cross_field_collisions(results, items, store)
        t_col_elapsed = time.perf_counter() - t_col_start

        found_count = sum(1 for r in results if r.found)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "--> [EXTRACTION TIMING] Retrieval: %.2fs | "
            "LLM Inference: %.2fs | Collision: %.2fs | Total: %.2fs",
            t_ret_elapsed,
            t_llm_elapsed,
            t_col_elapsed,
            elapsed_ms / 1000.0,
        )
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
