import logging
import re
import threading
from typing import Any

from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from docneedle.config import settings
from docneedle.ingest.models import DocumentChunk

logger = logging.getLogger(__name__)

_CACHED_TOKENIZER: Any = None
_TOKENIZER_LOCK = threading.Lock()
_PAGE_HEADING_RE = re.compile(r"^(?:sayfa|page)\s+(\d+)$", re.IGNORECASE)


def _resolve_default_tokenizer() -> Any:
    """Resolve tokenizer with configurable max_tokens, caching locally to avoid online calls."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    with _TOKENIZER_LOCK:
        if _CACHED_TOKENIZER is not None:
            return _CACHED_TOKENIZER

        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        # Prioritize local offline cache first to eliminate online network latency
        try:
            tok = AutoTokenizer.from_pretrained(
                settings.chunker_tokenizer_model,
                local_files_only=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(settings.chunker_tokenizer_model)

        _CACHED_TOKENIZER = HuggingFaceTokenizer(
            tokenizer=tok, max_tokens=settings.chunker_max_tokens
        )
        return _CACHED_TOKENIZER


class DoclingChunker:
    """Splits DoclingDocument into semantic chunks preserving headings,
    tables, and page metadata.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        repeat_table_header: bool | None = None,
        merge_peers: bool | None = None,
    ) -> None:
        if tokenizer is None:
            tokenizer = _resolve_default_tokenizer()

        rep_headers = (
            settings.chunker_repeat_table_header
            if repeat_table_header is None
            else repeat_table_header
        )
        m_peers = settings.chunker_merge_peers if merge_peers is None else merge_peers

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            repeat_table_header=rep_headers,
            merge_peers=m_peers,
        )

    def chunk_document(
        self,
        document: DoclingDocument,
        doc_id: str,
        doc_name: str,
    ) -> list[DocumentChunk]:
        """Convert a DoclingDocument into discrete DocumentChunk objects."""
        raw_chunks = list(self.chunker.chunk(document))
        chunks: list[DocumentChunk] = []

        for idx, raw in enumerate(raw_chunks):
            # Extract heading hierarchy if available
            headings_list: list[str] = getattr(raw.meta, "headings", []) or []
            heading = " > ".join(headings_list) if headings_list else None

            # Detect if chunk belongs to a table or has page provenance
            is_table = False
            page_numbers: set[int] = set()

            doc_items = getattr(raw.meta, "doc_items", []) or []
            for item in doc_items:
                label_str = str(getattr(item, "label", "")).lower()
                if "table" in label_str:
                    is_table = True

                for prov in getattr(item, "prov", []) or []:
                    page_no = getattr(prov, "page_no", None)
                    if isinstance(page_no, int):
                        page_numbers.add(page_no)

            if not page_numbers and headings_list:
                for h in headings_list:
                    m_page = _PAGE_HEADING_RE.match(str(h).strip())
                    if m_page:
                        page_numbers.add(int(m_page.group(1)))

            primary_page = min(page_numbers) if page_numbers else None
            chunk_id = f"{doc_id}_{idx}"

            chunk_text = raw.text.strip()
            if heading and not chunk_text.startswith(heading):
                chunk_text = f"{heading}\n{chunk_text}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                chunk_index=idx,
                text=chunk_text,
                page_number=primary_page,
                heading=heading,
                is_table=is_table,
                metadata={
                    "headings": headings_list,
                    "pages": sorted(page_numbers) if page_numbers else [],
                },
            )
            chunks.append(chunk)

        return self._coalesce_small_chunks(chunks)

    @staticmethod
    def _should_merge(
        current: DocumentChunk,
        candidate: DocumentChunk,
        max_chunk_chars: int | None = None,
        min_split_chars: int | None = None,
        overflow_factor: float | None = None,
    ) -> bool:
        """Evaluate if two consecutive chunks are eligible for coalescence."""
        max_chars = settings.chunker_max_chunk_chars if max_chunk_chars is None else max_chunk_chars
        min_chars = settings.chunker_min_split_chars if min_split_chars is None else min_split_chars
        factor = settings.chunker_overflow_factor if overflow_factor is None else overflow_factor

        same_page = (
            current.page_number == candidate.page_number
            or current.page_number is None
            or candidate.page_number is None
        )
        if not same_page:
            return False

        combined_len = len(current.text) + len(candidate.text) + 2
        is_within_limit = combined_len <= max_chars
        is_either_small = len(current.text) < min_chars or len(candidate.text) < min_chars
        return (is_within_limit or is_either_small) and combined_len <= int(max_chars * factor)

    @staticmethod
    def _merge_chunk_pair(
        current: DocumentChunk,
        candidate: DocumentChunk,
        chunk_index: int,
    ) -> DocumentChunk:
        """Combine two chunks while merging text, headings, and provenance metadata."""
        cand_text = candidate.text
        if (
            candidate.heading
            and cand_text.startswith(candidate.heading)
            and current.heading
            and (current.heading == candidate.heading or candidate.heading in current.heading)
        ):
            cand_text = cand_text[len(candidate.heading) :].lstrip("\n\r ")

        combined_text = f"{current.text}\n\n{cand_text}".strip()
        merged_pages = sorted(
            set(current.metadata.get("pages", []) + candidate.metadata.get("pages", []))
        )
        merged_headings = list(
            dict.fromkeys(
                current.metadata.get("headings", []) + candidate.metadata.get("headings", [])
            )
        )
        return DocumentChunk(
            chunk_id=current.chunk_id,
            doc_id=current.doc_id,
            doc_name=current.doc_name,
            chunk_index=chunk_index,
            text=combined_text,
            page_number=current.page_number or candidate.page_number,
            heading=current.heading or candidate.heading,
            is_table=current.is_table or candidate.is_table,
            metadata={"headings": merged_headings, "pages": merged_pages},
        )

    @classmethod
    def _coalesce_small_chunks(
        cls,
        chunks: list[DocumentChunk],
        max_chunk_chars: int | None = None,
        min_split_chars: int | None = None,
    ) -> list[DocumentChunk]:
        """Coalesce consecutive small chunks from the same page to preserve
        holistic document context.
        """
        max_chars = settings.chunker_max_chunk_chars if max_chunk_chars is None else max_chunk_chars
        min_chars = settings.chunker_min_split_chars if min_split_chars is None else min_split_chars

        if len(chunks) <= 1:
            return chunks

        merged: list[DocumentChunk] = []
        current: DocumentChunk | None = None

        for c in chunks:
            if current is None:
                current = c.model_copy()
                continue

            if cls._should_merge(current, c, max_chars, min_chars):
                current = cls._merge_chunk_pair(current, c, chunk_index=len(merged))
            else:
                current.chunk_index = len(merged)
                merged.append(current)
                current = c.model_copy()

        if current is not None:
            current.chunk_index = len(merged)
            merged.append(current)

        return merged
