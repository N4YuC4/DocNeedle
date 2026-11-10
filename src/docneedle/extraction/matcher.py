"""Chunk text matching and evidence resolution for extraction results."""

import re
from typing import Any

from docneedle.extraction.verifier import normalize_for_match
from docneedle.retrieval.bm25 import normalize_search_text
from docneedle.retrieval.models import SearchResult

DEFAULT_SNIPPET_MAX_CHARS: int = 350
SHORT_QUERY_CHAR_THRESHOLD: int = 3
SHORT_QUERY_TOKEN_THRESHOLD: int = SHORT_QUERY_CHAR_THRESHOLD
_WORD_PATTERN = re.compile(r"\w+")


def _is_visual_chunk(chunk: Any) -> bool:
    """Check if a chunk represents a visual-only image or scanned page without digital text."""
    if getattr(chunk, "image_bytes", None) is None:
        return False
    txt = chunk.text.strip() if getattr(chunk, "text", None) else ""
    return "[Görsel" in txt or not txt


def _truncate_snippet(text: str, max_chars: int = DEFAULT_SNIPPET_MAX_CHARS) -> str | None:
    """Format and truncate snippet text cleanly on word boundaries."""
    snippet = text.strip()
    if not snippet:
        return None
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    return snippet or None


def _match_exact_substring(norm_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with exact Turkish case-aware match, prioritizing the earliest page."""
    short_pattern = (
        re.compile(r"\b" + re.escape(norm_val) + r"\b")
        if len(norm_val) <= SHORT_QUERY_TOKEN_THRESHOLD
        else None
    )
    candidates: list[SearchResult] = []
    for m in matches:
        chunk_norm = normalize_for_match(m.chunk.text)
        if short_pattern:
            if short_pattern.search(chunk_norm):
                candidates.append(m)
        elif norm_val in chunk_norm:
            candidates.append(m)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda m: (
            m.chunk.page_number if m.chunk.page_number is not None else 9999,
            m.chunk.chunk_index,
        ),
    )


def _match_folded_substring(folded_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with accent-folded text match, prioritizing the earliest page."""
    if not folded_val:
        return None
    short_pattern = (
        re.compile(r"\b" + re.escape(folded_val) + r"\b")
        if len(folded_val) <= SHORT_QUERY_TOKEN_THRESHOLD
        else None
    )
    candidates: list[SearchResult] = []
    for m in matches:
        chunk_folded = normalize_search_text(m.chunk.text)
        if short_pattern:
            if short_pattern.search(chunk_folded):
                candidates.append(m)
        elif folded_val in chunk_folded:
            candidates.append(m)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda m: (
            m.chunk.page_number if m.chunk.page_number is not None else 9999,
            m.chunk.chunk_index,
        ),
    )


def _match_iso_date(val_str: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk containing the date in numeric or textual format across any language."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", val_str.strip())
    if not m:
        return None
    y, m_num, d = int(m.group(1)), int(m.group(2)), int(m.group(3))

    date_patterns = [
        re.compile(rf"\b{d:02d}[./-]{m_num:02d}[./-]{y}\b"),
        re.compile(rf"\b{d}[./-]{m_num}[./-]{y}\b"),
        re.compile(rf"\b{y}[./-]{m_num:02d}[./-]{d:02d}\b"),
        re.compile(rf"\b{y}[./-]{m_num}[./-]{d}\b"),
        re.compile(rf"\b{m_num:02d}[./-]{d:02d}[./-]{y}\b"),
        re.compile(rf"\b{d}\.?\s+(?:de\s+)?[^\W\d_]+(?:\s+de)?\s+{y}\b", re.IGNORECASE),
        re.compile(rf"\b[^\W\d_]+\s+{d},?\s+{y}\b", re.IGNORECASE),
    ]

    for match in matches:
        text = match.chunk.text
        if any(p.search(text) for p in date_patterns):
            return match
    return None


def _match_token_overlap(folded_val: str, matches: list[SearchResult]) -> SearchResult | None:
    """Find chunk with highest keyword token overlap for descriptive multi-word phrases."""
    words = [w for w in _WORD_PATTERN.findall(folded_val) if len(w) > 2 and not w.isdigit()]
    if len(words) < 2:
        return None

    best_match: SearchResult | None = None
    max_overlap = 0
    total_words = len(words)

    for m in matches:
        chunk_folded = normalize_search_text(m.chunk.text)
        overlap = sum(1 for w in words if w in chunk_folded)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = m

    if best_match is None or (max_overlap / total_words) < 0.70:
        return None

    return best_match


def _find_best_supporting_match(
    raw_value: Any,
    matches: list[SearchResult],
    is_cover_query: bool = False,
    candidate_chunks: list[Any] | None = None,
) -> SearchResult:
    """Find the specific chunk that actually contains or best supports the extracted value."""
    if not matches:
        if candidate_chunks:
            matches = [SearchResult(chunk=c, score=1.0) for c in candidate_chunks]
        else:
            raise ValueError("Matches list cannot be empty")

    if is_cover_query:
        p1 = next((m for m in matches if m.chunk.page_number == 1), None)
        if p1 is not None:
            return p1

    pool = list(matches)
    if candidate_chunks:
        existing_ids = {m.chunk.chunk_id for m in matches}
        for c in candidate_chunks:
            if c.chunk_id not in existing_ids:
                pool.append(SearchResult(chunk=c, score=0.5))

    def _search_scalar(val_str: str) -> SearchResult | None:
        s = val_str.strip()
        if not s or len(s) < 2:
            return None
        norm_val = normalize_for_match(s)
        if not norm_val:
            return None
        folded_val = normalize_search_text(s)

        matched = _match_exact_substring(norm_val, pool) or _match_folded_substring(
            folded_val, pool
        )
        if matched:
            return matched

        matched_date = _match_iso_date(s, pool)
        if matched_date:
            return matched_date

        return _match_token_overlap(folded_val, pool)

    if isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, (str, int, float)):
                        matched = _search_scalar(str(val))
                        if matched:
                            return matched
            elif isinstance(item, (str, int, float)):
                matched = _search_scalar(str(item))
                if matched:
                    return matched

        visual_match = next((m for m in pool if _is_visual_chunk(m.chunk)), None)
        if visual_match:
            return visual_match
        return matches[0]

    clean_val = str(raw_value).strip()
    matched = _search_scalar(clean_val)
    if matched:
        return matched

    visual_match = next((m for m in pool if _is_visual_chunk(m.chunk)), None)
    if visual_match:
        return visual_match

    return matches[0]
