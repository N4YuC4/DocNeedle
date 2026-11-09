"""Chunk text matching and evidence resolution for extraction results."""

import re
from typing import Any

from docneedle.extraction.verifier import normalize_for_match
from docneedle.retrieval.bm25 import normalize_search_text
from docneedle.retrieval.models import SearchResult

DEFAULT_SNIPPET_MAX_CHARS: int = 350
SHORT_QUERY_CHAR_THRESHOLD: int = 3
# Backward-compatibility alias
SHORT_QUERY_TOKEN_THRESHOLD: int = SHORT_QUERY_CHAR_THRESHOLD
_WORD_PATTERN = re.compile(r"\w+")


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


def _find_best_supporting_match(
    raw_value: Any,
    matches: list[SearchResult],
    is_cover_query: bool = False,
) -> SearchResult:
    """Find the specific chunk that actually contains or best supports the extracted value."""
    if not matches:
        raise ValueError("Matches list cannot be empty")

    if is_cover_query:
        p1 = next((m for m in matches if m.chunk.page_number == 1), None)
        if p1 is not None:
            return p1

    if isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, (str, int, float)):
                        s = str(val).strip()
                        if len(s) >= 3:
                            m = _match_exact_substring(normalize_for_match(s), matches)
                            if m:
                                return m
            elif isinstance(item, (str, int, float)):
                s = str(item).strip()
                if len(s) >= 3:
                    m = _match_exact_substring(normalize_for_match(s), matches)
                    if m:
                        return m
        return matches[0]

    clean_val = str(raw_value).strip()
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
