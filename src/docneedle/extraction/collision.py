"""Deterministic cross-field collision resolver.

Resolves spatial hallucinations where the LLM repeats a single salient number/entity
across multiple unpopulated fields by checking physical document occurrences on candidate
pages and assigning the value only to the field with highest context affinity.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from docneedle.extraction.schemas import ExtractionItemQuery, ExtractionItemResult

if TYPE_CHECKING:
    from docneedle.retrieval.store import SessionVectorStore

_BOOLEAN_TERMS = frozenset({"true", "false", "yes", "no", "evet", "hayır", "hayir", "t", "f"})
_EMPTY_TERMS = frozenset(
    {
        "",
        "none",
        "null",
        "n/a",
        "na",
        "-",
        "--",
        "—",
        "()",
        "( )",
        "nil",
        "not found",
        "not available",
        "not applicable",
        "not specified",
        "unknown",
    }
)
_CLEAN_NUM_PATTERN = re.compile(r"[\$€£¥₹%()\[\]]|(?:TL|TRY|USD|EUR|GBP|CAD|AUD)\b", re.IGNORECASE)
_CURRENCY_SYMBOLS = re.compile(r"[\$€£¥₹%()\[\]]|(?:TL|TRY|USD|EUR|GBP|CAD|AUD)\b", re.IGNORECASE)
_GENERIC_WORDS = frozenset(
    {
        "number",
        "no",
        "code",
        "id",
        "name",
        "date",
        "item",
        "line",
        "box",
        "field",
        "value",
        "type",
    }
)


def is_boolean_or_empty(val: Any) -> bool:
    """Check if the value represents a boolean, null, or placeholder."""
    if val is None:
        return True
    if isinstance(val, (list, tuple, dict, set)):
        return len(val) == 0
    s = str(val).strip().lower()
    return s in _BOOLEAN_TERMS or s in _EMPTY_TERMS


_DATE_DMY_PATTERN = re.compile(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})$")
_DATE_ISO_PATTERN = re.compile(r"^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$")


def normalize_value_for_collision(val: Any) -> str:
    """Normalize numeric/text values to a canonical string for collision grouping."""
    if not val or isinstance(val, (list, tuple, dict, set)):
        return ""
    s = str(val).strip()
    m_dmy = _DATE_DMY_PATTERN.match(s)
    if m_dmy:
        p1, p2, p3 = m_dmy.groups()
        if len(p3) == 2:
            p3 = f"20{p3}"
        return f"{int(p1):02d}/{int(p2):02d}/{p3}"

    m_iso = _DATE_ISO_PATTERN.match(s)
    if m_iso:
        year, month, day = m_iso.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    cleaned = _CLEAN_NUM_PATTERN.sub("", s).replace(",", "").strip(".").strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.endswith(".00"):
        cleaned = cleaned[:-3]
    # Strip leading minus/plus so that -5,307 and (5,307) group with 5307
    return cleaned.lstrip("-+").strip()


def count_value_occurrences_in_store(
    norm_val: str,
    store: SessionVectorStore,
    relevant_pages: set[Any] | None = None,
) -> int:
    """Count how many times a given literal value appears in the document chunks.

    If relevant_pages is provided, counts occurrences strictly on those pages to prevent
    repeated header/footer strings across a multi-page document from inflating the allowance.
    Supports either integer page numbers or (doc_name, page_number) tuples.
    """
    if not norm_val or len(norm_val) < 2:
        return 1

    # Lookaround pattern ensures boundary match without failing on negative signs or punctuation
    pat = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(norm_val) + r"(?![a-zA-Z0-9])", re.IGNORECASE)

    # Group chunk texts by page to eliminate artificial duplicates from chunk overlaps
    page_texts: dict[tuple[str, int | None], list[str]] = {}
    for chunk in store.chunks:
        if not chunk.text or chunk.text.startswith("[Görsel"):
            continue
        key = (chunk.doc_name or chunk.doc_id, chunk.page_number)
        if relevant_pages:
            if key not in relevant_pages and chunk.page_number not in relevant_pages:
                continue
        page_texts.setdefault(key, []).append(chunk.text)

    total_occurrences = 0
    for texts in page_texts.values():
        unique_lines: list[str] = []
        seen_lines: set[str] = set()
        for t in texts:
            for line in t.splitlines():
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    unique_lines.append(stripped)
        page_unified_text = "\n".join(unique_lines)
        cleaned_page = _CURRENCY_SYMBOLS.sub(" ", page_unified_text).replace(",", "").lower()
        matches = len(pat.findall(cleaned_page))
        total_occurrences += matches

    return max(1, total_occurrences)


def compute_field_affinity_score(
    item: ExtractionItemQuery,
    result: ExtractionItemResult,
) -> float:
    """Compute how strongly a field owns its predicted value based on context affinity."""
    base_score = float(result.confidence or 0.5)

    if not result.value or not result.evidence:
        return base_score

    snippet = (result.evidence.snippet or "").lower()
    val_str = str(result.value).strip().lower()

    # Keyword overlap between query description/key and supporting snippet (multilingual)
    key_words = set(re.findall(r"\w{3,}", item.key.replace("_", " ").lower()))
    desc_words = set(re.findall(r"\w{3,}", (item.description or "").lower()))
    query_words = key_words | desc_words

    snippet_words = set(re.findall(r"\w{3,}", snippet))

    # Weight distinctive words higher than generic form words (e.g. 'solicitation' > 'number')
    distinctive_query_words = query_words - _GENERIC_WORDS
    distinctive_overlap = len(distinctive_query_words & snippet_words)
    generic_overlap = len(query_words & _GENERIC_WORDS & snippet_words)
    overlap_score = (0.25 * distinctive_overlap) + (0.05 * generic_overlap)

    # Check proximity in snippet if value is present in snippet
    val_clean = val_str.replace("$", "").replace("%", "").strip()
    proximity_bonus = 0.0
    if val_clean and val_clean in snippet:
        val_pos = snippet.find(val_clean)
        min_dist = 9999
        words_to_measure = distinctive_query_words if distinctive_query_words else query_words
        for qw in words_to_measure:
            q_pos = snippet.find(qw)
            if q_pos != -1:
                dist = abs(q_pos - val_pos)
                if dist < min_dist:
                    min_dist = dist
        if min_dist < 40:
            proximity_bonus = 0.35
        elif min_dist < 100:
            proximity_bonus = 0.20

    # Type compatibility
    type_bonus = 0.0
    if item.type:
        t = item.type.lower()
        if t in {"number", "money", "currency", "percentage"}:
            if any(c.isdigit() for c in val_str):
                type_bonus = 0.10
        elif t == "boolean":
            if val_str in _BOOLEAN_TERMS:
                type_bonus = 0.10

    return base_score + overlap_score + proximity_bonus + type_bonus


def _extract_collision_scalars(v: Any) -> list[str]:
    """Extract all distinctive numeric or text scalars from nested dict or list structures."""
    scalars: list[str] = []
    if isinstance(v, dict):
        for sub_v in v.values():
            if sub_v is not None and not is_boolean_or_empty(sub_v):
                norm_sub = normalize_value_for_collision(sub_v)
                if any(c.isdigit() for c in norm_sub) and len(norm_sub) >= 2:
                    if norm_sub not in scalars:
                        scalars.append(norm_sub)
    elif isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, dict):
                for sub_v in item.values():
                    if sub_v is not None and not is_boolean_or_empty(sub_v):
                        norm_sub = normalize_value_for_collision(sub_v)
                        if any(c.isdigit() for c in norm_sub) and len(norm_sub) >= 2:
                            if norm_sub not in scalars:
                                scalars.append(norm_sub)
            elif item is not None and not is_boolean_or_empty(item):
                norm_sub = normalize_value_for_collision(item)
                if any(c.isdigit() for c in norm_sub) and len(norm_sub) >= 2:
                    if norm_sub not in scalars:
                        scalars.append(norm_sub)
    return scalars


def _extract_primary_collision_scalar(v: Any) -> str | None:
    """Backward-compatible helper returning first salient scalar from list/dict."""
    scalars = _extract_collision_scalars(v)
    return scalars[0] if scalars else None


def resolve_cross_field_collisions(
    results: list[ExtractionItemResult],
    items: list[ExtractionItemQuery],
    store: SessionVectorStore,
) -> list[ExtractionItemResult]:
    """Resolve duplicate value assignments across fields deterministically.

    Identifies when multiple fields claim the exact same value in the document
    (e.g. LLM copying a salient number to empty adjacent fields) and uses context
    affinity to assign it only to the true owner, nullifying hallucinated duplicates.
    """
    item_map = {item.key: item for item in items}

    # Step 1: Group by normalized value (including single & multi-row dict/array scalars)
    val_groups: dict[str, list[int]] = {}
    for idx, res in enumerate(results):
        if not res.found or is_boolean_or_empty(res.value):
            continue
        if isinstance(res.value, (list, dict)):
            sub_scalars = _extract_collision_scalars(res.value)
            for sub_scalar in sub_scalars:
                if len(sub_scalar) >= 2:
                    idx_list = val_groups.setdefault(sub_scalar, [])
                    if idx not in idx_list:
                        idx_list.append(idx)
            continue
        norm_v = normalize_value_for_collision(res.value)
        if len(norm_v) >= 2:
            idx_list = val_groups.setdefault(norm_v, [])
            if idx not in idx_list:
                idx_list.append(idx)

    # Step 2: Check collisions where M fields share the same value
    for norm_val, indices in val_groups.items():
        if len(indices) <= 1:
            continue

        # Target candidate pages to eliminate false inflation from multi-page header/footers
        candidate_pages: set[Any] = set()
        for i in indices:
            ev = results[i].evidence
            if ev and ev.page is not None:
                if ev.document:
                    candidate_pages.add((ev.document, ev.page))
                candidate_pages.add(ev.page)

        # Count genuine occurrences in the document (restricted to candidate pages if known)
        n_allowed = count_value_occurrences_in_store(
            norm_val,
            store,
            relevant_pages=candidate_pages if candidate_pages else None,
        )

        if len(indices) > n_allowed:
            # Score each competing field
            candidate_scores: list[tuple[float, int]] = []
            for idx in indices:
                res = results[idx]
                query = item_map.get(res.key)
                if query:
                    aff_score = compute_field_affinity_score(query, res)
                else:
                    aff_score = float(res.confidence or 0.0)
                candidate_scores.append((aff_score, idx))

            # Sort descending by affinity score
            candidate_scores.sort(key=lambda x: x[0], reverse=True)

            # Keep top n_allowed, nullify the rest
            for _, idx in candidate_scores[n_allowed:]:
                res = results[idx]
                res.found = False
                res.value = None
                res.confidence = 0.0
                res.evidence = None

    return results
