"""Confidence scoring engine."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from docneedle.config import settings
from docneedle.retrieval.bm25 import normalize_search_text


def normalize_for_match(text: str) -> str:
    """Normalize text for Turkish-aware case-insensitive matching without combining-dot split."""
    return text.replace("İ", "i").replace("I", "ı").lower().replace("i\u0307", "i")


# Alias for backward compatibility with private imports
_normalize_for_match = normalize_for_match

# Alignment & Type constants
CONFIDENCE_BOOLEAN_SCORE = 0.92
CONFIDENCE_EXACT_MATCH_SCORE = 0.96
CONFIDENCE_ALL_WORDS_SCORE = 0.88
CONFIDENCE_PARTIAL_BASE = 0.65
CONFIDENCE_PARTIAL_RATIO_WEIGHT = 0.22
CONFIDENCE_LOW_OVERLAP_SCORE = 0.60
CONFIDENCE_SHORT_MISMATCH_SCORE = 0.65
CONFIDENCE_CLAMP_MIN = 0.10
CONFIDENCE_CLAMP_MAX = 0.99

PENALTY_HIGH = 0.20
PENALTY_MEDIUM = 0.15

_IBAN_PATTERN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$")
_WORD_PATTERN = re.compile(r"\w+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_BOOLEAN_TERMS = frozenset({"true", "false", "yes", "no", "evet", "hayır", "hayir"})


@dataclass(slots=True, frozen=True)
class TypeConstraint:
    """Validation predicate and penalty weight for expected semantic data types."""

    is_valid: Callable[[str], bool]
    penalty: float


def _has_digits(val: str) -> bool:
    return any(ch.isdigit() for ch in val)


def _validate_iban(val: str) -> bool:
    clean_iban = _WHITESPACE_PATTERN.sub("", val.upper())
    is_valid_iban = bool(_IBAN_PATTERN.match(clean_iban))
    has_digit = any(ch.isdigit() for ch in clean_iban)
    return is_valid_iban or (has_digit and len(clean_iban) >= 12)


def _validate_email(val: str) -> bool:
    return "@" in val and "." in val


def _validate_phone(val: str) -> bool:
    return sum(ch.isdigit() for ch in val) >= 7


def _validate_id(val: str) -> bool:
    return sum(ch.isdigit() for ch in val) >= 4


TYPE_CONSTRAINTS: dict[str, TypeConstraint] = {
    "money": TypeConstraint(is_valid=_has_digits, penalty=PENALTY_HIGH),
    "number": TypeConstraint(is_valid=_has_digits, penalty=PENALTY_HIGH),
    "percentage": TypeConstraint(is_valid=_has_digits, penalty=PENALTY_HIGH),
    "iban": TypeConstraint(is_valid=_validate_iban, penalty=PENALTY_HIGH),
    "email": TypeConstraint(is_valid=_validate_email, penalty=PENALTY_HIGH),
    "phone": TypeConstraint(is_valid=_validate_phone, penalty=PENALTY_HIGH),
    "date": TypeConstraint(is_valid=_has_digits, penalty=PENALTY_MEDIUM),
    "id": TypeConstraint(is_valid=_validate_id, penalty=PENALTY_MEDIUM),
}


class ConfidenceCalculator:
    """Computes confidence score based on extraction status, chunk alignment, and type checks."""

    @staticmethod
    def _calculate_alignment_score(
        norm_val: str,
        folded_val: str,
        target_chunk_text: str,
        expected_type: str | None,
    ) -> float:
        """Evaluate how closely the extracted value aligns with the supporting evidence chunk."""
        norm_chunk = normalize_for_match(target_chunk_text)
        folded_chunk = normalize_search_text(target_chunk_text)
        is_boolean_val = norm_val in _BOOLEAN_TERMS

        if is_boolean_val and expected_type and expected_type.lower() == "boolean":
            return CONFIDENCE_BOOLEAN_SCORE

        if len(norm_val) <= 3:
            exact_short_match = bool(re.search(r"\b" + re.escape(norm_val) + r"\b", norm_chunk))
            folded_short_match = bool(
                folded_val and re.search(r"\b" + re.escape(folded_val) + r"\b", folded_chunk)
            )
            if exact_short_match or folded_short_match:
                return CONFIDENCE_EXACT_MATCH_SCORE
            return CONFIDENCE_SHORT_MISMATCH_SCORE

        if norm_val in norm_chunk or (folded_val and folded_val in folded_chunk):
            return CONFIDENCE_EXACT_MATCH_SCORE

        words = [w for w in _WORD_PATTERN.findall(folded_val) if len(w) > 1]
        if words and all(w in folded_chunk for w in words):
            return CONFIDENCE_ALL_WORDS_SCORE
        if words:
            matched = sum(1 for w in words if w in folded_chunk)
            ratio = matched / len(words)
            return CONFIDENCE_PARTIAL_BASE + (CONFIDENCE_PARTIAL_RATIO_WEIGHT * ratio)

        return CONFIDENCE_LOW_OVERLAP_SCORE

    @staticmethod
    def _apply_retrieval_boost(base_score: float, retrieval_score: float) -> float:
        """Apply a bounded adjustment based on vector/reranker retrieval score."""
        if retrieval_score == 0.0:
            return base_score
        boost = min(0.02, max(-0.05, retrieval_score * 0.01))
        return base_score + boost

    @staticmethod
    def _apply_type_penalties(
        base_score: float,
        val_str: str,
        expected_type: str | None,
    ) -> float:
        """Penalize score when the value violates expected semantic data type constraints."""
        if not expected_type:
            return base_score

        constraint = TYPE_CONSTRAINTS.get(expected_type.lower())
        if constraint and not constraint.is_valid(val_str):
            return base_score - constraint.penalty

        return base_score

    @classmethod
    def calculate(
        cls,
        found: bool,
        value: Any = None,
        retrieval_score: float = 0.0,
        target_chunk_text: str | None = None,
        expected_type: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Calculate confidence score between 0.0 and 1.0."""
        if not found or value is None:
            return settings.confidence_not_found_score

        val_str = str(value).strip()
        if not val_str:
            return settings.confidence_not_found_score

        if target_chunk_text is None and expected_type is None:
            return settings.confidence_found_score

        norm_val = normalize_for_match(val_str)
        folded_val = normalize_search_text(val_str)
        score = settings.confidence_found_score

        if target_chunk_text:
            score = cls._calculate_alignment_score(
                norm_val=norm_val,
                folded_val=folded_val,
                target_chunk_text=target_chunk_text,
                expected_type=expected_type,
            )

        score = cls._apply_retrieval_boost(score, retrieval_score)
        score = cls._apply_type_penalties(score, val_str, expected_type)

        clamped = max(CONFIDENCE_CLAMP_MIN, min(CONFIDENCE_CLAMP_MAX, score))
        return round(clamped, 2)
