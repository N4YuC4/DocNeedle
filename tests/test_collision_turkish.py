"""Unit tests for Turkish unicode support, evidence nullification, and ISO dates."""

from unittest.mock import MagicMock

from docneedle.extraction.collision import (
    compute_field_affinity_score,
    normalize_value_for_collision,
    resolve_cross_field_collisions,
)
from docneedle.extraction.schemas import (
    ExtractionEvidence,
    ExtractionItemQuery,
    ExtractionItemResult,
)
from docneedle.ingest.models import DocumentChunk
from docneedle.ingest.parser import _format_clean_lines


class TestTurkishCollisionAndParser:
    """Test suite for Turkish unicode handling, evidence cleanup, and parser safety."""

    def test_iso_date_normalization(self) -> None:
        """Verify that ISO dates (YYYY-MM-DD) are normalized to canonical DD/MM/YYYY."""
        assert normalize_value_for_collision("2024-05-15") == "15/05/2024"
        assert normalize_value_for_collision("2024.12.01") == "01/12/2024"
        assert normalize_value_for_collision("15/05/2024") == "15/05/2024"

    def test_turkish_affinity_scoring_with_unicode_letters(self) -> None:
        """Verify that Turkish characters (ç, ş, ğ, ö, ü, ı) contribute to keyword overlap."""
        query = ExtractionItemQuery(
            key="kira_bedeli",
            description="Aylık kira bedeli ve ödeme tutarı",
            type="money",
        )
        result = ExtractionItemResult(
            key="kira_bedeli",
            found=True,
            value="25.000 TL",
            confidence=0.85,
            evidence=ExtractionEvidence(
                document="kira_sozlesmesi.pdf",
                page=1,
                snippet="Madde 4: Aylık kira bedeli 25.000 TL olarak belirlenmiştir.",
            ),
        )
        score = compute_field_affinity_score(query, result)
        # Should receive strong overlap bonus and proximity bonus
        assert score > 1.0

    def test_nullified_collision_clears_evidence(self) -> None:
        """Verify that nullified duplicates have both value=None and evidence=None."""
        store = MagicMock()
        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="fatura.pdf",
            chunk_index=0,
            text="Fatura Toplam Tutarı: 1.500 TL\nÖn ödeme: Yok",
            page_number=1,
        )
        store.chunks = [chunk]

        items = [
            ExtractionItemQuery(
                key="toplam_tutar",
                description="Faturanın genel toplam tutarı",
                type="money",
            ),
            ExtractionItemQuery(
                key="ara_toplam",
                description="Ara toplam tutarı",
                type="money",
            ),
        ]

        # Model hallucinated 1500 for both fields, but document only has 1 occurrence
        results = [
            ExtractionItemResult(
                key="toplam_tutar",
                value="1.500 TL",
                found=True,
                confidence=0.95,
                evidence=ExtractionEvidence(
                    document="fatura.pdf",
                    page=1,
                    snippet="Fatura Toplam Tutarı: 1.500 TL",
                ),
            ),
            ExtractionItemResult(
                key="ara_toplam",
                value="1.500 TL",
                found=True,
                confidence=0.70,
                evidence=ExtractionEvidence(
                    document="fatura.pdf",
                    page=1,
                    snippet="Fatura Toplam Tutarı: 1.500 TL",
                ),
            ),
        ]

        resolved = resolve_cross_field_collisions(results, items, store)
        winner = next(r for r in resolved if r.key == "toplam_tutar")
        loser = next(r for r in resolved if r.key == "ara_toplam")

        assert winner.found is True
        assert winner.value == "1.500 TL"
        assert winner.evidence is not None

        # Loser must have found=False, value=None, and evidence=None
        assert loser.found is False
        assert loser.value is None
        assert loser.confidence == 0.0
        assert loser.evidence is None

    def test_format_clean_lines_preserves_numeric_lists(self) -> None:
        """Ensure single-digit numbers on separate lines are NOT merged into one number."""
        text = "1\n2\n3\n4"
        cleaned = _format_clean_lines(text)
        assert cleaned == "1\n\n2\n\n3\n\n4"

    def test_format_clean_lines_coalesces_vertical_words(self) -> None:
        """Ensure vertical single-letter alphabetical runs are merged into single words."""
        text = "S\nU\nM\nM\nA\nR\nY"
        cleaned = _format_clean_lines(text)
        assert cleaned == "SUMMARY"
