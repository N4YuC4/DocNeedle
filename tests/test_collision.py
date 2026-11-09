"""Unit tests for deterministic cross-field collision resolver."""

from unittest.mock import MagicMock

from docneedle.extraction.collision import (
    compute_field_affinity_score,
    count_value_occurrences_in_store,
    is_boolean_or_empty,
    normalize_value_for_collision,
    resolve_cross_field_collisions,
)
from docneedle.extraction.schemas import (
    ExtractionEvidence,
    ExtractionItemQuery,
    ExtractionItemResult,
)
from docneedle.ingest.models import DocumentChunk


class TestCollisionResolver:
    """Test suite for cross-field duplicate resolution."""

    def test_is_boolean_or_empty(self) -> None:
        assert is_boolean_or_empty(None) is True
        assert is_boolean_or_empty("") is True
        assert is_boolean_or_empty("   ") is True
        assert is_boolean_or_empty("none") is True
        assert is_boolean_or_empty("N/A") is True
        assert is_boolean_or_empty("not found") is True
        assert is_boolean_or_empty("--") is True
        assert is_boolean_or_empty("()") is True
        assert is_boolean_or_empty("true") is True
        assert is_boolean_or_empty("false") is True
        assert is_boolean_or_empty("yes") is True

        assert is_boolean_or_empty("123") is False
        assert is_boolean_or_empty("Acme Corp") is False
        assert is_boolean_or_empty("$5,307") is False

    def test_normalize_value_for_collision(self) -> None:
        assert normalize_value_for_collision("$5,307") == "5307"
        assert normalize_value_for_collision("-5,307") == "5307"
        assert normalize_value_for_collision("(5,307)") == "5307"
        assert normalize_value_for_collision("USD 5,307.00") == "5307"
        assert normalize_value_for_collision("Acme Corporation") == "acme corporation"
        assert normalize_value_for_collision("") == ""
        assert normalize_value_for_collision(None) == ""

    def test_count_value_occurrences_in_store(self) -> None:
        store = MagicMock()
        # Page 1 has two chunks with an overlapping line
        chunk1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=0,
            text="Line 1: Gross Income 50000\nLine 12: Deductions -5,307\nLine 13: Net 44693",
            page_number=1,
        )
        chunk2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=1,
            text="Line 12: Deductions -5,307\nLine 13: Net 44693\nLine 14: Tax 8000",
            page_number=1,
        )
        # Page 2 has no occurrence of 5307
        chunk3 = DocumentChunk(
            chunk_id="c3",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=2,
            text="Page 2 summary text without that number",
            page_number=2,
        )
        store.chunks = [chunk1, chunk2, chunk3]

        # Normalized 5307 appears only once on page 1 (despite duplicate line in overlapping chunk)
        count = count_value_occurrences_in_store("5307", store)
        assert count == 1

        # Number appearing on multiple pages
        chunk3.text = "Page 2 repeat: -5,307"
        count_multi = count_value_occurrences_in_store("5307", store)
        assert count_multi == 2

        # Restrict count to candidate pages (e.g. only page 1)
        count_page1 = count_value_occurrences_in_store("5307", store, relevant_pages={1})
        assert count_page1 == 1

    def test_resolve_cross_field_collisions_deduplication(self) -> None:
        store = MagicMock()
        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=0,
            text="Line 12: Other deductions -5,307\nLine 13: Empty\nLine 14: Empty",
            page_number=1,
        )
        store.chunks = [chunk]

        items = [
            ExtractionItemQuery(
                key="other_deductions",
                description="Line 12 other deductions",
                type="number",
            ),
            ExtractionItemQuery(
                key="rental_loss",
                description="Line 13 rental real estate loss",
                type="number",
            ),
            ExtractionItemQuery(
                key="farm_income",
                description="Line 14 net farm profit or loss",
                type="number",
            ),
        ]

        # LLM hallucinated -5,307 across all 3 fields, but it only belongs to other_deductions
        results = [
            ExtractionItemResult(
                key="other_deductions",
                value="-5,307",
                found=True,
                confidence=0.95,
                evidence=ExtractionEvidence(
                    document="doc.pdf",
                    page=1,
                    snippet="Line 12: Other deductions -5,307",
                ),
            ),
            ExtractionItemResult(
                key="rental_loss",
                value="-5,307",
                found=True,
                confidence=0.80,
                evidence=ExtractionEvidence(
                    document="doc.pdf",
                    page=1,
                    snippet="Line 12: Other deductions -5,307",
                ),
            ),
            ExtractionItemResult(
                key="farm_income",
                value="-5,307",
                found=True,
                confidence=0.75,
                evidence=ExtractionEvidence(
                    document="doc.pdf",
                    page=1,
                    snippet="Line 12: Other deductions -5,307",
                ),
            ),
        ]

        resolved = resolve_cross_field_collisions(results, items, store)

        # Winner must be other_deductions
        assert resolved[0].found is True
        assert resolved[0].value == "-5,307"

        # The other 2 fields must be nullified
        assert resolved[1].found is False
        assert resolved[1].value is None
        assert resolved[1].confidence == 0.0
        assert resolved[1].evidence is None

        assert resolved[2].found is False
        assert resolved[2].value is None
        assert resolved[2].confidence == 0.0
        assert resolved[2].evidence is None

    def test_compute_field_affinity_score(self) -> None:
        item_sol = ExtractionItemQuery(
            key="solicitation_number", description="Solicitation number block 5"
        )
        res_sol = ExtractionItemResult(
            key="solicitation_number",
            value="36C10B19Q0176",
            found=True,
            confidence=0.85,
            evidence=ExtractionEvidence(
                document="doc.pdf",
                page=1,
                snippet="5. SOLICITATION NUMBER 36C10B19Q0176",
            ),
        )

        item_con = ExtractionItemQuery(key="contract_number", description="Contract number block 2")
        res_con = ExtractionItemResult(
            key="contract_number",
            value="36C10B19Q0176",
            found=True,
            confidence=0.85,
            evidence=ExtractionEvidence(
                document="doc.pdf",
                page=1,
                snippet="5. SOLICITATION NUMBER 36C10B19Q0176",
            ),
        )

        score_sol = compute_field_affinity_score(item_sol, res_sol)
        score_con = compute_field_affinity_score(item_con, res_con)
        assert score_sol > score_con

    def test_extract_collision_scalars_multi_row(self) -> None:
        from docneedle.extraction.collision import _extract_collision_scalars

        table_val = [
            {"code": "A", "amount": "-5,307"},
            {"code": "C", "amount": "34,199"},
        ]
        scalars = _extract_collision_scalars(table_val)
        assert "5307" in scalars
        assert "34199" in scalars

    def test_collision_multi_row_table_with_scalar(self) -> None:
        from docneedle.extraction.collision import resolve_cross_field_collisions

        store = MagicMock()
        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=0,
            text=(
                "Box 1 Ordinary income -5,307\n"
                "Box 2 Rental income\n"
                "Box 14 Self employment A -5,307 C 34,199"
            ),
            page_number=1,
        )
        store.chunks = [chunk]

        items = [
            ExtractionItemQuery(
                key="box_1_ordinary_business_income",
                description="Box 1 ordinary income",
            ),
            ExtractionItemQuery(
                key="box_2_net_rental_real_estate",
                description="Box 2 net rental real estate",
            ),
            ExtractionItemQuery(
                key="box_14_self_employment_earnings",
                description="Box 14 self employment earnings",
            ),
        ]

        # Box 1 extracted -5,307
        res1 = ExtractionItemResult(
            key="box_1_ordinary_business_income",
            value="-5,307",
            found=True,
            confidence=0.95,
            evidence=ExtractionEvidence(
                document="doc.pdf", page=1, snippet="Box 1 Ordinary income -5,307"
            ),
        )
        # Box 2 hallucinated -5,307
        res2 = ExtractionItemResult(
            key="box_2_net_rental_real_estate",
            value="-5,307",
            found=True,
            confidence=0.60,
            evidence=ExtractionEvidence(
                document="doc.pdf", page=1, snippet="Box 1 Ordinary income -5,307"
            ),
        )
        # Box 14 extracted real table containing -5,307
        res14 = ExtractionItemResult(
            key="box_14_self_employment_earnings",
            value=[{"code": "A", "amount": "-5,307"}, {"code": "C", "amount": "34,199"}],
            found=True,
            confidence=0.92,
            evidence=ExtractionEvidence(
                document="doc.pdf",
                page=1,
                snippet="Box 14 Self employment A -5,307 C 34,199",
            ),
        )

        resolved = resolve_cross_field_collisions([res1, res2, res14], items, store)
        by_key = {r.key: r for r in resolved}

        # Box 1 and Box 14 legitimately own the two occurrences of 5307
        assert by_key["box_1_ordinary_business_income"].found is True
        assert by_key["box_14_self_employment_earnings"].found is True
        # Box 2 hallucination is nullified
        assert by_key["box_2_net_rental_real_estate"].found is False
        assert by_key["box_2_net_rental_real_estate"].value is None
