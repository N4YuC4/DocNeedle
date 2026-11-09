"""Unit tests for AcroForm interactive form extraction and formatting."""

import pypdfium2 as pdfium

from docneedle.ingest.parser import (
    _extract_acroform_fields_by_page,
    _format_acroform_fields_markdown,
)


class TestAcroFormExtraction:
    """Test suite for AcroForm interactive PDF form field parsing."""

    def test_format_acroform_fields_markdown_empty(self) -> None:
        assert _format_acroform_fields_markdown({}) == ""

    def test_format_acroform_fields_markdown_populated(self) -> None:
        fields = {
            "First Name": "John",
            "Total Income": "$50,000",
            "Has Deductions": "Yes",
        }
        md = _format_acroform_fields_markdown(fields)
        assert "### Dijital Form Alanları (AcroForm)" in md
        assert "| First Name | John |" in md
        assert "| Total Income | $50,000 |" in md
        assert "| Has Deductions | Yes |" in md

    def test_extract_acroform_fields_plain_pdf(self) -> None:
        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 200)
        res = _extract_acroform_fields_by_page(doc)
        doc.close()
        assert res == {}
