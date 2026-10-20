import tempfile
from pathlib import Path

import pytest

from docneedle.ingest.models import IngestResult
from docneedle.ingest.service import IngestionService


@pytest.fixture
def sample_markdown_file():
    content = (
        "# Ana Sözleşme Başlığı\n\n"
        "İşbu sözleşme DocNeedle sistemi için hazırlanmıştır.\n\n"
        "## Madde 1: Geçerlilik ve Süre\n"
        "Sözleşme başlangıç tarihi 1 Ocak 2026 olup bitiş tarihi "
        "31 Aralık 2028 olarak belirlenmiştir.\n\n"
        "## Madde 2: Bedel Tablosu\n"
        "| Hizmet Türü | Birim Fiyat | Toplam Tutar |\n"
        "|-------------|-------------|--------------|\n"
        "| RAG Analizi | 100 TL      | 5.000 TL     |\n"
        "| OCR Tarama  | 50 TL       | 1.500 TL     |\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def test_ingest_path(sample_markdown_file: str) -> None:
    service = IngestionService()
    result = service.ingest_path(sample_markdown_file, doc_id="doc_test_1")

    assert isinstance(result, IngestResult)
    assert result.doc_id == "doc_test_1"
    assert result.total_chunks > 0
    assert len(result.chunks) == result.total_chunks

    # Verify heading extraction
    texts = [c.text for c in result.chunks]
    assert any("1 Ocak 2026" in t for t in texts)

    # Verify headings and metadata
    first_chunk = result.chunks[0]
    assert first_chunk.chunk_id.startswith("doc_test_1_")
    assert first_chunk.doc_name.endswith(".md")


def test_ingest_bytes() -> None:
    service = IngestionService()
    raw_content = (
        b"# Teklif Belgesi\n\nFirma unvani: Acme Teknoloji AS\nTeklif bedeli: 125.000 USD\n"
    )
    result = service.ingest_bytes(
        content=raw_content,
        filename="teklif.md",
        doc_id="doc_test_bytes",
    )

    assert result.doc_id == "doc_test_bytes"
    assert result.doc_name == "teklif.md"
    assert result.total_chunks > 0
    assert any("Acme Teknoloji" in c.text for c in result.chunks)
