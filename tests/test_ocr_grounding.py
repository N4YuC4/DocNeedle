"""Tests for visual and multimodal ingestion, CMap readability validation, and cache."""

import io

from PIL import Image

from docneedle.ingest.cache import DocumentIngestionCache, global_ingestion_cache
from docneedle.ingest.models import DocumentChunk, IngestResult
from docneedle.ingest.parser import _is_extracted_text_readable
from docneedle.ingest.service import IngestionService


class TestPdfReadabilityValidation:
    """Test detection of corrupted PDF ToUnicode CMap text layers."""

    def test_rejects_corrupted_cmap_text(self) -> None:
        corrupted_yurda = (
            "SAYI 2023-EGM042023092922114509790 / 22:11 <(57$5ú+ ø17(51(7 "
            "ZZZWXUNL\\HJRYWU KONU <85'$ *ø5øùd,.,ù%(/* "
            "(6ø 9(5ú/(&(øú0$.$0 LOJLQWHUHNRQVRORVOX÷X 7&.ú0/ú.180$5$6,"
        )
        assert _is_extracted_text_readable(corrupted_yurda) is False

    def test_accepts_valid_turkish_digital_text(self) -> None:
        valid_text = (
            "T.C. İÇİŞLERİ BAKANLIĞI Nüfus ve Vatandaşlık İşleri Genel Müdürlüğü "
            "YERLEŞİM YERİ VE DİĞER ADRES BELGESİ ALİTAŞI MAH. MERKEZ / ADIYAMAN"
        )
        assert _is_extracted_text_readable(valid_text) is True


class TestIngestionCache:
    """Test in-memory thread-safe caching of ingested document chunks."""

    def test_cache_put_get_clear(self) -> None:
        cache = DocumentIngestionCache(max_size=3)
        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=0,
            text="Invoice content",
        )
        result = IngestResult(
            doc_id="d1",
            doc_name="doc.pdf",
            total_pages=1,
            total_chunks=1,
            chunks=[chunk],
        )

        cache.put("hash1", result)
        assert cache.contains("hash1")
        cached = cache.get("hash1")
        assert cached is not None
        assert cached.doc_name == "doc.pdf"

        # Test LRU eviction
        cache.put("hash2", result)
        cache.put("hash3", result)
        cache.put("hash4", result)  # should evict hash1

        assert not cache.contains("hash1")
        assert cache.contains("hash2")
        assert cache.contains("hash4")

        cache.clear()
        assert len(cache) == 0

    def test_ingestion_service_uses_cache(self, monkeypatch) -> None:
        global_ingestion_cache.clear()
        service = IngestionService()

        parse_count = 0
        original_parse_bytes = service.parser.parse_bytes

        def mock_parse_bytes(content: bytes, filename: str):
            nonlocal parse_count
            parse_count += 1
            return original_parse_bytes(content=content, filename=filename)

        monkeypatch.setattr(service.parser, "parse_bytes", mock_parse_bytes)

        test_bytes = b"# Sample Document\n\nContent for testing caching."
        filename = "test.md"

        res1 = service.ingest_bytes(content=test_bytes, filename=filename)
        assert parse_count == 1
        assert res1.total_chunks > 0

        res2 = service.ingest_bytes(content=test_bytes, filename=filename)
        assert parse_count == 1
        assert res2.total_chunks == res1.total_chunks

        global_ingestion_cache.clear()


class TestVisualIngestionChunks:
    """Test OCR-free ingestion preserving image_bytes on DocumentChunk."""

    def test_image_file_ingests_with_image_bytes(self) -> None:
        img = Image.new("RGB", (200, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        service = IngestionService()
        result = service.ingest_bytes(content=png_bytes, filename="test_doc.png")

        assert result.total_chunks >= 1
        chunk = result.chunks[0]
        assert chunk.image_bytes is not None
        assert len(chunk.image_bytes) > 0
        assert "Görsel Doküman" in chunk.text
