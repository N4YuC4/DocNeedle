"""Tests for RapidOCR engine, layout reconstruction, Turkish normalization, and cache."""

from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from docneedle.ingest.cache import DocumentIngestionCache, global_ingestion_cache
from docneedle.ingest.models import DocumentChunk, IngestResult
from docneedle.ingest.ocr_engine import (
    EphemeralRapidOCREngine,
    fit_image,
    format_ocr_layout,
    normalize_turkish_ocr_artifacts,
)
from docneedle.ingest.parser import _is_extracted_text_readable
from docneedle.ingest.service import IngestionService


class TestFitImage:
    """Test image resolution scaling before RapidOCR inference."""

    def test_fit_image_downscale(self) -> None:
        img = Image.new("RGB", (4096, 2048), color="white")
        fitted = fit_image(img, longest_edge=2048)
        assert max(fitted.size) == 2048
        assert fitted.size == (2048, 1024)

    def test_fit_image_never_upscales(self) -> None:
        img = Image.new("RGB", (800, 600), color="white")
        fitted = fit_image(img, longest_edge=2048)
        assert fitted.size == (800, 600)


class TestOcrLayoutAndNormalization:
    """Test spatial line reconstruction and conditional Turkish artifact normalization."""

    def test_format_ocr_layout_prevents_word_gluing_and_separates_columns(self) -> None:
        # Box 1 and Box 2 touch closely on line 1; Box 3 is a separate column on line 1;
        # Box 4 is stacked directly below Box 1.
        boxes = np.array(
            [
                [[10, 10], [80, 10], [80, 30], [10, 30]],  # "Kimlik No"
                [[82, 10], [180, 10], [180, 30], [82, 30]],  # "12757385200" (tiny gap=2)
                [[260, 10], [380, 10], [380, 30], [260, 30]],  # "Hüseyin ABAZ" (wide gap=80)
                [[10, 45], [120, 45], [120, 65], [10, 65]],  # "Baba Adı: Ömer" (next line)
            ],
            dtype=np.float32,
        )
        txts = ("Kimlik No", "12757385200", "Hüseyin ABAZ", "Baba Adı: Ömer")
        out = format_ocr_layout(boxes, txts)

        # Must have a space between touching boxes (never "Kimlik No12757385200")
        assert "Kimlik No 12757385200" in out
        # Must separate wide columns with " | "
        assert "12757385200 | Hüseyin ABAZ" in out
        # Next line must be on a separate line
        assert "\nBaba Adı: Ömer" in out

    def test_format_ocr_layout_prevents_vertical_stack_merging(self) -> None:
        # Two boxes with close center_y but overlapping X coordinates (stacked cell label & value)
        boxes = np.array(
            [
                [[20, 10], [140, 10], [140, 26], [20, 26]],  # "Adı Soyadı"
                [[20, 22], [140, 22], [140, 38], [20, 38]],  # "Hüseyin ABAZ"
            ],
            dtype=np.float32,
        )
        txts = ("Adı Soyadı", "Hüseyin ABAZ")
        out = format_ocr_layout(boxes, txts)
        assert out == "Adı Soyadı\nHüseyin ABAZ"

    def test_normalize_turkish_ocr_artifacts(self) -> None:
        raw_turkish = (
            "T.C. İÇÍSLERÍ BAKANLIČI\nHacı Kamil KADIOČLU Müdür\nDečerli Kaġit bedeli karşiliġı"
        )
        cleaned = normalize_turkish_ocr_artifacts(raw_turkish)
        assert "T.C. İÇİSLERİ BAKANLIĞI" in cleaned
        assert "Hacı Kamil KADIOĞLU" in cleaned
        assert "Değerli Kağit bedeli karşiliğı" in cleaned

    def test_normalize_preserves_non_turkish_text(self) -> None:
        czech_text = "Česká republika a Španělsko ÍNDICE"
        assert normalize_turkish_ocr_artifacts(czech_text) == czech_text


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


class TestEphemeralRapidOCRTranscription:
    """Test EphemeralRapidOCREngine output handling and empty image safety."""

    def test_transcribe_with_markdown_output(self, monkeypatch) -> None:
        mock_output = MagicMock()
        mock_output.boxes = None
        mock_output.txts = (
            "FATURA / INVOICE",
            "Toplam Tutar: 15.000 TL",
            "KDV Dahil tutardır.",
        )
        mock_output.to_markdown.return_value = (
            "FATURA / INVOICE\n\nToplam Tutar: 15.000 TL  KDV Dahil tutardır."
        )

        mock_rapid_inst = MagicMock(return_value=mock_output)
        monkeypatch.setattr("rapidocr.RapidOCR", MagicMock(return_value=mock_rapid_inst))

        img = Image.new("RGB", (500, 500), color="white")
        with EphemeralRapidOCREngine() as ocr:
            text = ocr.transcribe_image(img)
            assert "FATURA / INVOICE" in text
            assert "Toplam Tutar: 15.000 TL" in text
            assert "KDV Dahil tutardır." in text

    def test_transcribe_empty_image_returns_empty_string(self, monkeypatch) -> None:
        mock_output = MagicMock()
        mock_output.boxes = None
        mock_output.txts = None
        mock_output.to_markdown.return_value = "没有检测到任何文本。"

        mock_rapid_inst = MagicMock(return_value=mock_output)
        monkeypatch.setattr("rapidocr.RapidOCR", MagicMock(return_value=mock_rapid_inst))

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with EphemeralRapidOCREngine() as ocr:
            text = ocr.transcribe_image(arr)
            assert text == ""
