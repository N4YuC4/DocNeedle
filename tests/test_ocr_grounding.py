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


class TestMultiDocGroundingAndSnippetResolution:
    """Test document context boundary tagging, visual chunk detection, and snippet resolution."""

    def test_format_chunk_context_adds_boundaries(self) -> None:
        from docneedle.extraction.service import _format_chunk_context

        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="contract.pdf",
            page_number=2,
            chunk_index=0,
            text="Taraflar: Sirket A ve Sirket B.",
        )
        formatted = _format_chunk_context(chunk)
        assert formatted.startswith("[Doküman: contract.pdf (Sayfa 2)]\nTaraflar:")

        # Avoid double-tagging if already formatted
        already_tagged = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="contract.pdf",
            chunk_index=1,
            text="[Doküman: contract.pdf]\nSome text",
        )
        assert _format_chunk_context(already_tagged) == "[Doküman: contract.pdf]\nSome text"

    def test_is_visual_chunk_discrimination(self) -> None:
        from docneedle.extraction.matcher import _is_visual_chunk

        # Text PDF chunk with image_bytes (rendered PDF page)
        text_chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="invoice.pdf",
            page_number=1,
            chunk_index=0,
            text="Fatura No: INV-1234\nToplam Tutar: 15.000 TL\nKDV: 3.000 TL" * 5,
            image_bytes=b"fake_jpeg_bytes",
        )
        assert _is_visual_chunk(text_chunk) is False

        # Visual image chunk
        img_chunk = DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_name="receipt.png",
            page_number=1,
            chunk_index=0,
            text="receipt > Sayfa 1\n[Görsel Doküman - receipt]",
            image_bytes=b"fake_jpeg_bytes",
        )
        assert _is_visual_chunk(img_chunk) is True

    def test_evidence_and_confidence_preserves_digital_text_snippets(self) -> None:
        from docneedle.extraction.service import ExtractionService
        from docneedle.retrieval.models import SearchResult

        # Digital text PDF with image_bytes should produce text snippet, NOT "Görsel Doküman"
        digital_chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="commercial_invoice.pdf",
            page_number=1,
            chunk_index=0,
            text="Fatura No: SRV-88219\nToplam Tutar: 19.200,00 TL\nTarih: 18 Ekim 2026",
            image_bytes=b"rendered_page_bytes",
        )
        sr = SearchResult(chunk=digital_chunk, score=0.95)

        evidence, conf = ExtractionService._create_evidence_and_confidence(
            raw_val="SRV-88219",
            matches=[sr],
            expected_type="string",
        )
        assert evidence.document == "commercial_invoice.pdf"
        assert evidence.page == 1
        assert "SRV-88219" in evidence.snippet
        assert "Görsel Doküman" not in evidence.snippet

        # Visual chunk without text should produce descriptive visual snippet
        visual_chunk = DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_name="service_receipt.png",
            page_number=1,
            chunk_index=0,
            text="[Görsel Doküman - service_receipt]",
            image_bytes=b"rendered_page_bytes",
        )
        sr_vis = SearchResult(chunk=visual_chunk, score=0.9)
        vis_ev, vis_conf = ExtractionService._create_evidence_and_confidence(
            raw_val="Murat Simsek",
            matches=[sr_vis],
            expected_type="string",
        )
        assert vis_ev.document == "service_receipt.png"
        assert "Görsel Doküman service_receipt.png (Sayfa 1) doğrudan incelendi" in vis_ev.snippet

    def test_matcher_visual_chunk_fallback_when_digital_text_absent(self) -> None:
        from docneedle.extraction.matcher import _find_best_supporting_match
        from docneedle.retrieval.models import SearchResult

        text_chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="invoice.pdf",
            page_number=1,
            chunk_index=0,
            text="Fatura No: INV-1234\nFirma: ABC Anonim Sirketi",
            image_bytes=b"img",
        )
        vis_chunk = DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_name="receipt.png",
            page_number=1,
            chunk_index=0,
            text="[Görsel Doküman - receipt]",
            image_bytes=b"img",
        )

        matches = [
            SearchResult(chunk=text_chunk, score=0.9),
            SearchResult(chunk=vis_chunk, score=0.8),
        ]

        # "Murat Simsek" was extracted visually from receipt.png, not in text_chunk
        best = _find_best_supporting_match("Murat Simsek", matches)
        assert best.chunk.doc_name == "receipt.png"

    def test_schema_prompt_contains_multidoc_isolation_rule(self) -> None:
        from docneedle.llm.prompts import _build_schema_extraction_prompt

        prompt = _build_schema_extraction_prompt(
            fields=[{"key": "test_key", "type": "string"}],
            context="[Doküman: test.pdf]\nSome text",
        )
        assert "Multi-document isolation:" in prompt
        assert "extract each field strictly from its matching document type" in prompt

    def test_multilingual_iso_date_matching(self) -> None:
        from docneedle.extraction.matcher import _find_best_supporting_match, _match_iso_date
        from docneedle.retrieval.models import SearchResult

        c1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="invoice.pdf",
            chunk_index=0,
            text="Fatura No: INV-100\nTarih: 12 Kasim 2026",
            image_bytes=b"img",
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_name="lease_tr.pdf",
            chunk_index=0,
            text="Isbu sozlesme imza tarihi 5 Ocak 2026 olup...",
            image_bytes=b"img",
        )
        c3 = DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            doc_name="agreement_en.pdf",
            chunk_index=0,
            text="This agreement is dated January 5, 2026 by and between...",
            image_bytes=b"img",
        )
        c4 = DocumentChunk(
            chunk_id="c4",
            doc_id="d4",
            doc_name="vertrag_de.pdf",
            chunk_index=0,
            text="Der Vertrag tritt am 5. Januar 2026 in Kraft.",
            image_bytes=b"img",
        )

        matches = [SearchResult(chunk=c1, score=0.9), SearchResult(chunk=c2, score=0.8)]
        best = _find_best_supporting_match("2026-01-05", matches)
        assert best.chunk.doc_name == "lease_tr.pdf"

        # Verify English and German without any language-specific month lists
        res_en = _match_iso_date("2026-01-05", [SearchResult(chunk=c3, score=0.8)])
        assert res_en is not None and res_en.chunk.doc_name == "agreement_en.pdf"

        res_de = _match_iso_date("2026-01-05", [SearchResult(chunk=c4, score=0.8)])
        assert res_de is not None and res_de.chunk.doc_name == "vertrag_de.pdf"

    def test_strict_token_overlap_rejects_spurious_number_matches(self) -> None:
        from docneedle.extraction.matcher import _find_best_supporting_match
        from docneedle.retrieval.models import SearchResult

        # invoice has "00" and "TL", but does NOT have 19.200
        invoice_chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="commercial_invoice.pdf",
            chunk_index=0,
            text="Hizmet Tutari: 115.000,00 TL\nToplam: 113.850,00 TL",
            image_bytes=b"img",
        )
        receipt_chunk = DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_name="service_receipt.png",
            chunk_index=0,
            text="[Görsel Doküman - service_receipt]",
            image_bytes=b"img",
        )
        matches = [
            SearchResult(chunk=invoice_chunk, score=0.9),
            SearchResult(chunk=receipt_chunk, score=0.8),
        ]

        # 19.200,00 TL must fall back to the visual receipt, not match invoice on "00 TL"
        best = _find_best_supporting_match("19.200,00 TL", matches)
        assert best.chunk.doc_name == "service_receipt.png"
