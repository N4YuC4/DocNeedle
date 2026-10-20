"""Unit tests for newly refactored modular helper functions across DocNeedle."""

import pytest

from docneedle.extraction.service import _build_retrieval_query, _find_best_supporting_match
from docneedle.ingest.chunker import DoclingChunker
from docneedle.ingest.models import DocumentChunk
from docneedle.llm.local_engine import (
    _build_extraction_prompt,
    _is_not_found_value,
    _parse_model_response,
    _sanitize_extracted_value,
)
from docneedle.retrieval.models import SearchResult
from docneedle.retrieval.store import _apply_document_diversity


class TestLLMLocalEngineHelpers:
    """Test pure functions extracted from LocalGemmaEngine."""

    def test_parse_model_response_gemma_channel(self) -> None:
        raw = "<|channel>thought\nReasoning about the contract.\n<channel|>Ahmet Yilmaz"
        thinking, answer = _parse_model_response(raw)
        assert thinking == "Reasoning about the contract."
        assert answer == "Ahmet Yilmaz"

    def test_parse_model_response_think_tags(self) -> None:
        raw = "<think>Calculating rental value</think>25.000 TL"
        thinking, answer = _parse_model_response(raw)
        assert thinking == "Calculating rental value"
        assert answer == "25.000 TL"

    def test_parse_model_response_no_thinking(self) -> None:
        raw = "Direct Answer Without Thinking"
        thinking, answer = _parse_model_response(raw)
        assert thinking is None
        assert answer == "Direct Answer Without Thinking"

    def test_parse_model_response_variant_channel_tags(self) -> None:
        # Variant with leading pipe: <|channel|>
        raw1 = "<|channel>thought\nReasoning trace.<|channel|>Ahmet Yilmaz"
        th1, ans1 = _parse_model_response(raw1)
        assert th1 == "Reasoning trace."
        assert ans1 == "Ahmet Yilmaz"

        # Variant with closing tag </thought>
        raw2 = "Analiz ediliyor...</thought>25.000 TL"
        th2, ans2 = _parse_model_response(raw2)
        assert th2 == "Analiz ediliyor..."
        assert ans2 == "25.000 TL"

    def test_parse_model_response_unclosed_thought_never_leaks_as_answer(self) -> None:
        # Cutoff mid-thought: raw text must be preserved as thinking, NOT leaked as answer
        cutoff = "Dökümanda kiracı kim diye arıyorum. Metne baktığımda madde 4 içerisinde"
        th, ans = _parse_model_response(cutoff, started_in_thought=True)
        assert th == cutoff
        assert ans == ""  # Never leaked as answer

    def test_parse_model_response_unclosed_thought_with_answer_line(self) -> None:
        # Model concluded with Cevap: but omitted the <channel|> closing token
        raw = (
            "Dökümanda kiracı aranıyor. Belgede kiracı Ahmet Yılmaz olarak görünüyor.\n"
            "Cevap: Ahmet Yılmaz"
        )
        th, ans = _parse_model_response(raw, started_in_thought=True)
        assert th == "Dökümanda kiracı aranıyor. Belgede kiracı Ahmet Yılmaz olarak görünüyor."
        assert ans == "Ahmet Yılmaz"

    def test_sanitize_extracted_value_prefixes(self) -> None:
        assert _sanitize_extracted_value("Cevap: Tolga Arslan") == "Tolga Arslan"
        assert _sanitize_extracted_value("Answer: 15.05.1992") == "15.05.1992"
        assert _sanitize_extracted_value("Sonuç: Ankara") == "Ankara"
        assert _sanitize_extracted_value("Değer: 50.000") == "50.000"

    def test_sanitize_extracted_value_markdown_fences(self) -> None:
        wrapped = "```json\nIstanbul\n```"
        assert _sanitize_extracted_value(wrapped) == "Istanbul"

    def test_sanitize_extracted_value_control_tokens(self) -> None:
        raw_with_tokens = "<|channel>text\nAhmet Yılmaz<turn|>"
        assert _sanitize_extracted_value(raw_with_tokens) == "Ahmet Yılmaz"

    def test_is_not_found_value(self) -> None:
        assert _is_not_found_value("Bulunamadı") is True
        assert _is_not_found_value("Bulunamadi") is True
        assert _is_not_found_value("Yok") is True
        assert _is_not_found_value("N/A") is True
        assert _is_not_found_value("None") is True
        assert _is_not_found_value("Ahmet Yilmaz") is False

    def test_build_extraction_prompt_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from docneedle.config import settings

        # Verify Mistral / Ministral format
        monkeypatch.setattr(settings, "llm_model_name", "Ministral-3-3B-Instruct-2512-Q4_K_M")
        prompt = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=False
        )
        assert "[SYSTEM_PROMPT]" in prompt
        assert "[INST]" in prompt
        assert "[/INST]" in prompt
        assert "Kira: 1000 TL" in prompt

        # Verify explicit thinking mode in Ministral format
        prompt_thinking = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=True
        )
        assert "<think>\n" in prompt_thinking

        # Verify Gemma turn format
        monkeypatch.setattr(settings, "llm_model_name", "gemma-4-E2B-it-qat-UD-Q4_K_XL")
        prompt_gemma = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=False
        )
        assert "<|turn>system\n" in prompt_gemma
        assert "<|turn>user\n" in prompt_gemma
        assert "<|turn>model\n" in prompt_gemma

        prompt_thinking_gemma = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=True
        )
        assert "<|channel>thought\n" in prompt_thinking_gemma

    def test_build_extraction_prompt_with_type_guidelines(self) -> None:
        prompt = _build_extraction_prompt(
            context="Kira: 25.000 TL",
            query="Kira bedeli?",
            expected_type="money",
        )
        assert "Expected Data Type: money" in prompt
        assert "Monetary amount" in prompt
        assert "25.000 TL" in prompt


class TestStoreDiversityHelpers:
    """Test document diversity and ranking logic in SessionVectorStore."""

    def test_apply_document_diversity_limits_per_document(self) -> None:
        def make_sr(chunk_id: str, doc_name: str, score: float) -> SearchResult:
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_name,
                doc_name=doc_name,
                chunk_index=0,
                text=f"Text for {chunk_id}",
            )
            return SearchResult(chunk=chunk, score=score)

        candidates = [
            make_sr("c1_1", "doc1.pdf", 0.9),
            make_sr("c1_2", "doc1.pdf", 0.8),
            make_sr("c1_3", "doc1.pdf", 0.7),  # Should be skipped by max_per_doc=2
            make_sr("c2_1", "doc2.pdf", 0.6),
            make_sr("c3_1", "doc3.pdf", 0.5),
        ]

        diversified = _apply_document_diversity(candidates, top_k=3, max_per_doc=2)
        assert len(diversified) == 3
        chunk_ids = [r.chunk.chunk_id for r in diversified]
        assert chunk_ids == ["c1_1", "c1_2", "c2_1"]

    def test_apply_document_diversity_backfill(self) -> None:
        def make_sr(chunk_id: str, doc_name: str, score: float) -> SearchResult:
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_name,
                doc_name=doc_name,
                chunk_index=0,
                text=f"Text for {chunk_id}",
            )
            return SearchResult(chunk=chunk, score=score)

        # Only one document with 3 chunks, top_k=3, max_per_doc=1
        candidates = [
            make_sr("c1_1", "doc1.pdf", 0.9),
            make_sr("c1_2", "doc1.pdf", 0.8),
            make_sr("c1_3", "doc1.pdf", 0.7),
        ]

        diversified = _apply_document_diversity(candidates, top_k=3, max_per_doc=1)
        # Should take c1_1 first, then backfill c1_2 and c1_3 to reach top_k=3
        assert len(diversified) == 3
        assert [r.chunk.chunk_id for r in diversified] == ["c1_1", "c1_2", "c1_3"]


class TestExtractionQueryBuilder:
    """Test query synthesis for RAG retrieval."""

    def test_placeholder_key_uses_only_description(self) -> None:
        assert _build_retrieval_query("madde_1", "kiracı kimdir?") == "kiracı kimdir?"
        assert _build_retrieval_query("item_2", "kira bedeli") == "kira bedeli"
        assert _build_retrieval_query("field_3", "adres bilgisi") == "adres bilgisi"

    def test_descriptive_key_combines_with_description(self) -> None:
        query = _build_retrieval_query("kiraci", "Sözleşmedeki kiracı kimdir?")
        assert query == "Sözleşmedeki kiracı kimdir? (kiraci)"

        query2 = _build_retrieval_query("telefon_no", "İletişim numarası")
        assert query2 == "İletişim numarası (telefon no)"

    def test_none_description_returns_clean_key(self) -> None:
        assert _build_retrieval_query("kira_bedeli", None) == "kira bedeli"


class TestChunkerCoalescenceHelpers:
    """Test DoclingChunker chunk coalescing and merging helpers."""

    def test_should_merge_same_page_and_small(self) -> None:
        c1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=0,
            text="Kısa metin bir.",
            page_number=1,
            metadata={"pages": [1]},
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=1,
            text="Kısa metin iki.",
            page_number=1,
            metadata={"pages": [1]},
        )
        assert (
            DoclingChunker._should_merge(c1, c2, max_chunk_chars=1000, min_split_chars=200) is True
        )

    def test_should_not_merge_different_pages(self) -> None:
        c1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=0,
            text="Metin sayfa bir.",
            page_number=1,
            metadata={"pages": [1]},
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=1,
            text="Metin sayfa iki.",
            page_number=2,
            metadata={"pages": [2]},
        )
        assert (
            DoclingChunker._should_merge(c1, c2, max_chunk_chars=1000, min_split_chars=200) is False
        )

    def test_merge_chunk_pair_merges_metadata(self) -> None:
        c1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=0,
            text="Başlık",
            heading="Bölüm 1",
            page_number=1,
            metadata={"headings": ["Bölüm 1"], "pages": [1]},
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=1,
            text="İçerik",
            heading="Bölüm 2",
            page_number=1,
            metadata={"headings": ["Bölüm 2"], "pages": [1]},
        )
        merged = DoclingChunker._merge_chunk_pair(c1, c2, chunk_index=0)
        assert merged.text == "Başlık\n\nİçerik"
        assert merged.metadata["headings"] == ["Bölüm 1", "Bölüm 2"]
        assert merged.metadata["pages"] == [1]

    def test_merge_chunk_pair_deduplicates_same_heading(self) -> None:
        c1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=0,
            text="Bölüm 1\nMetin A",
            heading="Bölüm 1",
            page_number=1,
            metadata={"headings": ["Bölüm 1"], "pages": [1]},
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="f.md",
            chunk_index=1,
            text="Bölüm 1\nMetin B",
            heading="Bölüm 1",
            page_number=1,
            metadata={"headings": ["Bölüm 1"], "pages": [1]},
        )
        merged = DoclingChunker._merge_chunk_pair(c1, c2, chunk_index=0)
        assert merged.text == "Bölüm 1\nMetin A\n\nMetin B"


class TestEvidenceAndVerificationHelpers:
    """Test dynamic evidence resolution and graded confidence scoring."""

    def test_find_best_supporting_match_finds_correct_chunk(self) -> None:
        chunk1 = DocumentChunk(
            chunk_id="c1",
            doc_id="doc1",
            doc_name="genel_bilgiler.pdf",
            chunk_index=0,
            text="DocNeedle genel tanıtım metni.",
        )
        chunk2 = DocumentChunk(
            chunk_id="c2",
            doc_id="doc2",
            doc_name="sozlesme.pdf",
            chunk_index=0,
            text="Aylık kira bedeli 25.000 TL olarak belirlenmiştir.",
        )
        chunk3 = DocumentChunk(
            chunk_id="c3",
            doc_id="doc3",
            doc_name="ek.pdf",
            chunk_index=0,
            text="Ek protokol detayları.",
        )

        matches = [
            SearchResult(chunk=chunk1, score=0.9),  # top ranked but does not contain value
            SearchResult(chunk=chunk2, score=0.8),  # 2nd chunk contains 25.000 TL
            SearchResult(chunk=chunk3, score=0.7),
        ]

        best = _find_best_supporting_match("25.000 TL", matches)
        assert best.chunk.chunk_id == "c2"
        assert best.chunk.doc_name == "sozlesme.pdf"

    def test_confidence_calculator_alignment_and_type_check(self) -> None:
        from docneedle.extraction.verifier import ConfidenceCalculator

        # Exact match with valid money format
        exact = ConfidenceCalculator.calculate(
            found=True,
            value="25.000 TL",
            retrieval_score=0.5,
            target_chunk_text="Kira bedeli 25.000 TL dir.",
            expected_type="money",
        )
        assert exact >= 0.90

        # Type mismatch: expected money but returned string without digits
        mismatch = ConfidenceCalculator.calculate(
            found=True,
            value="Belirtilmedi",
            retrieval_score=0.1,
            target_chunk_text="Kira bedeli hakkında bir bilgi yok.",
            expected_type="money",
        )
        assert mismatch < 0.80


class TestConfigurableSettings:
    """Test that central settings cover all subsystem parameters."""

    def test_settings_fields_exist(self) -> None:
        from docneedle.config import settings

        # Ingest & Parser
        assert hasattr(settings, "ocr_pdf_render_scale")
        assert hasattr(settings, "pdf_embedded_text_min_chars")
        assert hasattr(settings, "ocr_fallback_min_chars")
        assert hasattr(settings, "ocr_model_name")
        assert hasattr(settings, "ocr_text_score")
        assert hasattr(settings, "ocr_longest_edge")

        # Chunker
        assert hasattr(settings, "chunker_tokenizer_model")
        assert hasattr(settings, "chunker_max_tokens")
        assert hasattr(settings, "chunker_max_chunk_chars")
        assert hasattr(settings, "chunker_min_split_chars")
        assert hasattr(settings, "chunker_overflow_factor")

        # BM25
        assert hasattr(settings, "bm25_k1")
        assert hasattr(settings, "bm25_b")
        assert hasattr(settings, "bm25_top_k")

        # Embedder & Reranker
        assert hasattr(settings, "embedding_batch_size")
        assert hasattr(settings, "embedding_instruction")
        assert hasattr(settings, "reranker_instruction")
        assert hasattr(settings, "retrieval_top_k")
        assert hasattr(settings, "rrf_constant")

        # LLM & Extraction
        assert hasattr(settings, "extraction_top_k")
        assert hasattr(settings, "extraction_context_delimiter")
        assert hasattr(settings, "llm_stop_tokens")
        assert hasattr(settings, "confidence_found_score")
        assert hasattr(settings, "confidence_not_found_score")

    def test_confidence_calculator_respects_settings(self) -> None:
        from docneedle.config import settings
        from docneedle.extraction.verifier import ConfidenceCalculator

        score = ConfidenceCalculator.calculate(found=True, value="test")
        assert score == settings.confidence_found_score

        zero_score = ConfidenceCalculator.calculate(found=False, value=None)
        assert zero_score == settings.confidence_not_found_score


class TestEphemeralRapidOCREngine:
    """Test ephemeral RapidOCR lifecycle, transcription, and memory reclamation."""

    def test_requires_context_manager(self) -> None:
        from PIL import Image

        from docneedle.ingest.ocr_engine import EphemeralRapidOCREngine

        engine = EphemeralRapidOCREngine()
        img = Image.new("RGB", (100, 100), color="white")
        with pytest.raises(RuntimeError, match="must be used within a 'with' block"):
            engine.transcribe_image(img)

    def test_ephemeral_engine_lifecycle_and_transcription(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from PIL import Image

        from docneedle.ingest.ocr_engine import EphemeralRapidOCREngine

        mock_output = MagicMock()
        mock_output.txts = ("FATURA", "Toplam Tutar: 15.000 TL")
        mock_output.to_markdown.return_value = "FATURA\n\nToplam Tutar: 15.000 TL"

        mock_rapid_inst = MagicMock(return_value=mock_output)
        mock_rapid_cls = MagicMock(return_value=mock_rapid_inst)

        monkeypatch.setattr("rapidocr.RapidOCR", mock_rapid_cls)

        img = Image.new("RGB", (100, 100), color="white")

        engine = EphemeralRapidOCREngine(text_score=0.6)
        assert engine.ocr is None

        with engine as ocr:
            assert ocr.ocr is not None
            text = ocr.transcribe_image(img)
            assert text == "FATURA\n\nToplam Tutar: 15.000 TL"
            mock_rapid_inst.assert_called_once()

        # After exiting with block, engine reference must be cleared
        assert engine.ocr is None


class TestCodeReviewRegressionFixes:
    """Regression tests verifying logic and accuracy bug fixes."""

    def test_is_not_found_value_preserves_boolean_false(self) -> None:
        assert _is_not_found_value("False", expected_type="boolean") is False
        assert _is_not_found_value("false", expected_type="boolean") is False
        assert _is_not_found_value("Not found", expected_type="boolean") is True
        assert _is_not_found_value("False", expected_type="string") is True

    def test_sanitize_extracted_value_prefix_before_code_fence(self) -> None:
        raw = "Cevap: ```json\n25.000 TL\n```"
        assert _sanitize_extracted_value(raw) == "25.000 TL"

    def test_turkish_dotted_i_evidence_and_confidence_matching(self) -> None:
        from docneedle.extraction.verifier import ConfidenceCalculator

        chunk1 = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=0,
            text="Genel sözleşme maddeleri.",
        )
        chunk2 = DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_name="doc.pdf",
            chunk_index=1,
            text="FATURA TARİHİ: 15 HAZİRAN 2026 - ŞEHİR: ESKISEHIR",
        )
        matches = [
            SearchResult(chunk=chunk1, score=0.9),
            SearchResult(chunk=chunk2, score=0.8),
        ]

        # Uppercase 'HAZİRAN' in chunk vs Titlecase '15 Haziran 2026' in value
        best = _find_best_supporting_match("15 Haziran 2026", matches)
        assert best.chunk.chunk_id == "c2"

        conf = ConfidenceCalculator.calculate(
            found=True,
            value="15 Haziran 2026",
            retrieval_score=0.5,
            target_chunk_text=chunk2.text,
            expected_type="date",
        )
        assert conf >= 0.95

        # Diacritic difference: 'ESKISEHIR' in chunk vs 'Eskişehir' in value
        best_city = _find_best_supporting_match("Eskişehir", matches)
        assert best_city.chunk.chunk_id == "c2"

    def test_chunker_extracts_page_numbers_from_sayfa_headings(self) -> None:
        from docneedle.ingest.parser import DoclingParser

        md_bytes = (
            "# Sözleşme\n\n"
            "## Sayfa 1\n\n"
            "Birinci sayfa metni burada yer almaktadır.\n\n"
            "## Sayfa 2\n\n"
            "İkinci sayfa metni burada yer almaktadır.\n"
        ).encode("utf-8")

        parser = DoclingParser(ocr_enabled=False)
        doc, fname = parser.parse_bytes(md_bytes, filename="test.md")
        chunker = DoclingChunker()
        chunks = chunker.chunk_document(doc, doc_id="doc_test", doc_name=fname)

        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2

    def test_ingestion_cache_hit_updates_doc_id_doc_name_and_chunk_ids(self) -> None:
        from docneedle.ingest.service import IngestionService

        svc = IngestionService()
        content = b"# Belge\n\n## Sayfa 1\n\nOrnek icerik metni burada."

        res1 = svc.ingest_bytes(content=content, filename="birinci.md", doc_id="doc_111")
        res2 = svc.ingest_bytes(content=content, filename="ikinci.md", doc_id="doc_222")

        assert res1.doc_id == "doc_111"
        assert res2.doc_id == "doc_222"
        assert res2.doc_name == "ikinci.md"
        assert res2.chunks[0].doc_id == "doc_222"
        assert res2.chunks[0].doc_name == "ikinci.md"
        assert res2.chunks[0].chunk_id != res1.chunks[0].chunk_id
        assert res2.chunks[0].chunk_id.startswith("doc_222_")

    def test_gradio_ui_html_escaping_and_type_mapping(self) -> None:
        import sys
        from pathlib import Path

        from docneedle.main import app as fastapi_app

        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        orig_routes = list(fastapi_app.router.routes)
        try:
            from app import FIELD_TYPE_KEYS, render_results_html

            assert "time" in FIELD_TYPE_KEYS
            assert "percentage" in FIELD_TYPE_KEYS

            sample_resp = {
                "total_items": 1,
                "found_items": 1,
                "processing_time_ms": 120.0,
                "results": [
                    {
                        "key": "email_<test>",
                        "found": True,
                        "value": "John Doe <john@example.com>",
                        "confidence": 0.96,
                        "evidence": {
                            "document": "doc_<1>.pdf",
                            "page": 2,
                            "snippet": "Contact: <john@example.com>",
                        },
                        "thinking": "Checking <email> tag",
                    }
                ],
            }
            rendered = render_results_html(sample_resp)
            assert "&lt;john@example.com&gt;" in rendered
            assert "&lt;email&gt;" in rendered
            assert "• Page 2" in rendered
        finally:
            fastapi_app.router.routes = orig_routes

    def test_docling_parser_preserves_multiline_formatting(self) -> None:
        from docneedle.ingest.chunker import DoclingChunker
        from docneedle.ingest.parser import DoclingParser

        multiline_content = (
            b"KIMLIK BILGILERI\n"
            b"Kimlik No: 12757385200\n"
            b"Adi: HUSEYIN\n"
            b"Soyadi: ABAZ\n"
            b"Dogum Tarihi: 04.05.2004\n"
        )
        parser = DoclingParser(ocr_enabled=False)
        doc, fname = parser.parse_bytes(multiline_content, filename="kimlik.txt")
        chunker = DoclingChunker()
        chunks = chunker.chunk_document(doc, doc_id="test_doc", doc_name=fname)

        assert len(chunks) > 0
        chunk_text = chunks[0].text
        # Must retain separate lines instead of gluing everything with spaces
        assert "Kimlik No: 12757385200\n" in chunk_text or "Kimlik No: 12757385200" in chunk_text
        assert "Adi: HUSEYIN\n" in chunk_text or "Adi: HUSEYIN" in chunk_text
        assert "HUSEYIN ABAZ 04.05.2004" not in chunk_text

    def test_format_ocr_layout_separates_adjacent_table_columns(self) -> None:
        import numpy as np

        from docneedle.ingest.ocr_engine import format_ocr_layout

        boxes = np.array(
            [
                [[10, 10], [90, 10], [90, 30], [10, 30]],  # "12757385200"
                [[105, 10], [210, 10], [210, 30], [105, 30]],  # "Hüseyin ABAZ" (gap=15)
                [[225, 10], [270, 10], [270, 30], [225, 30]],  # "Ömer" (gap=15)
                [[285, 10], [330, 10], [330, 30], [285, 30]],  # "Ayşe" (gap=15)
                [[345, 10], [390, 10], [390, 30], [345, 30]],  # "Erkek" (gap=15)
            ],
            dtype=np.float32,
        )
        txts = ("12757385200", "Hüseyin ABAZ", "Ömer", "Ayşe", "Erkek")
        layout = format_ocr_layout(boxes, txts)

        # Columns must be distinctly separated with ' | ' rather than glued with spaces
        assert "12757385200 | Hüseyin ABAZ" in layout
        assert "Hüseyin ABAZ | Ömer" in layout
        assert "Ömer | Ayşe" in layout
        assert "Ayşe | Erkek" in layout

    def test_confidence_calculator_new_type_validation(self) -> None:
        from docneedle.extraction.verifier import ConfidenceCalculator

        # Valid email
        c_email_valid = ConfidenceCalculator.calculate(
            found=True,
            value="user@example.com",
            target_chunk_text="Iletisim: user@example.com",
            expected_type="email",
        )
        assert c_email_valid >= 0.90

        # Invalid email (missing @ or .)
        c_email_invalid = ConfidenceCalculator.calculate(
            found=True,
            value="Ahmet Yilmaz",
            target_chunk_text="Iletisim: Ahmet Yilmaz",
            expected_type="email",
        )
        assert c_email_invalid <= 0.80

        # Boolean False should not be unfairly penalized
        c_bool_false = ConfidenceCalculator.calculate(
            found=True,
            value="False",
            target_chunk_text="Devir ve alt kiraya verme hakki taninmamistir.",
            expected_type="boolean",
        )
        assert c_bool_false >= 0.85

        # Short token matching requires word boundary
        c_short_boundary = ConfidenceCalculator.calculate(
            found=True,
            value="1",
            target_chunk_text="Madde 1: Sozlesme baslangic sartlari.",
            expected_type="number",
        )
        assert c_short_boundary >= 0.90


class TestRefactoredExtractionHelpers:
    """Test unit-level helpers extracted during refactoring."""

    def test_truncate_snippet_behaviors(self) -> None:
        from docneedle.extraction.service import _truncate_snippet

        assert _truncate_snippet("") is None
        assert _truncate_snippet("   ") is None

        short_text = "Kira bedeli 25.000 TL dir."
        assert _truncate_snippet(short_text, max_chars=100) == short_text

        long_text = "Kelime " * 70  # 490 chars
        truncated = _truncate_snippet(long_text, max_chars=100)
        assert truncated is not None
        assert truncated.endswith("...")
        assert len(truncated) <= 103
        assert not truncated[:-3].endswith(" ")

    def test_normalize_for_match_unified(self) -> None:
        from docneedle.extraction.verifier import normalize_for_match

        # Turkish dotted capital I and regular I
        assert normalize_for_match("İSTANBUL") == "istanbul"
        assert normalize_for_match("Isparta") == "ısparta"
        # Combining dot
        assert normalize_for_match("i\u0307") == "i"

    def test_build_empty_item_result(self) -> None:
        from docneedle.extraction.service import ExtractionService

        res = ExtractionService._build_empty_item_result("not_found_key")
        assert res.key == "not_found_key"
        assert res.found is False
        assert res.value is None
        assert res.confidence == 0.0
        assert res.evidence is None

    def test_create_evidence_and_confidence(self) -> None:
        from docneedle.extraction.service import ExtractionService

        chunk = DocumentChunk(
            chunk_id="c_test",
            doc_id="d_test",
            doc_name="test.pdf",
            chunk_index=0,
            text="Firma unvanı: ABC Bilişim A.Ş.",
            page_number=3,
        )
        match = SearchResult(chunk=chunk, score=0.95)
        evidence, conf = ExtractionService._create_evidence_and_confidence(
            raw_val="ABC Bilişim A.Ş.",
            matches=[match],
            expected_type="organization",
        )
        assert evidence.document == "test.pdf"
        assert evidence.page == 3
        assert evidence.snippet == "Firma unvanı: ABC Bilişim A.Ş."
        assert conf >= 0.90
