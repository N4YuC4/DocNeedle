"""Unit tests for newly refactored modular helper functions across DocNeedle."""

from pathlib import Path
from typing import Any

import pytest

from docneedle.extraction.service import _build_retrieval_query, _find_best_supporting_match
from docneedle.ingest.chunker import DoclingChunker
from docneedle.ingest.models import DocumentChunk
from docneedle.llm.local_engine import (
    TYPE_GUIDELINES,
    LocalLLMEngine,
    _build_extraction_prompt,
    _is_not_found_value,
    _log_extraction_trace,
    _parse_model_response,
    _probe_template_delimiters,
    _resolve_default_model_path,
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
        # Word-boundary check: words ending with a not-found substring must not be flagged
        assert _is_not_found_value("Kamyon") is False
        assert _is_not_found_value("Piyano") is False
        # Boolean field: "Yok" and "False" are valid negative boolean answers, not missing
        assert _is_not_found_value("False", expected_type="boolean") is False
        assert _is_not_found_value("Yok", expected_type="boolean") is False
        assert _is_not_found_value("Yoktur", expected_type="boolean") is False
        assert _is_not_found_value("Bulunamadı", expected_type="boolean") is True

    def test_build_extraction_prompt_structure(self) -> None:
        # Universal turn format when no chat_formatter is provided
        prompt = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=False
        )
        assert "<|turn>system\n" in prompt
        assert "<|turn>user\n" in prompt
        assert "<|turn>model\n" in prompt
        assert "Kira: 1000 TL" in prompt

        prompt_thinking = _build_extraction_prompt(
            context="Kira: 1000 TL", query="Kira bedeli?", enable_thinking=True
        )
        assert "<|channel>thought\n" in prompt_thinking

        # Dynamic GGUF Jinja2 chat template format (model-agnostic)
        chatml_template = (
            "{% for m in messages %}"
            "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking %}<think>\n{% endif %}{% endif %}"
        )
        formatter, open_tag, close_tag = _probe_template_delimiters(
            chatml_template, bos_token="", eos_token="<|im_end|>"
        )
        dyn_prompt = _build_extraction_prompt(
            context="Kira: 1000 TL",
            query="Kira bedeli?",
            enable_thinking=True,
            chat_formatter=formatter,
            thought_open_tag=open_tag,
        )
        assert "<|im_start|>system\n" in dyn_prompt
        assert "<|im_start|>user\n" in dyn_prompt
        assert "<|im_start|>assistant\n<think>\n" in dyn_prompt
        assert "Kira: 1000 TL" in dyn_prompt

    def test_build_extraction_prompt_truncates_only_context_preserving_question(self) -> None:
        class DummyLLM:
            def tokenize(self, b: bytes) -> list[int]:
                return [1] * len(b.decode("utf-8").split())

            def detokenize(self, toks: list[int]) -> bytes:
                return (" ".join(["kelime"] * len(toks))).encode("utf-8")

        long_ctx = " ".join(f"satir_{i}" for i in range(500))
        prompt = _build_extraction_prompt(
            context=long_ctx,
            query="Kira bedeli nedir?",
            max_tokens=500,
            llm=DummyLLM(),
            enable_thinking=False,
        )
        assert "Question / Target Field: Kira bedeli nedir?" in prompt
        assert prompt.endswith("<turn|>\n<|turn>model\n")
        assert "satir_499" not in prompt

    def test_resolve_default_model_path_strict_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from docneedle.config import settings

        llm_dir = tmp_path / "llm"
        llm_dir.mkdir(parents=True)
        # Put an unrelated GGUF file in the directory; must NOT silently fall back to it
        (llm_dir / "other-unrelated-model.gguf").write_bytes(b"GGUF")

        monkeypatch.setattr(settings, "model_storage_dir", str(tmp_path))
        monkeypatch.setattr(settings, "llm_model_name", "missing-configured-model")

        with pytest.raises(FileNotFoundError, match="missing-configured-model"):
            _resolve_default_model_path()

    def test_log_extraction_trace_redacts_pii_by_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from docneedle.config import settings

        monkeypatch.setattr(settings, "llm_log_extraction_payloads", False)
        with caplog.at_level(logging.DEBUG):
            _log_extraction_trace(
                query_key="tc_kimlik",
                thinking="Gizli dusunce 12345678901",
                value="12345678901",
                found=True,
            )
        assert "12345678901" not in caplog.text

    def test_type_guidelines_examples_have_no_ambiguous_or(self) -> None:
        for key, guide in TYPE_GUIDELINES.items():
            assert " or " not in guide["example"], f"Ambiguous 'or' in example for {key}"

    def test_continuation_respects_context_window_bounds(self) -> None:
        import threading

        class FakeLlama:
            def __init__(self) -> None:
                self.last_max_tokens: int | None = None

            def tokenize(self, b: bytes) -> list[int]:
                # Simulate prompt + thinking already consuming 990 of 1024 tokens
                return [1] * 990

            def __call__(self, **kwargs):
                self.last_max_tokens = kwargs["max_tokens"]
                return {"choices": [{"text": "25.000 TL"}]}

        engine = object.__new__(LocalLLMEngine)
        engine.n_ctx = 1024
        engine.llm = FakeLlama()
        engine._lock = threading.Lock()
        engine._thought_open_tag = "<|channel>thought\n"
        engine._thought_close_tag = "\n<channel|>\n"

        ans = engine._attempt_continuation("prompt", "uzun dusunce", "kira_bedeli")
        assert ans == "25.000 TL"
        assert engine.llm.last_max_tokens is not None
        assert 990 + engine.llm.last_max_tokens <= engine.n_ctx

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
        assert hasattr(settings, "pdf_render_scale")
        assert hasattr(settings, "pdf_embedded_text_min_chars")
        assert hasattr(settings, "pdf_embedded_text_max_check_pages")
        assert hasattr(settings, "ingestion_cache_enabled")

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


class TestVisualPageIngestion:
    """Test OCR-free visual document chunk generation."""

    def test_visual_document_chunk_model(self) -> None:
        from docneedle.ingest.models import DocumentChunk

        chunk = DocumentChunk(
            chunk_id="c_img_1",
            doc_id="d1",
            doc_name="scan.pdf",
            chunk_index=0,
            text="[Görsel Belge Sayfası 1 - scan.pdf]",
            page_number=1,
            image_bytes=b"fake_jpeg_bytes",
        )
        assert chunk.image_bytes == b"fake_jpeg_bytes"
        assert chunk.page_number == 1


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

        parser = DoclingParser()
        doc, fname, _ = parser.parse_bytes(md_bytes, filename="test.md")
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
        parser = DoclingParser()
        doc, fname, _ = parser.parse_bytes(multiline_content, filename="kimlik.txt")
        chunker = DoclingChunker()
        chunks = chunker.chunk_document(doc, doc_id="test_doc", doc_name=fname)

        assert len(chunks) > 0
        chunk_text = chunks[0].text
        # Must retain separate lines instead of gluing everything with spaces
        assert "Kimlik No: 12757385200\n" in chunk_text or "Kimlik No: 12757385200" in chunk_text
        assert "Adi: HUSEYIN\n" in chunk_text or "Adi: HUSEYIN" in chunk_text
        assert "HUSEYIN ABAZ 04.05.2004" not in chunk_text

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


class TestConfigTomlLoading:
    """Test TOML configuration loading and resolution."""

    def test_toml_section_and_flat_key_loading(self, tmp_path) -> None:
        from docneedle.config import (
            DocNeedleTomlConfigSettingsSource,
            Settings,
        )

        test_toml = tmp_path / "custom_config.toml"
        test_toml.write_text(
            """
[server]
port = 9876
app_name = "DocNeedle Test Suite"

[hardware]
gpu_enabled = true
llm_n_gpu_layers = -1

[ingest]
pdf_render_scale = 2.0
pdf_embedded_text_min_chars = 100
"""
        )

        source = DocNeedleTomlConfigSettingsSource(Settings, toml_file=test_toml)
        data = source._read_file(test_toml)

        assert data["port"] == 9876
        assert data["app_name"] == "DocNeedle Test Suite"
        assert data["gpu_enabled"] is True
        assert data["llm_n_gpu_layers"] == -1
        assert data["pdf_render_scale"] == 2.0
        assert data["pdf_embedded_text_min_chars"] == 100

    def test_settings_loads_with_toml_source(self, monkeypatch, tmp_path) -> None:
        from docneedle.config import Settings

        test_toml = tmp_path / "config.toml"
        test_toml.write_text(
            """
[server]
port = 9123
host = "127.0.0.1"

[hardware]
gpu_enabled = true
"""
        )

        monkeypatch.setenv("CONFIG_TOML", str(test_toml))
        s = Settings()
        assert s.port == 9123
        assert s.host == "127.0.0.1"
        assert s.gpu_enabled is True


class TestRefactoredParserAndEmbedderHelpers:
    """Test newly extracted pure helpers in parser, ocr_engine, and embedder."""

    def test_parser_markdown_helpers(self) -> None:
        from docneedle.ingest.parser import _extract_markdown_body, _format_clean_lines

        assert _format_clean_lines("  Line 1 \n\n  Line 2  \n ") == "Line 1\n\nLine 2"
        md = "# Document Title\n\n## Sayfa 1\n\nActual paragraph content.\n## Sayfa 2\nMore text."
        body = _extract_markdown_body(md)
        assert "Document Title" not in body
        assert "Sayfa 1" not in body
        assert "Actual paragraph content." in body
        assert "More text." in body

    def test_embedder_vector_helpers(self) -> None:
        import numpy as np

        from docneedle.retrieval.embedder import _normalize_vector

        raw = np.array([3.0, 4.0], dtype=np.float32)
        vec = _normalize_vector(raw)
        assert vec.shape == (2,)
        assert np.allclose(vec, np.array([0.6, 0.8], dtype=np.float32))


class TestApiValidationHelpers:
    """Test extracted API schema and upload validation functions."""

    def test_parse_extraction_schema_valid_list(self) -> None:
        import json

        from docneedle.api.validation import parse_extraction_schema

        schema_json = json.dumps(
            [{"key": "test_key", "description": "test query", "type": "string"}]
        )
        items = parse_extraction_schema(schema_json)
        assert len(items) == 1
        assert items[0].key == "test_key"
        assert items[0].type == "string"

    def test_parse_extraction_schema_single_dict(self) -> None:
        import json

        from docneedle.api.validation import parse_extraction_schema

        schema_json = json.dumps(
            {"key": "single_key", "description": "single desc", "type": "money"}
        )
        items = parse_extraction_schema(schema_json)
        assert len(items) == 1
        assert items[0].key == "single_key"
        assert items[0].description == "single desc"
        assert items[0].type == "money"

    def test_parse_extraction_schema_invalid_json(self) -> None:
        import pytest
        from fastapi import HTTPException

        from docneedle.api.validation import parse_extraction_schema

        with pytest.raises(HTTPException) as exc_info:
            parse_extraction_schema("not-a-json")
        assert exc_info.value.status_code == 400

    def test_parse_extraction_schema_empty(self) -> None:
        import pytest
        from fastapi import HTTPException

        from docneedle.api.validation import parse_extraction_schema

        with pytest.raises(HTTPException) as exc_info:
            parse_extraction_schema("[]")
        assert exc_info.value.status_code == 400

    def test_validate_file_size_exceeded(self) -> None:
        import pytest
        from fastapi import HTTPException

        from docneedle.api.validation import validate_file_size

        with pytest.raises(HTTPException) as exc_info:
            validate_file_size("large.pdf", b"x" * 100, max_bytes=50)
        assert exc_info.value.status_code == 413

    def test_parse_extraction_schema_exceeds_max_items(self) -> None:
        import json

        import pytest
        from fastapi import HTTPException

        from docneedle.api.validation import parse_extraction_schema
        from docneedle.config import settings

        items = [
            {"key": f"k{i}", "description": f"desc {i}"}
            for i in range(settings.max_schema_items + 1)
        ]
        with pytest.raises(HTTPException) as exc_info:
            parse_extraction_schema(json.dumps(items))
        assert exc_info.value.status_code == 400
        assert "exceeding maximum allowed limit" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_read_and_validate_uploads_exceeds_max_files(self) -> None:
        import io

        import pytest
        from fastapi import HTTPException, UploadFile

        from docneedle.api.validation import read_and_validate_uploads
        from docneedle.config import settings

        excess_files = [
            UploadFile(filename=f"doc_{i}.txt", file=io.BytesIO(b"sample text"))
            for i in range(settings.max_upload_files + 1)
        ]
        with pytest.raises(HTTPException) as exc_info:
            await read_and_validate_uploads(excess_files)
        assert exc_info.value.status_code == 400
        assert "exceeding maximum allowed limit" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_read_and_validate_uploads_exceeds_total_size(self) -> None:
        import io

        import pytest
        from fastapi import HTTPException, UploadFile

        from docneedle.api.validation import BYTES_PER_MEGABYTE, read_and_validate_uploads

        # Three 35MB files (each under 50MB per-file limit, but 105MB > 100MB total limit)
        part_size = 35 * BYTES_PER_MEGABYTE
        files = [
            UploadFile(filename="part1.bin", file=io.BytesIO(b"x" * part_size)),
            UploadFile(filename="part2.bin", file=io.BytesIO(b"y" * part_size)),
            UploadFile(filename="part3.bin", file=io.BytesIO(b"z" * part_size)),
        ]
        with pytest.raises(HTTPException) as exc_info:
            await read_and_validate_uploads(files)
        assert exc_info.value.status_code == 413
        assert "Total upload payload size exceeds maximum allowed limit" in exc_info.value.detail


class TestRerankerStrictNoFallbackAndErrors:
    """Test Qwen3Reranker error classification and strict fail-fast policy."""

    def test_reranker_network_error_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        import urllib.error

        import pytest

        from docneedle.config import settings
        from docneedle.retrieval.reranker import Qwen3Reranker

        monkeypatch.setattr(settings, "model_storage_dir", str(tmp_path))

        def mock_download_fail(models_dir: str) -> None:
            raise urllib.error.URLError("Network connection timed out")

        monkeypatch.setattr("scripts.download_models.download_reranker_model", mock_download_fail)

        with pytest.raises(RuntimeError) as exc_info:
            Qwen3Reranker._resolve_model_path()
        assert "network connectivity error" in str(exc_info.value)
        assert "URLError" in str(exc_info.value)

    def test_reranker_io_error_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        import pytest

        from docneedle.config import settings
        from docneedle.retrieval.reranker import Qwen3Reranker

        monkeypatch.setattr(settings, "model_storage_dir", str(tmp_path))

        def mock_download_fail(models_dir: str) -> None:
            raise PermissionError("Permission denied: /models/reranker")

        monkeypatch.setattr("scripts.download_models.download_reranker_model", mock_download_fail)

        with pytest.raises(RuntimeError) as exc_info:
            Qwen3Reranker._resolve_model_path()
        assert "filesystem/IO error" in str(exc_info.value)
        assert "PermissionError" in str(exc_info.value)

    def test_reranker_unusable_model_fails_loudly(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pytest

        from docneedle.config import settings
        from docneedle.retrieval.reranker import Qwen3Reranker

        # Create a corrupted/invalid model file
        fake_model_dir = tmp_path / "reranker"
        fake_model_dir.mkdir(parents=True)
        fake_model = fake_model_dir / "Qwen3-Reranker-0.6B.Q4_K_M.gguf"
        fake_model.write_bytes(b"corrupted gguf data")

        monkeypatch.setattr(settings, "model_storage_dir", str(tmp_path))

        # Initializing Qwen3Reranker with corrupted file should raise informative RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            Qwen3Reranker(model_path=fake_model)
        assert "failed to initialize or execute in llama.cpp" in str(exc_info.value)

    def test_get_reranker_strict_no_fallback_raises_on_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pytest

        import docneedle.retrieval.reranker as reranker_mod
        from docneedle.retrieval.reranker import get_reranker

        monkeypatch.setattr(reranker_mod, "_reranker_instance", None)

        def mock_init_fail() -> None:
            raise RuntimeError("CUDA out of memory for reranker")

        monkeypatch.setattr(reranker_mod, "Qwen3Reranker", mock_init_fail)

        # Calling get_reranker(force=True) must fail fast and propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            get_reranker(force=True)
        assert "CUDA out of memory for reranker" in str(exc_info.value)

        monkeypatch.setattr(reranker_mod, "Qwen3Reranker", mock_init_fail)

        # Calling get_reranker(force=True) must fail fast and propagate the exception
        with pytest.raises(RuntimeError) as exc_info:
            get_reranker(force=True)
        assert "CUDA out of memory for reranker" in str(exc_info.value)
