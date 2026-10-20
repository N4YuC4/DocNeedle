import io
import logging
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from docneedle.config import settings

logger = logging.getLogger(__name__)

# Readability & CMap validation thresholds
CORRUPTED_CHAR_RATIO_THRESHOLD = 0.03
GARBLED_TOKEN_RATIO_THRESHOLD = 0.08
MIN_GARBLED_TOKEN_LEN = 4

_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{7,}")
_CORRUPTED_CMAP_CHARS = frozenset({"÷", "ø", "Ø", "þ", "Þ", "ð", "Ŧ", "ŧ", "Ŭ", "ŭ", "ŝ", "ŵ", "ů"})
_GARBLED_MIDWORD_SYMBOLS = ("$", "<", ">", "*", "\\", "]")


def _is_extracted_text_readable(text: str, min_chars: int | None = None) -> bool:
    """Validate that digital PDF text is genuinely readable and not a broken ToUnicode CMap."""
    threshold = settings.ocr_fallback_min_chars if min_chars is None else min_chars
    cleaned = text.strip()
    if len(cleaned) < threshold:
        return False

    non_ws = [c for c in cleaned if not c.isspace()]
    if not non_ws:
        return False

    total = len(non_ws)
    corrupted_chars = 0
    for ch in non_ws:
        cp = ord(ch)
        if (
            0x0180 <= cp <= 0x02AF
            or 0x03E0 <= cp <= 0x03EF
            or 0xE000 <= cp <= 0xF8FF
            or cp == 0xFFFD
            or ch in _CORRUPTED_CMAP_CHARS
        ):
            corrupted_chars += 1

    if corrupted_chars / total > CORRUPTED_CHAR_RATIO_THRESHOLD:
        return False

    tokens = cleaned.split()
    if tokens:
        garbled_tokens = 0
        for tok in tokens:
            if len(tok) >= MIN_GARBLED_TOKEN_LEN:
                has_Garbled_sym = any(sym in tok for sym in _GARBLED_MIDWORD_SYMBOLS) and any(
                    c.isalnum() for c in tok
                )
                if has_Garbled_sym or _CONSONANT_RUN_RE.search(tok):
                    garbled_tokens += 1
        if garbled_tokens / len(tokens) > GARBLED_TOKEN_RATIO_THRESHOLD:
            return False

    return True


def _pdf_has_embedded_text(
    path_obj: Path,
    max_check_pages: int | None = None,
    min_chars: int | None = None,
) -> bool:
    """Check if PDF has readable embedded digital text to skip redundant OCR."""
    max_pages = (
        settings.pdf_embedded_text_max_check_pages if max_check_pages is None else max_check_pages
    )
    threshold = settings.pdf_embedded_text_min_chars if min_chars is None else min_chars
    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(path_obj)
        try:
            total_len = 0
            collected: list[str] = []
            pages_to_check = min(len(pdf), max_pages)
            for i in range(pages_to_check):
                page = pdf[i]
                try:
                    text_page = page.get_textpage()
                    try:
                        text = text_page.get_text_range().strip()
                        if text:
                            collected.append(text)
                            total_len += len(text)
                    finally:
                        text_page.close()
                finally:
                    page.close()

            if total_len <= threshold:
                return False

            combined = "\n".join(collected)
            if not _is_extracted_text_readable(combined, min_chars=threshold):
                logger.info(
                    f"PDF '{path_obj.name}' has corrupted embedded font CMap; routing to RapidOCR."
                )
                return False

            return True
        finally:
            pdf.close()
    except Exception as e:
        logger.debug(f"Could not inspect PDF embedded text: {e}")
        return False


def _markdown_to_docling_document(
    markdown_content: str,
    title: str,
    converter: DocumentConverter,
) -> DoclingDocument:
    """Convert an in-memory markdown string into a DoclingDocument via DocumentStream."""
    stream = DocumentStream(
        name=f"{title}.md",
        stream=io.BytesIO(markdown_content.encode("utf-8")),
    )
    return converter.convert(stream).document


class DoclingParser:
    """Handles high-fidelity parsing of documents across PDF, DOCX, XLSX, HTML, and images."""

    def __init__(self, ocr_enabled: bool | None = None) -> None:
        self.ocr_enabled = settings.ocr_enabled if ocr_enabled is None else ocr_enabled
        self._converter_fast: DocumentConverter | None = None
        self._active_ocr_engine: Any = None

    def __enter__(self) -> "DoclingParser":
        from docneedle.ingest.ocr_engine import EphemeralRapidOCREngine

        if self.ocr_enabled and self._active_ocr_engine is None:
            self._active_ocr_engine = EphemeralRapidOCREngine().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._active_ocr_engine is not None:
            try:
                self._active_ocr_engine.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self._active_ocr_engine = None

    @contextmanager
    def _acquire_ocr_engine(self) -> Any:
        """Acquire active OCR engine or create an ephemeral engine for the context."""
        if self._active_ocr_engine is not None:
            yield self._active_ocr_engine
        elif self.ocr_enabled:
            from docneedle.ingest.ocr_engine import EphemeralRapidOCREngine

            with EphemeralRapidOCREngine() as engine:
                yield engine
        else:
            yield None

    def _get_fast_converter(self) -> DocumentConverter:
        """Lazily initialize the DocumentConverter for digital text and markdown."""
        if self._converter_fast is None:
            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_ocr = False
            pipeline_opts.do_table_structure = False
            self._converter_fast = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                    InputFormat.IMAGE: ImageFormatOption(),
                }
            )
        return self._converter_fast

    def _ocr_single_pdf_page(self, page: Any, engine: Any) -> str:
        """Render a single PDF page and transcribe it with the given RapidOCR engine."""
        longest_edge = settings.ocr_longest_edge
        w, h = page.get_size()
        scale = (longest_edge / max(w, h)) if (w > 0 and h > 0) else settings.ocr_pdf_render_scale
        bitmap = page.render(scale=scale)
        try:
            pil_img = bitmap.to_pil().convert("RGB")
        finally:
            bitmap.close()

        try:
            return engine.transcribe_image(pil_img).strip()
        finally:
            pil_img.close()

    def _parse_digital_pdf(self, path_obj: Path, doc_title: str | None = None) -> DoclingDocument:
        """Extract digital text from PDF using pypdfium2 with per-page OCR fallback."""
        import pypdfium2

        from docneedle.ingest.ocr_engine import EphemeralRapidOCREngine

        title = doc_title or path_obj.stem
        pdf = pypdfium2.PdfDocument(path_obj)
        local_ocr_engine: Any = None
        ocr_pages_count = 0
        max_ocr_p = settings.max_ocr_pages
        try:
            md_parts = [f"# {title}\n"]
            for i in range(len(pdf)):
                page = pdf[i]
                try:
                    text_page = page.get_textpage()
                    try:
                        text = text_page.get_text_range().strip()
                    finally:
                        text_page.close()

                    if (
                        not _is_extracted_text_readable(
                            text, min_chars=settings.ocr_fallback_min_chars
                        )
                        and self.ocr_enabled
                    ):
                        can_ocr = max_ocr_p <= 0 or ocr_pages_count < max_ocr_p
                        if can_ocr:
                            engine = self._active_ocr_engine
                            if engine is None:
                                if local_ocr_engine is None:
                                    local_ocr_engine = EphemeralRapidOCREngine().__enter__()
                                engine = local_ocr_engine
                            ocr_text = self._ocr_single_pdf_page(page, engine)
                            ocr_pages_count += 1
                            if ocr_text:
                                text = ocr_text

                    if text:
                        clean_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        page_content = "\n\n".join(clean_lines)
                        md_parts.append(f"## Sayfa {i + 1}\n\n{page_content}\n")
                finally:
                    page.close()
        finally:
            if local_ocr_engine is not None:
                local_ocr_engine.__exit__(None, None, None)
            pdf.close()

        return _markdown_to_docling_document(
            markdown_content="\n".join(md_parts),
            title=title,
            converter=self._get_fast_converter(),
        )

    def _parse_scanned_pdf(self, path_obj: Path, doc_title: str | None = None) -> DoclingDocument:
        """Extract text from scanned PDF pages using pypdfium2 and ephemeral RapidOCR."""
        import pypdfium2

        title = doc_title or path_obj.stem
        pdf = pypdfium2.PdfDocument(path_obj)
        try:
            total_pages = len(pdf)
            max_p = settings.max_ocr_pages
            pages_to_ocr = min(total_pages, max_p) if max_p > 0 else total_pages

            md_parts = [f"# {title}\n"]
            with self._acquire_ocr_engine() as engine:
                for page_idx in range(pages_to_ocr):
                    page = pdf[page_idx]
                    try:
                        page_text = self._ocr_single_pdf_page(page, engine)
                        if page_text:
                            clean_lines = [
                                ln.strip() for ln in page_text.splitlines() if ln.strip()
                            ]
                            page_content = "\n\n".join(clean_lines)
                            md_parts.append(f"## Sayfa {page_idx + 1}\n\n{page_content}\n")
                    finally:
                        page.close()
        finally:
            pdf.close()

        return _markdown_to_docling_document(
            markdown_content="\n".join(md_parts),
            title=title,
            converter=self._get_fast_converter(),
        )

    def _parse_image(self, path_obj: Path, doc_title: str | None = None) -> DoclingDocument:
        """Extract text from an image file using ephemeral RapidOCR."""
        from PIL import Image

        title = doc_title or path_obj.stem
        with Image.open(path_obj) as img:
            rgb_img = img.convert("RGB")

        with self._acquire_ocr_engine() as engine:
            page_text = engine.transcribe_image(rgb_img)

        clean_lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        page_content = "\n\n".join(clean_lines)
        full_md = f"# {title}\n\n{page_content}\n"
        return _markdown_to_docling_document(
            markdown_content=full_md,
            title=title,
            converter=self._get_fast_converter(),
        )

    def parse_path(
        self, path: Path | str, display_name: str | None = None
    ) -> tuple[DoclingDocument, str]:
        """Parse a document file from a filesystem path."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Document file does not exist: {path_obj}")

        filename = display_name or path_obj.name
        doc_title = Path(filename).stem
        suffix = path_obj.suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            if self.ocr_enabled:
                logger.info(f"Image '{filename}' detected; running ephemeral RapidOCR.")
                return self._parse_image(path_obj, doc_title=doc_title), filename
            else:
                conv = self._get_fast_converter()
                return conv.convert(path_obj).document, filename

        if suffix == ".pdf":
            has_text = _pdf_has_embedded_text(path_obj)
            if not has_text and self.ocr_enabled:
                logger.info(
                    f"PDF '{filename}' is scanned or unreadable; running ephemeral RapidOCR."
                )
                return self._parse_scanned_pdf(path_obj, doc_title=doc_title), filename
            else:
                logger.info(f"PDF '{filename}' contains digital text; using fast parser.")
                doc = self._parse_digital_pdf(path_obj, doc_title=doc_title)
                md_text = doc.export_to_markdown().strip()
                body_lines = [
                    ln
                    for ln in md_text.splitlines()
                    if not ln.startswith("# ") and not re.match(r"^##\s+Sayfa\s+\d+\s*$", ln)
                ]
                body_text = "\n".join(body_lines).strip()
                md_len = len(body_text)
                if (
                    md_len < settings.ocr_fallback_min_chars
                    or not _is_extracted_text_readable(
                        body_text, min_chars=settings.ocr_fallback_min_chars
                    )
                ) and self.ocr_enabled:
                    logger.info(
                        f"Digital extraction yielded low/garbled text ({md_len} chars); "
                        "running ephemeral RapidOCR."
                    )
                    return self._parse_scanned_pdf(path_obj, doc_title=doc_title), filename
                return doc, filename

        # DOCX, XLSX, HTML, Markdown, etc.
        conv = self._get_fast_converter()
        result = conv.convert(path_obj)
        return result.document, filename

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        suffix: str | None = None,
    ) -> tuple[DoclingDocument, str]:
        """Safely write bytes to a temporary file, parse with Docling, and remove the file."""
        if not suffix:
            suffix = Path(filename).suffix or ".tmp"

        temp_name = f"docneedle_ingest_{uuid.uuid4().hex}{suffix}"
        temp_path = Path(tempfile.gettempdir()) / temp_name

        try:
            temp_path.write_bytes(content)
            doc, _ = self.parse_path(temp_path, display_name=filename)
            return doc, filename
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
