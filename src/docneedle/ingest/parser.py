"""High-fidelity document parser supporting PDF, DOCX, XLSX, HTML, TXT, and images.

Pure OCR-free architecture: Scanned PDF pages and images are preserved as high-resolution
visual page chunks with raw image bytes in memory for multimodal retrieval and extraction.
"""

import io
import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument
from PIL import Image

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
                has_garbled_sym = any(sym in tok for sym in _GARBLED_MIDWORD_SYMBOLS) and any(
                    c.isalnum() for c in tok
                )
                if has_garbled_sym or _CONSONANT_RUN_RE.search(tok):
                    garbled_tokens += 1
        if garbled_tokens / len(tokens) > GARBLED_TOKEN_RATIO_THRESHOLD:
            return False

    return True


_PAGE_HEADING_LINE_RE = re.compile(r"^##\s+Sayfa\s+\d+\s*$")


def _format_clean_lines(text: str) -> str:
    """Strip blank lines and join non-empty lines with double newlines for markdown blocks."""
    return "\n\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _extract_markdown_body(md_text: str) -> str:
    """Return markdown body text excluding top-level document title and page headings."""
    body_lines = [
        ln
        for ln in md_text.splitlines()
        if not ln.startswith("# ") and not _PAGE_HEADING_LINE_RE.match(ln)
    ]
    return "\n".join(body_lines).strip()


def _markdown_to_docling_document(
    markdown_content: str,
    title: str,
) -> DoclingDocument:
    """Convert markdown string into a DoclingDocument preserving hierarchy and lines."""
    doc = DoclingDocument(name=title)
    if title:
        doc.add_title(text=title)

    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            doc.add_text(text="\n".join(current_lines), label="text")
            current_lines = []

    for raw_line in markdown_content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            h_text = stripped.lstrip("#").strip()
            doc.add_heading(text=h_text, level=min(level, 6))
        else:
            current_lines.append(line)
    flush()
    return doc


def _render_pdf_page_image(page: Any, scale: float | None = None) -> bytes:
    """Render a single PDF page to JPEG byte stream in memory."""
    s = settings.pdf_render_scale if scale is None else scale
    bitmap = page.render(scale=s)
    try:
        pil_img = bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()

    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    finally:
        pil_img.close()


class DoclingParser:
    """Handles high-fidelity parsing of documents across PDF, DOCX, XLSX, HTML, TXT, and images.

    Completely OCR-free: Scanned PDF pages and images are ingested as visual page chunks
    preserving in-memory JPEG bytes for multimodal embedding and LLM reasoning.
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "DoclingParser":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def _parse_pdf(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Extract digital text or render high-resolution visual page chunks."""
        import pypdfium2

        title = doc_title or path_obj.stem
        pdf = pypdfium2.PdfDocument(path_obj)
        page_images: dict[int, bytes] = {}
        try:
            md_parts = [f"# {title}\n"]
            for i in range(len(pdf)):
                page_idx = i + 1
                page = pdf[i]
                try:
                    text_page = page.get_textpage()
                    try:
                        text = text_page.get_text_range().strip()
                    finally:
                        text_page.close()

                    if text and _is_extracted_text_readable(
                        text, min_chars=settings.ocr_fallback_min_chars
                    ):
                        page_content = _format_clean_lines(text)
                        md_parts.append(f"## Sayfa {page_idx}\n\n{page_content}\n")
                    else:
                        logger.info(
                            "--> [Visual Page] Page %d/%d of '%s' is scanned; preserving image.",
                            page_idx,
                            len(pdf),
                            title,
                        )
                        img_bytes = _render_pdf_page_image(page)
                        page_images[page_idx] = img_bytes
                        md_parts.append(
                            f"## Sayfa {page_idx}\n\n[Görsel Belge Sayfası {page_idx} - {title}]\n"
                        )
                finally:
                    page.close()
        finally:
            pdf.close()

        doc = _markdown_to_docling_document(
            markdown_content="\n".join(md_parts),
            title=title,
        )
        return doc, page_images

    def _parse_image(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Ingest an image file as a visual document chunk preserving raw RGB bytes."""
        title = doc_title or path_obj.stem
        with Image.open(path_obj) as img:
            rgb_img = img.convert("RGB")
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=92)
            img_bytes = buf.getvalue()

        page_images: dict[int, bytes] = {1: img_bytes}
        full_md = f"# {title}\n\n## Sayfa 1\n\n[Görsel Doküman - {title}]\n"
        doc = _markdown_to_docling_document(
            markdown_content=full_md,
            title=title,
        )
        return doc, page_images

    def _parse_docx(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Extract structured text and tables from a DOCX file."""
        import docx

        title = doc_title or path_obj.stem
        doc = docx.Document(path_obj)
        md_parts = [f"# {title}\n"]

        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            style_name = getattr(p.style, "name", "")
            if "Heading" in style_name:
                md_parts.append(f"## {txt}\n")
            else:
                md_parts.append(f"{txt}\n")

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                md_parts.append("| " + " | ".join(cells) + " |")
            md_parts.append("")

        return _markdown_to_docling_document("\n".join(md_parts), title=title), {}

    def _parse_xlsx(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Extract structured sheets and tables from an XLSX file."""
        import openpyxl

        title = doc_title or path_obj.stem
        wb = openpyxl.load_workbook(path_obj, data_only=True)
        md_parts = [f"# {title}\n"]

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            md_parts.append(f"## {sheet_name}\n")
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header_written = False
            for row in rows:
                cell_strs = [str(c) if c is not None else "" for c in row]
                if not any(cell_strs):
                    continue
                md_parts.append("| " + " | ".join(cell_strs) + " |")
                if not header_written:
                    md_parts.append("| " + " | ".join(["---"] * len(cell_strs)) + " |")
                    header_written = True
            md_parts.append("")

        return _markdown_to_docling_document("\n".join(md_parts), title=title), {}

    def _parse_html(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Extract structured text from an HTML file."""
        from bs4 import BeautifulSoup

        title = doc_title or path_obj.stem
        html_text = path_obj.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html_text, "html.parser")
        title_tag = soup.find("title")
        t = title_tag.get_text().strip() if title_tag else title
        md_parts = [f"# {t}\n"]

        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
            txt = tag.get_text(separator=" ", strip=True)
            if not txt:
                continue
            if tag.name in ("h1", "h2", "h3", "h4"):
                lvl = int(tag.name[1])
                md_parts.append(f"{'#' * min(lvl + 1, 6)} {txt}\n")
            else:
                md_parts.append(f"{txt}\n")

        return _markdown_to_docling_document("\n".join(md_parts), title=title), {}

    def _parse_text(
        self, path_obj: Path, doc_title: str | None = None
    ) -> tuple[DoclingDocument, dict[int, bytes]]:
        """Extract text from plain text or markdown files."""
        title = doc_title or path_obj.stem
        text = path_obj.read_text(encoding="utf-8", errors="ignore")
        return _markdown_to_docling_document(text, title=title), {}

    def parse_path(
        self, path: Path | str, display_name: str | None = None
    ) -> tuple[DoclingDocument, str, dict[int, bytes]]:
        """Parse a document file from a filesystem path, extracting text and visual page images."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Document file does not exist: {path_obj}")

        filename = display_name or path_obj.name
        doc_title = Path(filename).stem
        suffix = path_obj.suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            logger.info(f"Image '{filename}' detected; loading as multimodal visual chunk.")
            doc, images = self._parse_image(path_obj, doc_title=doc_title)
            return doc, filename, images

        if suffix == ".pdf":
            doc, images = self._parse_pdf(path_obj, doc_title=doc_title)
            return doc, filename, images

        if suffix == ".docx":
            doc, images = self._parse_docx(path_obj, doc_title=doc_title)
            return doc, filename, images

        if suffix in {".xlsx", ".xls"}:
            doc, images = self._parse_xlsx(path_obj, doc_title=doc_title)
            return doc, filename, images

        if suffix in {".html", ".htm"}:
            doc, images = self._parse_html(path_obj, doc_title=doc_title)
            return doc, filename, images

        if suffix in {".txt", ".md", ".json", ".csv", ".tsv", ".xml"}:
            doc, images = self._parse_text(path_obj, doc_title=doc_title)
            return doc, filename, images

        # Default fallback: try reading as text, else image
        try:
            doc, images = self._parse_text(path_obj, doc_title=doc_title)
            return doc, filename, images
        except Exception:
            doc, images = self._parse_image(path_obj, doc_title=doc_title)
            return doc, filename, images

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        suffix: str | None = None,
    ) -> tuple[DoclingDocument, str, dict[int, bytes]]:
        """Safely write bytes to a temporary file, parse with Docling, and remove the file."""
        if not suffix:
            suffix = Path(filename).suffix or ".tmp"

        temp_name = f"docneedle_ingest_{uuid.uuid4().hex}{suffix}"
        temp_path = Path(tempfile.gettempdir()) / temp_name

        try:
            temp_path.write_bytes(content)
            doc, actual_name, images = self.parse_path(temp_path, display_name=filename)
            return doc, actual_name, images
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
