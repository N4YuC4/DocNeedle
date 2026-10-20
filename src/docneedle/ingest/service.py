"""High-level ingestion service orchestrating parsing, OCR, and chunking with in-memory caching."""

import hashlib
import logging
import uuid
from pathlib import Path

from docneedle.config import settings
from docneedle.ingest.cache import global_ingestion_cache
from docneedle.ingest.chunker import DoclingChunker
from docneedle.ingest.models import IngestResult
from docneedle.ingest.parser import DoclingParser

logger = logging.getLogger(__name__)


class IngestionService:
    """Service responsible for turning incoming files or byte streams into structured chunks."""

    def __init__(
        self,
        parser: DoclingParser | None = None,
        chunker: DoclingChunker | None = None,
    ) -> None:
        self.parser = parser or DoclingParser()
        self.chunker = chunker or DoclingChunker()

    def ingest_path(
        self,
        path: Path | str,
        doc_id: str | None = None,
        use_cache: bool = True,
    ) -> IngestResult:
        """Parse and chunk a document from a local path, checking in-memory cache first."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Document file does not exist: {path_obj}")

        content = path_obj.read_bytes()
        return self.ingest_bytes(
            content=content,
            filename=path_obj.name,
            doc_id=doc_id,
            use_cache=use_cache,
        )

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        doc_id: str | None = None,
        use_cache: bool = True,
    ) -> IngestResult:
        """Parse and chunk raw file bytes safely without leaving artifacts on disk.

        Caches parsed chunks in-memory by content SHA256 to eliminate redundant OCR.
        """
        actual_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        content_hash = hashlib.sha256(content).hexdigest()

        if use_cache and settings.ingestion_cache_enabled:
            cached = global_ingestion_cache.get(content_hash)
            if cached is not None:
                logger.info(
                    f"[CACHE HIT] Reusing parsed chunks for '{filename}' ({content_hash[:12]}). "
                    "Skipping OCR and re-parsing."
                )
                cloned_chunks = [
                    c.model_copy(
                        deep=True,
                        update={
                            "chunk_id": f"{actual_doc_id}_{c.chunk_index}",
                            "doc_id": actual_doc_id,
                            "doc_name": filename,
                        },
                    )
                    for c in cached.chunks
                ]
                return IngestResult(
                    doc_id=actual_doc_id,
                    doc_name=filename,
                    total_pages=cached.total_pages,
                    total_chunks=len(cloned_chunks),
                    chunks=cloned_chunks,
                )

        document, actual_filename = self.parser.parse_bytes(
            content=content,
            filename=filename,
        )
        chunks = self.chunker.chunk_document(
            document=document,
            doc_id=actual_doc_id,
            doc_name=actual_filename,
        )

        all_pages = {c.page_number for c in chunks if c.page_number is not None}
        total_pages = max(all_pages) if all_pages else None

        result = IngestResult(
            doc_id=actual_doc_id,
            doc_name=actual_filename,
            total_pages=total_pages,
            total_chunks=len(chunks),
            chunks=chunks,
        )

        if use_cache and settings.ingestion_cache_enabled:
            global_ingestion_cache.put(content_hash, result.model_copy(deep=True))

        return result
