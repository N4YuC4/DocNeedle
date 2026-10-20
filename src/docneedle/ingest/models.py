"""Data models for document ingestion, parsing, and chunking."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete semantic chunk extracted from a document."""

    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    page_number: int | None = None
    heading: str | None = None
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Result of parsing and chunking a document."""

    doc_id: str
    doc_name: str
    total_pages: int | None = None
    total_chunks: int = 0
    chunks: list[DocumentChunk] = Field(default_factory=list)
