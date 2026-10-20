"""Data models for search and retrieval."""

from pydantic import BaseModel

from docneedle.ingest.models import DocumentChunk


class SearchResult(BaseModel):
    """Result of hybrid search query matching a chunk."""

    chunk: DocumentChunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
