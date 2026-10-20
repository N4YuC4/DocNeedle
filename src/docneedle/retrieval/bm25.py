"""BM25 sparse keyword retrieval engine with multilingual normalization."""

import re

from rank_bm25 import BM25Okapi

from docneedle.config import settings
from docneedle.ingest.models import DocumentChunk

RAW_STOPWORDS = {
    "ve",
    "veya",
    "ile",
    "bir",
    "bu",
    "şu",
    "o",
    "de",
    "da",
    "ki",
    "icin",
    "için",
    "ne",
    "nasil",
    "nasıl",
    "gibi",
    "kadar",
    "olan",
    "olarak",
    "daha",
    "en",
    "cok",
    "çok",
    "ama",
    "fakat",
    "lakin",
    "ancak",
    "ise",
    "ya",
    "hem",
    "her",
    "bazi",
    "bazı",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "is",
    "are",
}


def normalize_search_text(text: str) -> str:
    """Normalize Turkish and English text with accent folding for unified keyword indexing."""
    return (
        text.replace("İ", "i")
        .replace("I", "i")
        .replace("ı", "i")
        .replace("Ğ", "g")
        .replace("ğ", "g")
        .replace("Ü", "u")
        .replace("ü", "u")
        .replace("Ş", "s")
        .replace("ş", "s")
        .replace("Ö", "o")
        .replace("ö", "o")
        .replace("Ç", "c")
        .replace("ç", "c")
        .lower()
        .replace("i\u0307", "i")
    )


TURKISH_STOPWORDS = {normalize_search_text(w) for w in RAW_STOPWORDS}


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text across Turkish, English, and technical terms."""
    norm_text = normalize_search_text(text)
    # Extract alphanumeric word tokens (including numbers, dates, currency digits)
    tokens = re.findall(r"\b[\w\d]+(?:[.,/-][\w\d]+)*\b", norm_text, re.UNICODE)
    return [t for t in tokens if t not in TURKISH_STOPWORDS]


class BM25Engine:
    """Manages BM25 index for sparse keyword retrieval with Lucene-smoothed IDF."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.corpus = [tokenize_text(c.text) for c in chunks]
        self.index = (
            BM25Okapi(self.corpus, k1=settings.bm25_k1, b=settings.bm25_b) if self.corpus else None
        )
        if self.index:
            # Lucene-style smoothing ensures positive IDF values even on small corpora (1-3 chunks)
            import math

            for word in list(self.index.idf.keys()):
                freq = sum(1 for d in self.index.doc_freqs if word in d)
                self.index.idf[word] = math.log(
                    1.0 + (self.index.corpus_size - freq + 0.5) / (freq + 0.5)
                )

    def search(self, query: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
        """Score all chunks against query and return positive top-k matches."""
        if not self.index or not self.chunks:
            return []

        limit = settings.bm25_top_k if top_k is None else top_k
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        scored_pairs = [(c, float(s)) for c, s in zip(self.chunks, scores, strict=False) if s > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:limit]
