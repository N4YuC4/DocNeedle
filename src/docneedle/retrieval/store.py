import logging

import numpy as np

from docneedle.config import settings
from docneedle.ingest.models import DocumentChunk
from docneedle.retrieval.bm25 import BM25Engine
from docneedle.retrieval.embedder import BaseEmbedder, get_embedder
from docneedle.retrieval.models import SearchResult
from docneedle.retrieval.reranker import Qwen3Reranker, get_reranker

logger = logging.getLogger(__name__)


def _apply_document_diversity(
    candidates: list[SearchResult],
    top_k: int,
    max_per_doc: int,
) -> list[SearchResult]:
    """Filter candidates to prevent single-document dominance while preserving top_k ranking."""
    diversified: list[SearchResult] = []
    doc_counts: dict[str, int] = {}

    for r in candidates:
        doc_key = r.chunk.doc_name or r.chunk.doc_id
        if doc_counts.get(doc_key, 0) < max_per_doc:
            diversified.append(r)
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
        if len(diversified) >= top_k:
            break

    # Backfill if diversity constraint yielded fewer than top_k results
    if len(diversified) < top_k:
        seen_ids = {r.chunk.chunk_id for r in diversified}
        for r in candidates:
            if r.chunk.chunk_id not in seen_ids:
                diversified.append(r)
                seen_ids.add(r.chunk.chunk_id)
            if len(diversified) >= top_k:
                break

    return diversified


class SessionVectorStore:
    """Session-scoped in-memory vector and keyword store.

    Guarantees privacy by maintaining data only in RAM and clearing all structures
    when the extraction task finishes.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        reranker: Qwen3Reranker | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self._custom_reranker = reranker is not None
        self.reranker = reranker if reranker is not None else get_reranker()
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Engine | None = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add new chunks to the in-memory store and recompute indexes."""
        if not chunks:
            return

        logger.info(f"--> [VECTOR STORE] Embedding {len(chunks)} chunks in RAM (text/images)...")
        self.chunks.extend(chunks)

        # Compute embeddings for new chunks in RAM (supporting multimodal visual & text)
        new_embs_list: list[np.ndarray] = []
        for c in chunks:
            if c.image_bytes is not None:
                img_emb = self.embedder.embed_images([c.image_bytes])
                new_embs_list.append(img_emb)
            else:
                txt_emb = self.embedder.embed_texts(
                    [c.text], is_query=False, doc_names=[c.doc_name]
                )
                new_embs_list.append(txt_emb)

        new_embs = (
            np.vstack(new_embs_list)
            if new_embs_list
            else np.zeros((0, self.embedder.dimension), dtype=np.float32)
        )

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        # Re-index BM25
        self.bm25 = BM25Engine(self.chunks)
        logger.info(
            f"--> [INDEX COMPLETE] Indexed {len(self.chunks)} total chunks in RAM (Dense + BM25)."
        )

    def _rank_dense(self, query_vec: np.ndarray) -> dict[str, tuple[int, float]]:
        """Compute cosine similarity ranking for dense embeddings."""
        if self.embeddings is None:
            return {}
        dense_scores = np.dot(self.embeddings, query_vec)
        dense_rankings = np.argsort(-dense_scores)

        dense_ranks_by_id: dict[str, tuple[int, float]] = {}
        for rank, idx in enumerate(dense_rankings):
            c = self.chunks[idx]
            dense_ranks_by_id[c.chunk_id] = (rank + 1, float(dense_scores[idx]))
        return dense_ranks_by_id

    def _rank_sparse(self, query: str) -> dict[str, tuple[int, float]]:
        """Compute BM25 keyword relevance ranking."""
        sparse_ranks_by_id: dict[str, tuple[int, float]] = {}
        if self.bm25:
            bm25_matches = [
                m for m in self.bm25.search(query, top_k=len(self.chunks)) if m[1] > 0.0
            ]
            for rank, (c, score) in enumerate(bm25_matches):
                sparse_ranks_by_id[c.chunk_id] = (rank + 1, float(score))
        return sparse_ranks_by_id

    def _fuse_rrf(
        self,
        dense_ranks: dict[str, tuple[int, float]],
        sparse_ranks: dict[str, tuple[int, float]],
        rrf_constant: int = 60,
    ) -> list[SearchResult]:
        """Combine dense and sparse search rankings using Reciprocal Rank Fusion."""
        scored_results: list[SearchResult] = []
        for c in self.chunks:
            chunk_id = c.chunk_id
            rrf_score = 0.0

            dense_score = None
            if chunk_id in dense_ranks:
                rank_dense, d_score = dense_ranks[chunk_id]
                rrf_score += 1.0 / (rrf_constant + rank_dense)
                dense_score = d_score

            sparse_score = None
            if chunk_id in sparse_ranks:
                rank_sparse, s_score = sparse_ranks[chunk_id]
                rrf_score += 1.0 / (rrf_constant + rank_sparse)
                sparse_score = s_score

            scored_results.append(
                SearchResult(
                    chunk=c,
                    score=float(rrf_score),
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rrf_score=float(rrf_score),
                )
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        rrf_constant: int | None = None,
        enable_reranker: bool | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining Dense cosine similarity and BM25 via RRF."""
        if not self.chunks or self.embeddings is None:
            return []

        target_top_k = settings.retrieval_top_k if top_k is None else top_k
        const = settings.rrf_constant if rrf_constant is None else rrf_constant
        if enable_reranker is not None:
            should_rerank = enable_reranker
        else:
            should_rerank = self._custom_reranker or settings.reranker_enabled

        # 1. Dense and Sparse hybrid scoring
        query_vec = self.embedder.embed_query(query)
        dense_ranks = self._rank_dense(query_vec)
        sparse_ranks = self._rank_sparse(query)
        scored_results = self._fuse_rrf(dense_ranks, sparse_ranks, rrf_constant=const)

        max_per_doc = settings.max_chunks_per_document

        # 2. Cross-encoder reranking (if available and enabled)
        if should_rerank and scored_results:
            if self.reranker is None:
                self.reranker = get_reranker(force=True)
            if self.reranker is not None:
                candidate_k = min(
                    len(scored_results),
                    settings.reranker_candidate_k,
                )
                candidates = scored_results[:candidate_k]
                reranked = self.reranker.rerank(
                    query=query, candidates=candidates, top_k=target_top_k
                )
                return _apply_document_diversity(
                    reranked, top_k=target_top_k, max_per_doc=max_per_doc
                )

        # 3. Direct RRF ranking when reranker is disabled or not loaded
        return _apply_document_diversity(
            scored_results, top_k=target_top_k, max_per_doc=max_per_doc
        )

    def clear(self) -> None:
        """Clear all chunks and vectors from RAM immediately."""
        self.chunks.clear()
        self.embeddings = None
        self.bm25 = None
        import gc

        gc.collect()
