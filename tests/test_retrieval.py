import numpy as np
import pytest

from docneedle.ingest.models import DocumentChunk
from docneedle.retrieval.bm25 import BM25Engine, tokenize_text
from docneedle.retrieval.embedder import BaseEmbedder
from docneedle.retrieval.store import SessionVectorStore


class MockDeterministicEmbedder(BaseEmbedder):
    """Deterministic mock embedder for fast unit tests without loading neural weights."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for idx, word in enumerate(text.lower().split()):
                h = hash(word) % self.dimension
                vec[h] += 1.0 / (1.0 + idx)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doc_name="sozlesme.pdf",
            chunk_index=0,
            text="Sözleşmenin bitiş tarihi 31 Mart 2027 olarak kararlaştırılmıştır.",
            page_number=1,
            heading="Süre ve Fesih",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            doc_name="sozlesme.pdf",
            chunk_index=1,
            text="Toplam sözleşme bedeli KDV hariç 150.000 TL dir.",
            page_number=2,
            heading="Ödeme Koşulları",
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            doc_id="doc_1",
            doc_name="sozlesme.pdf",
            chunk_index=2,
            text="Taraflar: Acme Anonim Şirketi ve Beta Bilişim Limited Şirketi.",
            page_number=1,
            heading="Taraflar",
        ),
    ]


def test_tokenize_text() -> None:
    tokens = tokenize_text("İşbu Sözleşme, 150.000 TL bedelle imzalandı.")
    assert "isbu" in tokens
    assert "sozlesme" in tokens
    assert "150.000" in tokens
    assert "tl" in tokens


def test_multilingual_and_casing_bm25_match() -> None:
    chunk_invoice = DocumentChunk(
        chunk_id="c_inv",
        doc_id="d1",
        doc_name="invoice.pdf",
        chunk_index=0,
        text="INVOICE ID: INV-9988 TAX ID: 1234567890",
    )
    chunk_kimlik = DocumentChunk(
        chunk_id="c_kimlik",
        doc_id="d2",
        doc_name="kimlik.pdf",
        chunk_index=0,
        text="T.C. KIMLIK NUMARASI: 11223344556 ADI: AHMET",
    )
    bm25 = BM25Engine([chunk_invoice, chunk_kimlik])

    # Lowercase English query matches all-caps doc
    res_inv = bm25.search("invoice id")
    assert len(res_inv) > 0
    assert res_inv[0][0].chunk_id == "c_inv"

    # Lowercase Turkish query with 'i' matches ASCII 'KIMLIK'
    res_kimlik = bm25.search("kimlik numarası")
    assert len(res_kimlik) > 0
    assert res_kimlik[0][0].chunk_id == "c_kimlik"


def test_bm25_search(sample_chunks: list[DocumentChunk]) -> None:
    bm25 = BM25Engine(sample_chunks)
    results = bm25.search("sözleşme bedeli ne kadar?", top_k=2)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_2"
    assert "150.000 TL" in top_chunk.text
    assert score > 0


def test_session_vector_store_hybrid(sample_chunks: list[DocumentChunk]) -> None:
    embedder = MockDeterministicEmbedder(dimension=128)
    store = SessionVectorStore(embedder=embedder)
    store.add_chunks(sample_chunks)

    # Search for bitiş tarihi
    results = store.search("bitiş tarihi ne zaman?", top_k=2)
    assert len(results) == 2
    top_result = results[0]
    assert top_result.chunk.chunk_id == "chunk_1"
    assert top_result.rrf_score is not None
    assert top_result.score > 0

    # Clear memory
    store.clear()
    assert len(store.chunks) == 0
    assert store.embeddings is None


def test_session_vector_store_with_reranker(sample_chunks: list[DocumentChunk]) -> None:
    embedder = MockDeterministicEmbedder(dimension=128)

    class DummyReranker:
        def rerank(self, query: str, candidates: list, top_k: int = 5):
            sorted_candidates = sorted(
                candidates,
                key=lambda x: 10.0 if x.chunk.chunk_id == "chunk_2" else 1.0,
                reverse=True,
            )
            for c in sorted_candidates:
                c.rerank_score = 10.0 if c.chunk.chunk_id == "chunk_2" else 1.0
                c.score = c.rerank_score
            return sorted_candidates[:top_k]

    store = SessionVectorStore(embedder=embedder, reranker=DummyReranker())
    store.add_chunks(sample_chunks)

    results = store.search("ödeme", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk_2"
    assert results[0].rerank_score == 10.0


def test_get_embedder_missing_model_raises_filenotfound(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import docneedle.retrieval.embedder as emb_mod
    from docneedle.config import settings

    monkeypatch.setattr(settings, "model_storage_dir", str(tmp_path))
    monkeypatch.setattr(emb_mod, "_embedder_instance", None)
    with pytest.raises(FileNotFoundError) as exc_info:
        emb_mod.get_embedder()
    assert "No Harrier-OSS embedding" in str(exc_info.value)


def test_onnx_harrier_embedder_lifecycle() -> None:
    """Verify OnnxHarrierEmbedder text and query embedding, dimension, and L2 normalization."""
    from docneedle.retrieval.embedder import OnnxHarrierEmbedder, get_embedder

    emb = get_embedder()
    assert isinstance(emb, OnnxHarrierEmbedder)
    texts = ["Ahmet Yılmaz", "Kira Sözleşmesi 25.000 TL", ""]
    embs = emb.embed_texts(texts)
    assert embs.shape == (3, 512)
    assert np.all(embs[2] == 0.0)
    assert abs(float(np.linalg.norm(embs[0])) - 1.0) < 1e-4
    assert abs(float(np.linalg.norm(embs[1])) - 1.0) < 1e-4

    q_vec = emb.embed_query("Kira bedeli")
    assert q_vec.shape == (512,)
    assert abs(float(np.linalg.norm(q_vec)) - 1.0) < 1e-4
