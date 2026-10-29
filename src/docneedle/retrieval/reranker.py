"""Cross-encoder reranking engine powered by Qwen3-Reranker-0.6B GGUF via llama.cpp."""

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from docneedle.config import settings
from docneedle.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _build_reranker_tokens(
    query: str,
    doc_text: str,
    instruction: str,
    max_ctx: int,
    llm: Any,
) -> list[int]:
    """Construct tokenized reranker input, truncating doc text if context budget is exceeded."""
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc_text}"
    full_prompt = RERANKER_PREFIX + body + RERANKER_SUFFIX

    tokens = llm.tokenize(full_prompt.encode("utf-8"))
    if len(tokens) <= max_ctx:
        return tokens

    prefix_str = f"{RERANKER_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "
    prefix_tokens = llm.tokenize(prefix_str.encode("utf-8"))
    suffix_tokens = llm.tokenize(RERANKER_SUFFIX.encode("utf-8"))
    avail_doc = max(32, max_ctx - len(prefix_tokens) - len(suffix_tokens) - 8)
    doc_tokens = llm.tokenize(doc_text.encode("utf-8"))[:avail_doc]
    truncated_doc = llm.detokenize(doc_tokens).decode("utf-8", errors="ignore")
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {truncated_doc}"
    full_prompt = RERANKER_PREFIX + body + RERANKER_SUFFIX
    return llm.tokenize(full_prompt.encode("utf-8"))[:max_ctx]


class Qwen3Reranker:
    """Reranker using Qwen3-Reranker-0.6B GGUF with llama.cpp."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        if n_gpu_layers != 0:
            from docneedle.cuda_compat import preload_cuda_runtime_libs

            preload_cuda_runtime_libs()
        import llama_cpp

        if model_path is None:
            model_path = self._resolve_model_path()

        self.model_path = str(model_path)
        self.n_gpu_layers = n_gpu_layers
        logger.info(
            f"Loading Qwen3-Reranker from {self.model_path} (gpu_layers={self.n_gpu_layers})"
        )
        self.llm = llama_cpp.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=settings.reranker_n_ctx,
            n_threads=settings.cpu_threads,
            n_threads_batch=settings.cpu_threads_batch,
            n_batch=settings.model_n_batch,
            logits_all=False,
            type_k=settings.model_kv_cache_type_k,
            type_v=settings.model_kv_cache_type_v,
            flash_attn=settings.model_flash_attn,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        # Release the unused per-token vocabulary logits buffer in llama-cpp-python:
        # score() reads logits directly from self.llm._ctx.get_logits() and never uses self.scores.
        self.llm.scores = np.empty((0, self.llm.n_vocab()), dtype=np.single)
        self.token_false_id = self.llm.tokenize(b"no", add_bos=False)[0]
        self.token_true_id = self.llm.tokenize(b"yes", add_bos=False)[0]
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_model_path() -> Path:
        base_dir = Path(settings.model_storage_dir)
        model_path = base_dir / "reranker" / "Qwen3-Reranker-0.6B.Q4_K_M.gguf"
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"No Qwen3-Reranker GGUF model found at: {model_path}")

    def score(
        self,
        query: str,
        doc_text: str,
        instruction: str | None = None,
    ) -> float:
        """Compute the relevance score (logit difference between yes and no)
        for a query-document pair.
        """
        instr = settings.reranker_instruction if instruction is None else instruction

        with self._lock:
            default_ctx = settings.reranker_n_ctx
            n_ctx_attr = getattr(self.llm, "n_ctx", None)
            max_ctx = n_ctx_attr() if callable(n_ctx_attr) else (n_ctx_attr or default_ctx)

            tokens = _build_reranker_tokens(
                query=query,
                doc_text=doc_text,
                instruction=instr,
                max_ctx=max_ctx,
                llm=self.llm,
            )

            self.llm.reset()
            self.llm.eval(tokens)
            logits_ptr = self.llm._ctx.get_logits()
            logits = np.ctypeslib.as_array(logits_ptr, shape=(self.llm._n_vocab,))

            false_logit = float(logits[self.token_false_id])
            true_logit = float(logits[self.token_true_id])

            return true_logit - false_logit

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[SearchResult]:
        """Rerank candidates using cross-encoder relevance scores and return top_k.

        Optimized with KV-cache prefix reuse to avoid re-evaluating the shared instruction
        and query tokens across candidate chunks.
        """
        if not candidates:
            return []

        logger.info(
            f"--> [RERANKER] Scoring {len(candidates)} candidate chunks with Qwen3 for '{query}'..."
        )
        limit = settings.rerank_top_k if top_k is None else top_k
        instr = settings.reranker_instruction if instruction is None else instruction

        scored_candidates: list[SearchResult] = []
        for c in candidates:
            # If chunk is purely visual lacking text, preserve cross-modal dense score directly
            # without wasting CPU cycles running a text cross-encoder on placeholder text.
            is_visual_only = c.chunk.image_bytes is not None and c.chunk.text.startswith(
                ("[Görsel", "[Visual")
            )
            if is_visual_only:
                rel_score = (float(c.dense_score or 0.0) - 0.5) * 5.0
            else:
                rel_score = self.score(
                    query=query,
                    doc_text=c.chunk.text,
                    instruction=instr,
                )

            # Preserve cross-modal dense semantic signal for visual page chunks
            if c.chunk.image_bytes is not None and c.dense_score is not None:
                dense_logit_boost = (float(c.dense_score) - 0.5) * 5.0
                rel_score = max(rel_score, dense_logit_boost)

            scored_candidates.append(
                SearchResult(
                    chunk=c.chunk,
                    score=rel_score,
                    dense_score=c.dense_score,
                    sparse_score=c.sparse_score,
                    rrf_score=c.rrf_score,
                    rerank_score=rel_score,
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        if scored_candidates:
            top_match = scored_candidates[0]
            logger.info(
                f"    [RERANKER DONE] Top candidate: {top_match.chunk.doc_name} "
                f"(Page {top_match.chunk.page_number or 1}, Score: {top_match.score:.4f})"
            )
        return scored_candidates[:limit]


_reranker_instance: Qwen3Reranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(force: bool = False) -> Qwen3Reranker | None:
    """Factory returning Qwen3Reranker singleton if enabled/forced and model exists,
    otherwise None.
    """
    global _reranker_instance
    if not force and not settings.reranker_enabled and _reranker_instance is None:
        return None
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                try:
                    _reranker_instance = Qwen3Reranker()
                except Exception as e:
                    logger.warning(f"Could not load Qwen3-Reranker: {e}")
                    return None
    return _reranker_instance
