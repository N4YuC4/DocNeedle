"""Embedding engine powered by Harrier-OSS-v1-0.6B GGUF via llama.cpp."""

import logging
import threading
from pathlib import Path

import numpy as np

from docneedle.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """Abstract base interface for document and query embedders."""

    def embed_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], is_query=True)[0]


class LlamaCppHarrierEmbedder(BaseEmbedder):
    """Embedder using Harrier-OSS-v1-0.6B GGUF via llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        dimension: int | None = None,
    ) -> None:
        import llama_cpp

        self.model_path = str(model_path)
        self.dimension = dimension if dimension is not None else settings.embedding_dimension
        self.n_ctx = max(2048, settings.reranker_n_ctx)
        eff_batch = max(settings.model_n_batch, 2048)
        logger.info(f"Loading Harrier-OSS embedding from {self.model_path} (dim={self.dimension})")
        self.llm = llama_cpp.Llama(
            model_path=self.model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_ctx=self.n_ctx,
            n_gpu_layers=settings.effective_gpu_layers,
            n_threads=settings.cpu_threads,
            n_threads_batch=settings.cpu_threads_batch,
            n_batch=eff_batch,
            type_k=settings.model_kv_cache_type_k,
            type_v=settings.model_kv_cache_type_v,
            flash_attn=settings.model_flash_attn,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        self._lock = threading.Lock()

    def _sanitize_prompt(self, prompt: str) -> str:
        """Ensure prompt fits within safe context budget before sending to batch embedding."""
        if len(prompt) > (self.n_ctx * 3):
            tokens = self.llm.tokenize(prompt.encode("utf-8"))
            safe_limit = max(64, self.n_ctx - 32)
            if len(tokens) > safe_limit:
                return self.llm.detokenize(tokens[:safe_limit]).decode("utf-8", errors="ignore")
        return prompt

    def _embed_single(self, text: str, is_query: bool = False) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        if is_query:
            prompt = settings.embedding_instruction.format(text=text)
        else:
            prompt = text

        with self._lock:
            try:
                res = self.llm.create_embedding(prompt)
            except Exception:
                # Truncate prompt if single chunk exceeds model context window
                tokens = self.llm.tokenize(prompt.encode("utf-8"))
                safe_tokens = tokens[: max(64, self.n_ctx - 16)]
                truncated_prompt = self.llm.detokenize(safe_tokens).decode("utf-8", errors="ignore")
                res = self.llm.create_embedding(truncated_prompt)
        raw_emb = np.array(res["data"][0]["embedding"], dtype=np.float32)

        # MRL truncation to target dimension & L2 normalization
        vec = raw_emb[: self.dimension]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed_texts(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        bs = settings.embedding_batch_size if batch_size is None else batch_size
        results: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            if any(not t.strip() for t in batch):
                for t in batch:
                    results.append(self._embed_single(t, is_query=is_query))
                continue
            prompts = [
                self._sanitize_prompt(
                    settings.embedding_instruction.format(text=t) if is_query else t
                )
                for t in batch
            ]
            try:
                with self._lock:
                    res = self.llm.create_embedding(prompts)
                for item in res["data"]:
                    raw_emb = np.array(item["embedding"], dtype=np.float32)
                    vec = raw_emb[: self.dimension]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    results.append(vec)
            except Exception as e:
                logger.debug(f"Batch embedding fallback triggered ({e}). Retrying individually...")
                # Retry individually if batch exceeds context
                for t in batch:
                    results.append(self._embed_single(t, is_query=is_query))

        return np.vstack(results).astype(np.float32)


class OnnxHarrierEmbedder(BaseEmbedder):
    """High-performance thread-safe embedder using Harrier-OSS-v1-0.6B ONNX via onnxruntime."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        dimension: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path)
        self.dimension = dimension if dimension is not None else settings.embedding_dimension

        logger.info(
            f"Loading Harrier-OSS ONNX embedding from {self.model_path} (dim={self.dimension})"
        )
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        pad_id = self.tokenizer.token_to_id("<|endoftext|>") or 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<|endoftext|>")
        self.tokenizer.enable_truncation(max_length=settings.chunker_max_tokens)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = settings.cpu_threads_batch
        sess_opts.inter_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    def embed_texts(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        bs = settings.embedding_batch_size if batch_size is None else batch_size
        results: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            valid_prompts: list[str] = []
            valid_indices: list[int] = []

            for idx, t in enumerate(batch):
                clean_t = t.strip()
                if clean_t:
                    prompt = (
                        settings.embedding_instruction.format(text=clean_t) if is_query else clean_t
                    )
                    valid_prompts.append(prompt)
                    valid_indices.append(idx)

            batch_embs = np.zeros((len(batch), self.dimension), dtype=np.float32)

            if valid_prompts:
                encoded = self.tokenizer.encode_batch(valid_prompts)
                input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
                attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

                outputs = self.session.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )
                raw_out = outputs[0]  # shape: (B, 1024)

                # MRL truncation & normalization
                vecs = raw_out[:, : self.dimension].astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs /= norms

                for sub_idx, orig_idx in enumerate(valid_indices):
                    batch_embs[orig_idx] = vecs[sub_idx]

            results.append(batch_embs)

        return np.vstack(results).astype(np.float32)


# Alias for backwards compatibility
LlamaCppQwenEmbedder = LlamaCppHarrierEmbedder

_embedder_instance: BaseEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> BaseEmbedder:
    """Factory creating BaseEmbedder singleton powered by Harrier-OSS ONNX or GGUF."""
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    with _embedder_lock:
        if _embedder_instance is not None:
            return _embedder_instance

        base_dir = Path(settings.model_storage_dir)

        # 1. Prefer ONNX model if present
        onnx_candidates = [
            base_dir / "embedding" / "harrier-onnx",
            base_dir / "embedding" / "onnx",
        ]
        for od in onnx_candidates:
            onnx_model = od / "onnx" / "model_q4.onnx"
            onnx_tok = od / "tokenizer.json"
            if onnx_model.exists() and onnx_tok.exists():
                _embedder_instance = OnnxHarrierEmbedder(
                    model_path=onnx_model,
                    tokenizer_path=onnx_tok,
                    dimension=settings.embedding_dimension,
                )
                return _embedder_instance

        # 2. GGUF via llama.cpp
        quant = settings.embedding_quantization.upper()
        gguf_model = base_dir / "embedding" / f"harrier-oss-v1-0.6b.{quant}.gguf"
        if gguf_model.exists():
            _embedder_instance = LlamaCppHarrierEmbedder(
                model_path=gguf_model,
                dimension=settings.embedding_dimension,
            )
            return _embedder_instance

        raise FileNotFoundError(
            f"No Harrier-OSS embedding (ONNX/GGUF) model in candidate locations: {[gguf_model]}"
        )
