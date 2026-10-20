import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_model_dir() -> str:
    if "MODEL_STORAGE_DIR" in os.environ:
        return os.environ["MODEL_STORAGE_DIR"]
    backend_models = Path(__file__).resolve().parent.parent.parent / "models"
    if backend_models.exists():
        return str(backend_models)
    cwd_models = Path(os.getcwd()) / "models"
    if cwd_models.exists():
        return str(cwd_models)
    cwd_backend_models = Path(os.getcwd()) / "backend" / "models"
    if cwd_backend_models.exists():
        return str(cwd_backend_models)
    return str(backend_models)


def _detect_effective_cores() -> int:
    """Detect available CPU cores respecting cgroups and container quotas (e.g. HF Spaces)."""
    if hasattr(os, "process_cpu_count"):
        proc_cores = os.process_cpu_count()
        if proc_cores:
            return proc_cores
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                return affinity
        except Exception:
            pass
    return os.cpu_count() or 2


def _default_cpu_threads() -> int:
    cores = _detect_effective_cores()
    # Reserve CPU cores for OS, display compositor, and desktop responsiveness.
    # On hybrid architectures (e.g. 12-16 thread Intel P+E cores), limit to physical P-cores
    # to avoid barrier stalls and thread contention.
    if cores >= 12:
        return 6
    elif cores > 8:
        return min(8, cores - 2)
    elif cores > 4:
        return cores - 1
    return max(1, cores)


def _default_cpu_threads_batch() -> int:
    cores = _detect_effective_cores()
    return max(1, cores)


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Server & Application
    app_name: str = "DocNeedle"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://docneedle.vercel.app",
        ]
    )
    cors_origin_regex: str | None = (
        r"^https?://("
        r"localhost|"
        r"127\.0\.0\.1|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    )

    # 2. Ingestion & Document Parser (RapidOCR & PDF inspection)
    model_storage_dir: str = Field(default_factory=_resolve_model_dir)
    max_upload_size_mb: int = 50
    ocr_enabled: bool = True
    ocr_model_name: str = "RapidOCR (PP-OCRv6 Det-Small / Rec-Small ONNX)"
    ocr_det_model_type: str = "small"
    ocr_rec_model_type: str = "small"
    ocr_text_score: float = 0.5
    ocr_longest_edge: int = 1000
    ocr_det_limit_side_len: int = 960
    ocr_det_box_thresh: float = 0.4
    ocr_det_unclip_ratio: float = 1.8
    max_ocr_pages: int = 15
    ocr_pdf_render_scale: float = 1.6
    pdf_embedded_text_min_chars: int = 50
    pdf_embedded_text_max_check_pages: int = 5
    ocr_fallback_min_chars: int = 30
    ingestion_cache_enabled: bool = True

    # 3. Chunking & Splitting (HybridChunker & Coalescence)
    chunker_tokenizer_model: str = "Qwen/Qwen3-Reranker-0.6B"
    chunker_max_tokens: int = 768
    chunker_repeat_table_header: bool = True
    chunker_merge_peers: bool = True
    chunker_max_chunk_chars: int = 1000
    chunker_min_split_chars: int = 200
    chunker_overflow_factor: float = 1.25

    # 4. BM25 Keyword Search
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 10

    # 5. Vector Embedding (Harrier-OSS-v1-0.6B Q4_K_M GGUF)
    embedding_model: str = "microsoft/harrier-oss-v1-0.6b"
    embedding_quantization: str = "q4_k_m"
    embedding_dimension: int = 512
    embedding_batch_size: int = 4
    embedding_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: {text}"
    )

    # 6. Hybrid Retrieval & Reranker
    retrieval_top_k: int = 10
    rrf_constant: int = 60
    reranker_enabled: bool = False
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_candidate_k: int = 4
    rerank_top_k: int = 5
    max_chunks_per_document: int = 2
    reranker_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    # 7. Extraction Service & LLM Inference (Gemma-4-E2B-it-qat-UD-Q4_K_XL GGUF)
    extraction_top_k: int = 5
    extraction_context_delimiter: str = "\n\n---\n\n"
    llm_model_name: str = "gemma-4-E2B-it-qat-UD-Q4_K_XL"
    llm_n_ctx: int = 16384
    llm_temperature: float = 0.0
    llm_top_p: float = 0.95
    llm_max_tokens: int = 2048
    llm_single_field_max_tokens: int = 128
    llm_thinking_max_tokens: int = 512
    llm_enable_thinking: bool = False
    llm_stop_tokens: list[str] = Field(
        default_factory=lambda: [
            "</s>",
            "[/INST]",
            "[INST]",
            "<turn|>",
            "<end_of_turn>",
            "<|turn>",
        ]
    )
    # Speculative Decoding (Prompt Lookup Decoding)
    llm_speculative_enabled: bool = True
    llm_speculative_cpu_enabled: bool = False
    llm_speculative_num_pred_tokens: int = 5
    llm_speculative_max_ngram_size: int = 2
    confidence_found_score: float = 0.95
    confidence_not_found_score: float = 0.0
    job_retention_minutes: int = 15

    # 8. Hardware & Performance Acceleration (CPU-optimized for HF Spaces by default)
    gpu_enabled: bool = False
    llm_n_gpu_layers: int = 0
    cpu_threads: int = Field(default_factory=_default_cpu_threads)
    cpu_threads_batch: int = Field(default_factory=_default_cpu_threads_batch)
    model_n_batch: int = 512
    reranker_n_ctx: int = 2048
    model_kv_cache_type_k: int = 8  # 8 = GGML_TYPE_Q8_0 (8-bit quantized KV cache)
    model_kv_cache_type_v: int = 8  # 8 = GGML_TYPE_Q8_0 (8-bit quantized KV cache)
    model_flash_attn: bool = True  # Required for quantized V cache in llama.cpp

    @property
    def effective_gpu_layers(self) -> int:
        """Return -1 for full GPU offload if enabled, else 0 for CPU execution."""
        if not self.gpu_enabled:
            return 0
        return self.llm_n_gpu_layers if self.llm_n_gpu_layers != 0 else -1


settings = Settings()

if not settings.gpu_enabled and "CUDA_VISIBLE_DEVICES" not in os.environ:
    # Disable broken/absent CUDA runtime offload when running in CPU-only mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
