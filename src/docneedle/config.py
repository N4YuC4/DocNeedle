import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _resolve_config_toml() -> Path | None:
    """Resolve active config.toml path from environment, project root, or CWD."""
    if "CONFIG_TOML" in os.environ:
        custom_path = Path(os.environ["CONFIG_TOML"])
        if custom_path.exists():
            return custom_path
    root_toml = Path(__file__).resolve().parent.parent.parent / "config.toml"
    if root_toml.exists():
        return root_toml
    cwd_toml = Path(os.getcwd()) / "config.toml"
    if cwd_toml.exists():
        return cwd_toml
    return None


class DocNeedleTomlConfigSettingsSource(TomlConfigSettingsSource):
    """TOML settings source that supports both sectioned tables and flat keys."""

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        flat: dict[str, Any] = {}
        for section, content in data.items():
            if isinstance(content, dict):
                for k, v in content.items():
                    flat[k] = v
                    flat[f"{section}_{k}"] = v
            else:
                flat[section] = content
        return flat


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
    # 1. Check Hugging Face Spaces environment flag (Spaces containers are capped at 2 vCPUs)
    if bool(os.environ.get("SPACES_ZERO_GPU")) or os.environ.get("SYSTEM") == "spaces":
        return 2

    # 2. Linux cgroups CFS bandwidth quota detection (Docker, K8s, containerized runtimes)
    try:
        # cgroup v2: /sys/fs/cgroup/cpu.max contains "$MAX $PERIOD" (e.g. "200000 100000")
        cg2 = Path("/sys/fs/cgroup/cpu.max")
        if cg2.exists():
            parts = cg2.read_text().strip().split()
            if len(parts) >= 2 and parts[0] != "max":
                q, p = float(parts[0]), float(parts[1])
                if p > 0 and q > 0:
                    return max(1, int(round(q / p)))
    except Exception:
        pass

    try:
        # cgroup v1: cpu.cfs_quota_us and cpu.cfs_period_us
        cg1_q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        cg1_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if cg1_q.exists() and cg1_p.exists():
            q = float(cg1_q.read_text().strip())
            p = float(cg1_p.read_text().strip())
            if p > 0 and q > 0:
                return max(1, int(round(q / p)))
    except Exception:
        pass

    # 3. Process CPU count and sched_getaffinity fallbacks for bare-metal systems
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
    # Cap batch threads to avoid scheduling freezes under cgroup CPU limits
    if cores > 8:
        return min(8, cores)
    return max(1, cores)


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_path = _resolve_config_toml()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if toml_path is not None and toml_path.exists():
            sources.append(DocNeedleTomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        sources.extend([dotenv_settings, file_secret_settings])
        return tuple(sources)

    # 1. Server & Application
    app_name: str = "DocNeedle"
    app_version: str = "0.1.0"
    debug: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_field(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on", "debug")
        return bool(v)

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

    # 2. Ingestion & Document Parser (OCR-Free Visual & Digital Text Ingestion)
    model_storage_dir: str = Field(default_factory=_resolve_model_dir)
    max_upload_size_mb: int = 50
    ocr_enabled: bool = False
    pdf_render_scale: float = 1.5
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

    # 5. Vector Embedding (EmbeddingGemma-2 UD-Q4_K_XL GGUF + mmproj-F16 Cross-Modal)
    embedding_model: str = "unsloth/embeddinggemma-2-GGUF"
    embedding_quantization: str = "ud-q4_k_xl"
    embedding_dimension: int = 768
    embedding_batch_size: int = 4
    embedding_instruction: str = "task: search result | query: {text}"
    embedding_doc_instruction: str = "title: {doc_name} | text: {text}"

    # 6. Hybrid Retrieval & Reranker
    retrieval_top_k: int = 10
    rrf_constant: int = 60
    reranker_enabled: bool = Field(
        default_factory=lambda: os.environ.get("RERANKER_ENABLED", "").lower() in ("1", "true")
    )
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_candidate_k: int = 4
    rerank_top_k: int = 3
    max_chunks_per_document: int = 2
    reranker_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    # 7. Extraction Service & LLM Inference (Local GGUF Model)
    extraction_top_k: int = 5
    extraction_context_delimiter: str = "\n\n---\n\n"
    llm_model_name: str = "gemma-4-E2B-it-qat-UD-Q4_K_XL"
    llm_mmproj_name: str = "mmproj-F16"
    llm_n_ctx: int = 16384
    llm_temperature: float = 0.0
    llm_top_p: float = 0.95
    llm_max_tokens: int = 2048
    llm_single_field_max_tokens: int = 128
    llm_thinking_max_tokens: int = 512
    llm_enable_thinking: bool = False
    llm_log_extraction_payloads: bool = False
    llm_domain_rules: str = (
        "# DOMAIN SPECIFICS\n"
        "- In lease and rental contracts, 'Kira Bedeli' denotes the agreed periodic or "
        "monthly rent amount; extract this stated amount directly.\n"
        "- For address or location queries, extract the physical street/city address, omitting "
        "document timestamps and agency headers.\n"
    )
    llm_stop_tokens: list[str] = Field(
        default_factory=lambda: [
            "</s>",
            "<turn|>",
            "<end_of_turn>",
            "<|turn>",
            "<|im_end|>",
            "<|eot_id|>",
        ]
    )
    # Speculative Decoding (Prompt Lookup Decoding)
    llm_speculative_enabled: bool = True
    llm_speculative_cpu_enabled: bool = False
    llm_speculative_max_vocab_size: int = 65536
    llm_speculative_num_pred_tokens: int = 5
    llm_speculative_max_ngram_size: int = 2
    confidence_found_score: float = 0.95
    confidence_not_found_score: float = 0.0
    job_retention_minutes: int = 15

    # 8. Hardware & Performance Acceleration (GPU-accelerated when on ZeroGPU or GPU_ENABLED=1)
    gpu_enabled: bool = Field(
        default_factory=lambda: (
            bool(os.environ.get("SPACES_ZERO_GPU"))
            or os.environ.get("SYSTEM") == "spaces"
            or os.environ.get("GPU_ENABLED", "").lower() in ("1", "true")
            or False
        )
    )
    llm_n_gpu_layers: int = 0
    cpu_threads: int = Field(default_factory=_default_cpu_threads)
    cpu_threads_batch: int = Field(default_factory=_default_cpu_threads_batch)
    model_n_batch: int = 512
    reranker_n_ctx: int = 2048
    model_kv_cache_type_k: int = 8  # 8 = GGML_TYPE_Q8_0 (8-bit quantized KV cache)
    model_kv_cache_type_v: int = 8  # 8 = GGML_TYPE_Q8_0 (8-bit quantized KV cache)
    model_flash_attn: bool = True  # Required for quantized V cache in llama.cpp

    # 9. Security & Resource Constraints (Configurable via [security] or flat keys in config.toml)
    max_schema_items: int = 50
    max_upload_files: int = 20
    max_total_upload_size_mb: int = 100
    max_concurrent_jobs: int = 10

    @property
    def effective_gpu_layers(self) -> int:
        """Return -1 for full GPU offload if enabled, else 0 for CPU execution."""
        if not self.gpu_enabled:
            return 0
        return self.llm_n_gpu_layers if self.llm_n_gpu_layers != 0 else -1


settings = Settings()

if (
    not settings.gpu_enabled
    and "CUDA_VISIBLE_DEVICES" not in os.environ
    and not os.environ.get("SPACES_ZERO_GPU")
    and os.environ.get("SYSTEM") != "spaces"
):
    # Disable broken/absent CUDA runtime offload when running in local CPU-only mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
