import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docneedle.api.routes_extract import router as extract_router
from docneedle.api.routes_jobs import router as jobs_router
from docneedle.config import settings

logger = logging.getLogger(__name__)

_SERVER_START_TIME = time.time()


class RootResponse(BaseModel):
    """Response payload for root service endpoint."""

    service: str = Field(..., description="Application service name")
    version: str = Field(..., description="Application service version")
    docs_url: str = Field(..., description="Interactive Swagger documentation endpoint")
    health_url: str = Field(..., description="Health check endpoint")


class HealthHardwareInfo(BaseModel):
    """Hardware acceleration and GPU detection diagnostics."""

    device: str = Field(..., description="Active execution device ('cuda' or 'cpu')")
    gpu_enabled: bool = Field(..., description="Whether GPU execution is configured")
    torch_cuda: bool | str = Field(
        ...,
        description="PyTorch CUDA availability flag or error description string",
    )
    cuda_visible_devices: str | None = Field(
        default=None,
        description="Active CUDA_VISIBLE_DEVICES environment variable",
    )
    spaces_zero_gpu: str | None = Field(
        default=None,
        description="Hugging Face Spaces ZeroGPU leasing flag",
    )


class ModelStatusItem(BaseModel):
    """Status and disk footprint of an individual neural model file."""

    name: str = Field(..., description="Model filename or identifier")
    ready: bool = Field(..., description="Whether model file exists on disk")
    size_mb: float | None = Field(default=None, description="Model file size in megabytes")


class HealthModelsInfo(BaseModel):
    """Local model files readiness and storage integrity."""

    all_ready: bool = Field(..., description="True if all essential models are available on disk")
    llm: ModelStatusItem = Field(..., description="Multimodal LLM GGUF model status")
    llm_mmproj: ModelStatusItem = Field(..., description="Multimodal vision projector status")
    embedding: ModelStatusItem = Field(..., description="Cross-modal embedder GGUF status")
    embedding_mmproj: ModelStatusItem = Field(..., description="Embedding vision projector status")
    reranker: ModelStatusItem = Field(..., description="Cross-encoder reranker GGUF status")


class HealthJobsInfo(BaseModel):
    """In-memory asynchronous background job queue metrics."""

    active_jobs: int = Field(default=0, description="Currently running background extraction jobs")
    total_cached_jobs: int = Field(
        default=0, description="Total cached job results currently retained in RAM"
    )


class HealthConfigInfo(BaseModel):
    """Active configuration parameters for neural models and pipeline."""

    embedding_model: str = Field(..., description="Active embedding model identifier")
    embedding_quantization: str = Field(..., description="Embedding model quantization type")
    llm_model: str = Field(..., description="Active LLM model name")
    llm_speculative_enabled: bool = Field(..., description="Whether speculative decoding is active")
    multimodal_embedding: bool = Field(
        default=True,
        description="Whether cross-modal vision embeddings are enabled",
    )
    multimodal_llm: bool = Field(
        default=True,
        description="Whether multimodal vision extraction is supported",
    )
    reranker_enabled: bool = Field(
        ...,
        description="Whether cross-encoder reranker is enabled by default",
    )
    gpu_enabled: bool = Field(..., description="Whether GPU inference is requested")
    cpu_threads: int = Field(..., description="Configured CPU worker threads")
    max_upload_size_mb: int = Field(..., description="Maximum single file upload limit in MB")
    max_total_upload_size_mb: int = Field(..., description="Maximum total payload size in MB")
    max_schema_items: int = Field(..., description="Maximum extraction schema fields limit")


class HealthResponse(BaseModel):
    """Diagnostic response detailing runtime environment, memory, models, and job queue."""

    status: str = Field(
        default="healthy",
        description="Service health status indicator ('healthy' or 'degraded')",
    )
    service: str = Field(..., description="Application service name")
    version: str = Field(..., description="Application service version")
    uptime_seconds: float = Field(..., description="Microservice uptime in seconds")
    process_memory_mb: float = Field(..., description="Current process resident memory (RSS) in MB")
    python_version: str = Field(..., description="Python interpreter version string")
    free_threaded: bool = Field(
        ...,
        description="True if running free-threaded Python with GIL disabled (PEP 703)",
    )
    gil_enabled: bool = Field(
        ...,
        description="True if Global Interpreter Lock (GIL) is active",
    )
    hardware: HealthHardwareInfo = Field(..., description="Hardware and GPU diagnostics")
    models: HealthModelsInfo = Field(..., description="Local GGUF models presence and size on disk")
    jobs: HealthJobsInfo = Field(..., description="Background jobs queue metrics")
    config: HealthConfigInfo = Field(..., description="Model and engine configuration")


def check_torch_cuda_status() -> bool | str:
    """Inspect PyTorch CUDA availability safely without crashing if PyTorch is absent."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:
        return str(exc)


def _inspect_model_file(path: Path, name: str) -> ModelStatusItem:
    exists = path.exists()
    size_mb = round(path.stat().st_size / (1024 * 1024), 1) if exists else None
    return ModelStatusItem(name=name, ready=exists, size_mb=size_mb)


def check_models_status() -> HealthModelsInfo:
    """Check existence and size of local quantized model weights without loading them."""
    models_dir = Path(settings.model_storage_dir)
    llm_name = (
        settings.llm_model_name
        if settings.llm_model_name.endswith(".gguf")
        else f"{settings.llm_model_name}.gguf"
    )
    llm_path = models_dir / "llm" / llm_name

    mmproj_name = (
        settings.llm_mmproj_name
        if settings.llm_mmproj_name.endswith(".gguf")
        else f"{settings.llm_mmproj_name}.gguf"
    )
    llm_mmproj_path = models_dir / "llm" / mmproj_name

    emb_path = models_dir / "embedding" / "embeddinggemma-2-UD-Q4_K_XL.gguf"
    emb_mmproj_path = models_dir / "embedding" / "mmproj-F16.gguf"
    reranker_path = models_dir / "reranker" / "Qwen3-Reranker-0.6B.Q4_K_M.gguf"

    llm_item = _inspect_model_file(llm_path, llm_name)
    llm_mmproj_item = _inspect_model_file(llm_mmproj_path, mmproj_name)
    emb_item = _inspect_model_file(emb_path, "embeddinggemma-2-UD-Q4_K_XL.gguf")
    emb_mmproj_item = _inspect_model_file(emb_mmproj_path, "mmproj-F16.gguf")
    reranker_item = _inspect_model_file(reranker_path, "Qwen3-Reranker-0.6B.Q4_K_M.gguf")

    all_ready = llm_item.ready and llm_mmproj_item.ready and emb_item.ready
    return HealthModelsInfo(
        all_ready=all_ready,
        llm=llm_item,
        llm_mmproj=llm_mmproj_item,
        embedding=emb_item,
        embedding_mmproj=emb_mmproj_item,
        reranker=reranker_item,
    )


def get_process_memory_mb() -> float:
    """Fetch current process resident memory (RSS) in megabytes."""
    try:
        import resource

        # On Linux, ru_maxrss is reported in kilobytes
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return 0.0


def get_system_health() -> HealthResponse:
    """Collect runtime environment, memory footprint, model files, and queue status."""
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    models_info = check_models_status()

    try:
        from docneedle.jobs.manager import job_manager

        job_stats = job_manager.get_stats()
        jobs_info = HealthJobsInfo(
            active_jobs=job_stats.get("active_jobs", 0),
            total_cached_jobs=job_stats.get("total_cached_jobs", 0),
        )
    except Exception:
        jobs_info = HealthJobsInfo(active_jobs=0, total_cached_jobs=0)

    status = "healthy" if models_info.all_ready else "degraded"
    device = "cuda" if settings.gpu_enabled else "cpu"

    return HealthResponse(
        status=status,
        service=settings.app_name,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _SERVER_START_TIME, 1),
        process_memory_mb=get_process_memory_mb(),
        python_version=sys.version,
        free_threaded=not gil_enabled,
        gil_enabled=gil_enabled,
        hardware=HealthHardwareInfo(
            device=device,
            gpu_enabled=settings.gpu_enabled,
            torch_cuda=check_torch_cuda_status(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            spaces_zero_gpu=os.environ.get("SPACES_ZERO_GPU"),
        ),
        models=models_info,
        jobs=jobs_info,
        config=HealthConfigInfo(
            embedding_model=settings.embedding_model,
            embedding_quantization=settings.embedding_quantization,
            llm_model=settings.llm_model_name,
            llm_speculative_enabled=settings.llm_speculative_enabled,
            multimodal_embedding=True,
            multimodal_llm=True,
            reranker_enabled=settings.reranker_enabled,
            gpu_enabled=settings.gpu_enabled,
            cpu_threads=settings.cpu_threads,
            max_upload_size_mb=settings.max_upload_size_mb,
            max_total_upload_size_mb=settings.max_total_upload_size_mb,
            max_schema_items=settings.max_schema_items,
        ),
    )


router = APIRouter(tags=["System"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="Root service information",
)
async def root() -> RootResponse:
    """Root endpoint providing service information."""
    return RootResponse(
        service=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        health_url="/healthz",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check (alias)",
    include_in_schema=False,
)
@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """Health check endpoint detailing runtime environment, GIL state, and hardware config."""
    return get_system_health()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for application startup and shutdown lifecycle events."""
    logger.info(
        "--> [DOCNEEDLE API] Initializing %s v%s (Python %s)...",
        settings.app_name,
        settings.app_version,
        sys.version.split()[0],
    )
    yield


def create_app() -> FastAPI:
    """Application factory configuring routes, middlewares, and OpenAPI documentation."""
    app_instance = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Privacy-first local RAG document extraction microservice",
        lifespan=lifespan,
    )

    # Configure CORS for Vercel frontend, local development & local network (LAN)
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_instance.include_router(router)
    app_instance.include_router(extract_router)
    app_instance.include_router(jobs_router)

    return app_instance


app = create_app()
