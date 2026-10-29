import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docneedle.api.routes_extract import router as extract_router
from docneedle.api.routes_jobs import router as jobs_router
from docneedle.config import settings

logger = logging.getLogger(__name__)


class RootResponse(BaseModel):
    """Response payload for root service endpoint."""

    service: str = Field(..., description="Application service name")
    version: str = Field(..., description="Application service version")
    docs_url: str = Field(..., description="Interactive Swagger documentation endpoint")
    health_url: str = Field(..., description="Health check endpoint")


class HealthHardwareInfo(BaseModel):
    """Hardware acceleration and GPU detection diagnostics."""

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


class HealthResponse(BaseModel):
    """Diagnostic response detailing runtime environment, GIL status, and model config."""

    status: str = Field(default="healthy", description="Service health status indicator")
    service: str = Field(..., description="Application service name")
    version: str = Field(..., description="Application service version")
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
    config: HealthConfigInfo = Field(..., description="Model and engine configuration")


def check_torch_cuda_status() -> bool | str:
    """Inspect PyTorch CUDA availability safely without crashing if PyTorch is absent."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:
        return str(exc)


def get_system_health() -> HealthResponse:
    """Collect runtime environment, GIL state, hardware acceleration, and model configuration."""
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        python_version=sys.version,
        free_threaded=not gil_enabled,
        gil_enabled=gil_enabled,
        hardware=HealthHardwareInfo(
            torch_cuda=check_torch_cuda_status(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            spaces_zero_gpu=os.environ.get("SPACES_ZERO_GPU"),
        ),
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
