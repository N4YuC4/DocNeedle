import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docneedle.api.routes_extract import router as extract_router
from docneedle.api.routes_jobs import router as jobs_router
from docneedle.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-first local RAG document extraction microservice",
)

# Configure CORS for Vercel frontend, local development & local network (LAN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)
app.include_router(jobs_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint providing service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/healthz",
    }


@app.get("/health")
@app.get("/healthz")
async def health_check() -> dict[str, object]:
    """Health check endpoint detailing runtime environment and GIL state."""
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "python_version": sys.version,
        "free_threaded": not gil_enabled,
        "gil_enabled": gil_enabled,
        "config": {
            "embedding_model": settings.embedding_model,
            "embedding_quantization": settings.embedding_quantization,
            "llm_model": settings.llm_model_name,
            "llm_speculative_enabled": settings.llm_speculative_enabled,
            "multimodal_embedding": True,
            "multimodal_llm": True,
            "reranker_enabled": settings.reranker_enabled,
            "gpu_enabled": settings.gpu_enabled,
            "cpu_threads": settings.cpu_threads,
        },
    }
