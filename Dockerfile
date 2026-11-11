FROM python:3.14-slim AS prod

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies for document parsing, OCR, and native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency specifications first for Docker layer caching
COPY pyproject.toml uv.lock* requirements.txt* packages.txt* README.md* /app/

# Install dependencies into system environment
RUN uv pip install --system -r requirements.txt

# Copy configuration, static assets, scripts and application source code
COPY config.toml /app/config.toml
COPY assets /app/assets
COPY scripts /app/scripts
COPY src /app/src
COPY app.py /app/app.py

# Optional build arg for CPU-only environments (e.g. CI runners lacking NVIDIA GPU drivers)
ARG INSTALL_CPU_LLAMA=false
RUN if [ "$INSTALL_CPU_LLAMA" = "true" ]; then \
        uv pip uninstall --system llama-cpp-python && \
        uv pip install --system "llama-cpp-python @ https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"; \
    fi

# Install docneedle package itself without re-fetching dependencies
RUN uv pip install --system --no-deps .

# Pre-bake Hugging Face tokenizer cache into image layer for air-gapped offline execution
ENV HF_HOME=/app/hf-cache
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')"

# Port 7860 is the standard port for the unified Gradio UI + FastAPI root app (e.g. Hugging Face Spaces).
# For headless microservice API deployments on port 8000, see docker-compose.prod.yml.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:7860/healthz || exit 1

# Start unified Gradio UI + FastAPI backend
CMD ["python", "app.py"]

# ==============================================================================
# Dedicated test stage for container-enforced read-only rootfs validation in CI
# ==============================================================================
FROM prod AS test
COPY tests /app/tests
RUN uv pip install --system pytest pytest-asyncio anyio
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-m", "pytest", "tests/test_zero_disk.py", "-v", "-p", "no:cacheprovider", "-o", "cache_dir=/tmp/.pytest_cache"]
