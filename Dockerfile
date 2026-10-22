FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHON_GIL=0 \
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
RUN uv pip install --system -r requirements.txt || uv pip install --system .

# Copy configuration, static assets, scripts and application source code
COPY config.toml /app/config.toml
COPY assets /app/assets
COPY scripts /app/scripts
COPY src /app/src
COPY app.py /app/app.py

# Install docneedle package itself without re-fetching dependencies
RUN uv pip install --system --no-deps .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:7860/healthz || exit 1

# Start unified Gradio UI + FastAPI backend
CMD ["python", "app.py"]
