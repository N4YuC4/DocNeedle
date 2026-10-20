FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHON_GIL=0 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies for document parsing and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification
COPY pyproject.toml README.md* /app/

# Install dependencies
RUN uv pip install --system -e .

# Copy source code and download script
COPY src /app/src
COPY scripts /app/scripts
COPY app.py /app/app.py

EXPOSE 7860 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

