---
title: DocNeedle
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12
app_file: app.py
pinned: false
---

<p align="center">
  <img src="assets/logo.svg" alt="DocNeedle Logo" width="110" />
</p>

<h1 align="center">DocNeedle</h1>

<p align="center">
  Privacy-first local document extraction microservice.<br/>
  Extract structured key-value data with verified citations and confidence scores using local in-process AI models.
</p>

---

## Features

- **Strict Zero-Disk Privacy:** Document chunks and vectors reside strictly in RAM and are purged immediately after extraction. No external vector databases, no cloud APIs.
- **In-Process AI:** Runs quantized Gemma 4, Harrier-OSS embeddings, RapidOCR, and BM25 directly in Python via `llama-cpp-python` and ONNX Runtime.
- **Free-Threaded Performance:** Runs on Python 3.14t (GIL disabled) with speculative prompt lookup decoding and 8-bit quantized KV caching.
- **Web UI & API:** Unified dark-mode Gradio 6 dashboard and FastAPI REST API with real-time Server-Sent Events (SSE) streaming on port `7860`.

---

## Quickstart

### Prerequisites (Linux)
```bash
sudo apt-get install -y libgl1 libglib2.0-0  # Required for OCR/OpenCV
```

### Setup & Run
```bash
# 1. Clone & create Python 3.14t virtual environment
git clone https://github.com/your-username/DocNeedle.git && cd DocNeedle
uv python install 3.14t
uv venv --python 3.14t && source .venv/bin/activate

# 2. Install dependencies & download local models (~3.7 GB)
uv sync
uv run python scripts/download_models.py

# 3. Launch application (UI + API)
uv run python app.py
```

- **Web Dashboard:** `http://localhost:7860`
- **Swagger API Docs:** `http://localhost:7860/docs`
- **Health Check:** `http://localhost:7860/healthz`

---

## API Usage

### Extract Data (`POST /v1/extract`)

```bash
curl -X POST "http://localhost:7860/v1/extract" \
  -F "files=@contract.pdf" \
  -F 'schema=[
    {"key": "monthly_rent", "description": "Agreed monthly rent amount", "type": "money"},
    {"key": "tenant_name", "description": "Full name of the tenant", "type": "person"}
  ]'
```

**Response:**
```json
{
  "status": "completed",
  "processing_time_ms": 1240.0,
  "total_items": 2,
  "found_items": 2,
  "results": [
    {
      "key": "monthly_rent",
      "found": true,
      "value": "25.000 TL",
      "confidence": 0.96,
      "evidence": {
        "document": "contract.pdf",
        "page": 1,
        "snippet": "Kira Bedeli: Aylık 25.000 TL olup her ayın 5. gününe kadar..."
      },
      "thinking": null
    },
    {
      "key": "tenant_name",
      "found": true,
      "value": "Tolga Arslan",
      "confidence": 0.95,
      "evidence": {
        "document": "contract.pdf",
        "page": 1,
        "snippet": "KİRACI: Tolga Arslan (T.C. Kimlik No: 12345678901)"
      },
      "thinking": null
    }
  ]
}
```

*For large documents, use `POST /v1/jobs` and stream real-time progress via `GET /v1/jobs/{job_id}/events` (SSE).*

---

## Configuration

Settings are configured in `config.toml` or overridden via environment variables:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `gpu_enabled` | `true` | Enable CUDA GPU offloading (`true` / `false`) |
| `llm_n_gpu_layers` | `-1` | Layers offloaded to GPU (`-1` = all, `0` = CPU) |
| `ocr_enabled` | `true` | Ephemeral RapidOCR for scanned files |
| `retrieval_top_k` | `10` | Candidate chunks retrieved via hybrid RRF |
| `reranker_enabled`| `true` | Qwen3 cross-encoder logit reranker |

---

## Docker

Run the complete stack with Docker Compose:

```bash
docker compose up --build -d
```

---

## Development

```bash
# Run test suite
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

---

## License

[Apache 2.0](LICENSE)
