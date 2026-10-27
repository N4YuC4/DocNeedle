import asyncio
import json
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from docneedle.config import settings
from docneedle.cuda_compat import gpu_decorator
from docneedle.extraction.schemas import ExtractionItemQuery, ExtractionResponse
from docneedle.extraction.service import ExtractionService
from docneedle.ingest.models import DocumentChunk
from docneedle.ingest.service import IngestionService
from docneedle.retrieval.store import SessionVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Extraction"])


def _parse_extraction_schema(items_schema: str) -> list[ExtractionItemQuery]:
    """Parse and validate extraction schema JSON payload into query models."""
    try:
        raw_items = json.loads(items_schema)
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        items = [ExtractionItemQuery.model_validate(it) for it in raw_items]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema JSON format: {e}") from e

    if not items:
        raise HTTPException(
            status_code=400,
            detail="At least one extraction item must be provided in schema.",
        )
    return items


def _validate_file_size(filename: str, content: bytes, max_bytes: int) -> None:
    """Enforce maximum upload payload constraint per file."""
    if max_bytes > 0 and len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{filename}' exceeds maximum allowed upload size of "
                f"{settings.max_upload_size_mb} MB."
            ),
        )


async def _read_and_validate_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read uploaded files into memory and validate non-empty list and per-file size limits."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one document file must be uploaded.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "uploaded_document"
        _validate_file_size(filename, content, max_bytes)
        files_data.append((filename, content))
    return files_data


@gpu_decorator(duration=60)
def _sync_llm_extract(
    items: list[ExtractionItemQuery],
    store: SessionVectorStore,
    enable_reranker: bool | None,
) -> ExtractionResponse:
    """Execute LLM neural extraction synchronously within the ZeroGPU lease."""
    import concurrent.futures

    extraction_service = ExtractionService()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            extraction_service.extract_items(
                items=items, store=store, enable_reranker=enable_reranker
            ),
        ).result()


@router.post("/extract", response_model=ExtractionResponse)
async def extract_from_documents(
    files: list[UploadFile] = File(
        ...,
        description="One or more document files (PDF, DOCX, XLSX, images)",
    ),
    items_schema: str = Form(
        ...,
        alias="schema",
        description="JSON string list of requested items to extract",
    ),
    enable_reranker: bool | None = Form(
        default=None,
        description=(
            "Whether to run Qwen3 cross-encoder reranker (defaults to settings.reranker_enabled)"
        ),
    ),
) -> ExtractionResponse:
    """Extract structured data items from documents in a single privacy-preserving session."""
    files_data = await _read_and_validate_uploads(files)
    items = _parse_extraction_schema(items_schema)
    request_start = time.perf_counter()
    ingestion_service = IngestionService()
    store = SessionVectorStore()

    try:
        filenames = [fname for fname, _ in files_data]
        query_keys = [it.key for it in items]
        logger.info(f"[EXTRACTION SESSION] Files: {filenames} | Queries: {query_keys}")
        all_chunks: list[DocumentChunk] = []

        with ingestion_service.parser:
            for idx, (filename, content) in enumerate(files_data, 1):
                kb_size = len(content) / 1024
                logger.info(
                    f"--> [{idx}/{len(files_data)}] Parsing: {filename} ({kb_size:.1f} KB)..."
                )
                ingest_result = await asyncio.to_thread(
                    ingestion_service.ingest_bytes,
                    content=content,
                    filename=filename,
                )
                all_chunks.extend(ingest_result.chunks)
                logger.info(
                    f"    Extracted {len(ingest_result.chunks)} chunks (Total: {len(all_chunks)})."
                )

        if not all_chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any readable text from the uploaded documents.",
            )

        logger.info(
            f"--> Indexing {len(all_chunks)} chunks into vector store & building BM25 index..."
        )
        await asyncio.to_thread(store.add_chunks, all_chunks)
        logger.info(f"[INGESTION COMPLETE] Total Chunks indexed in RAM: {len(store.chunks)}")

        # Execute only the LLM inference step within the ZeroGPU lease
        response = await asyncio.to_thread(
            _sync_llm_extract, items, store, enable_reranker
        )
        total_time_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
        response.processing_time_ms = total_time_ms
        logger.info(
            f"[SESSION COMPLETE] Found {response.found_items}/{response.total_items} "
            f"in {response.processing_time_ms:.1f}ms"
        )
        return response

    finally:
        # Guarantee privacy: wipe RAM vectors and chunks immediately after completion
        store.clear()
