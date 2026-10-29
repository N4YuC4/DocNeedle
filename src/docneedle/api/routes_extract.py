import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from docneedle.api.validation import (
    parse_extraction_schema,
    read_and_validate_uploads,
    validate_file_size,
)
from docneedle.cuda_compat import gpu_decorator
from docneedle.extraction.schemas import ExtractionItemQuery, ExtractionResponse
from docneedle.extraction.service import ExtractionService
from docneedle.ingest.models import DocumentChunk
from docneedle.ingest.service import IngestionService
from docneedle.retrieval.store import SessionVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Extraction"])

# Backward-compatibility aliases
_parse_extraction_schema = parse_extraction_schema
_validate_file_size = validate_file_size
_read_and_validate_uploads = read_and_validate_uploads


@gpu_decorator(duration=60)
def _sync_extract_session(
    files_data: list[tuple[str, bytes]],
    items_dicts: list[dict[str, Any]],
    enable_reranker: bool | None,
) -> dict[str, Any]:
    """Execute complete document ingestion and extraction synchronously with ZeroGPU leasing.

    Takes strictly picklable primitives (tuples, bytes, dicts) across the process boundary.
    """
    import concurrent.futures

    request_start = time.perf_counter()
    items = [ExtractionItemQuery.model_validate(it) for it in items_dicts]
    ingestion_service = IngestionService()
    extraction_service = ExtractionService()
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
                ingest_result = ingestion_service.ingest_bytes(
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
        store.add_chunks(all_chunks)
        logger.info(f"[INGESTION COMPLETE] Total Chunks indexed in RAM: {len(store.chunks)}")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(
                    asyncio.run,
                    extraction_service.extract_items(
                        items=items, store=store, enable_reranker=enable_reranker
                    ),
                ).result()
        else:
            response = asyncio.run(
                extraction_service.extract_items(
                    items=items, store=store, enable_reranker=enable_reranker
                )
            )

        total_time_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
        response.processing_time_ms = total_time_ms
        logger.info(
            f"[SESSION COMPLETE] Found {response.found_items}/{response.total_items} "
            f"in {response.processing_time_ms:.1f}ms"
        )
        return response.model_dump()

    finally:
        # Guarantee privacy: wipe RAM vectors and chunks immediately after completion
        store.clear()


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
    files_data = await read_and_validate_uploads(files)
    items = parse_extraction_schema(items_schema)
    items_dicts = [it.model_dump() for it in items]

    result_dict = await asyncio.to_thread(
        _sync_extract_session, files_data, items_dicts, enable_reranker
    )
    return ExtractionResponse.model_validate(result_dict)
