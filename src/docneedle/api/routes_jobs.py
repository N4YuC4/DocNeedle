"""API routes for asynchronous job management and Server-Sent Events (SSE) streaming."""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from docneedle.api.routes_extract import _parse_extraction_schema, _validate_file_size
from docneedle.config import settings
from docneedle.jobs.manager import job_manager
from docneedle.jobs.models import JobCreateResponse, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Jobs"])


@router.post("/jobs", response_model=JobCreateResponse, status_code=202)
async def create_extraction_job(
    request: Request,
    files: list[UploadFile] = File(
        ...,
        description="One or more document files (PDF, DOCX, XLSX, images)",
    ),
    items_schema: str = Form(
        ...,
        alias="schema",
        description="JSON string list of requested items to extract",
    ),
    enable_reranker: bool = Form(
        default=False,
        description="Whether to run Qwen3 cross-encoder reranker on CPU (default: False)",
    ),
) -> JobCreateResponse:
    """Submit an asynchronous extraction job.

    Returns 202 Accepted immediately with job tracking URLs.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one document file must be uploaded.")

    items = _parse_extraction_schema(items_schema)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "uploaded_document"
        _validate_file_size(filename, content, max_bytes)
        files_data.append((filename, content))

    job_id = job_manager.create_job(total_items=len(items))

    # Launch background worker
    asyncio.create_task(
        job_manager.run_extraction_worker(
            job_id=job_id,
            files_data=files_data,
            items=items,
            enable_reranker=enable_reranker,
        )
    )

    base_path = "/v1/jobs"
    return JobCreateResponse(
        job_id=job_id,
        status=job_manager.get_job(job_id).status,  # type: ignore
        poll_url=f"{base_path}/{job_id}",
        events_url=f"{base_path}/{job_id}/events",
        created_at=job_manager.get_job(job_id).created_at,  # type: ignore
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the real-time progress and results of an extraction job."""
    job_info = job_manager.get_job(job_id)
    if not job_info:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or has expired from in-memory cache.",
        )
    return job_info


@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> Response:
    """Stream real-time progress and extracted field items via Server-Sent Events (SSE)."""
    job_info = job_manager.get_job(job_id)
    if not job_info:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or has expired from in-memory cache.",
        )

    async def _sse_generator() -> AsyncGenerator[str, None]:
        q = await job_manager.subscribe(job_id)
        if not q:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    event_type = msg.get("event", "message")
                    data_json = json.dumps(msg.get("data", {}))
                    yield f"event: {event_type}\ndata: {data_json}\n\n"
                    if event_type in ("completed", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat for Nginx/Cloudflare/ALB proxies
                    yield ": keep-alive\n\n"
        finally:
            job_manager.unsubscribe(job_id, q)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering in Nginx
        },
    )
