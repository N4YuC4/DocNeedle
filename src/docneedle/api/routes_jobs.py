"""API routes for asynchronous job management and Server-Sent Events (SSE) streaming."""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from docneedle.api.validation import (
    parse_extraction_schema,
    read_and_validate_uploads,
)
from docneedle.jobs.manager import job_manager
from docneedle.jobs.models import JobCreateResponse, JobStatus, JobStatusResponse

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
    enable_reranker: bool | None = Form(
        default=None,
        description=(
            "Whether to run Qwen3 cross-encoder reranker (defaults to settings.reranker_enabled)"
        ),
    ),
) -> JobCreateResponse:
    """Submit an asynchronous extraction job.

    Returns 202 Accepted immediately with job tracking URLs.
    """
    files_data = await read_and_validate_uploads(files)
    items = parse_extraction_schema(items_schema)
    job_id = job_manager.create_job(total_items=len(items))
    job_info = job_manager.get_job(job_id)

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
        status=job_info.status if job_info else JobStatus.PENDING,
        poll_url=f"{base_path}/{job_id}",
        events_url=f"{base_path}/{job_id}/events",
        created_at=job_info.created_at if job_info else 0.0,
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
