"""Thread-safe in-memory Job Manager for asynchronous document extraction and SSE streaming."""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from docneedle.config import settings
from docneedle.extraction.schemas import (
    ExtractionItemQuery,
    ExtractionItemResult,
    ExtractionResponse,
)
from docneedle.extraction.service import ExtractionService
from docneedle.ingest.models import DocumentChunk
from docneedle.ingest.service import IngestionService
from docneedle.jobs.models import (
    JobProgress,
    JobStatus,
    JobStatusResponse,
)
from docneedle.retrieval.store import SessionVectorStore

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Internal representation of an active or completed in-memory extraction job."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    results: list[ExtractionItemResult] = field(default_factory=list)
    response: ExtractionResponse | None = None
    error: str | None = None
    queues: list[asyncio.Queue] = field(default_factory=list)


class JobManager:
    """Manages asynchronous extraction jobs, progress tracking, and SSE subscriptions in RAM."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self, total_items: int) -> str:
        """Register a new job in PENDING state."""
        self.cleanup_expired_jobs()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=JobProgress(
                current_step="Queued for processing",
                total_items=total_items,
                completed_items=0,
                percent=0.0,
            ),
        )
        with self._lock:
            self._jobs[job_id] = record
        return job_id

    def get_job(self, job_id: str) -> JobStatusResponse | None:
        """Return the current public state of a job, or None if not found/expired."""
        self.cleanup_expired_jobs()
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None

            proc_ms = None
            if record.completed_at and record.created_at:
                proc_ms = round((record.completed_at - record.created_at) * 1000.0, 2)

            return JobStatusResponse(
                job_id=record.job_id,
                status=record.status,
                progress=record.progress.model_copy(),
                created_at=record.created_at,
                completed_at=record.completed_at,
                processing_time_ms=proc_ms,
                results=list(record.results),
                response=record.response,
                error=record.error,
            )

    async def subscribe(self, job_id: str) -> asyncio.Queue | None:
        """Subscribe to real-time events for an active job."""
        record = self._jobs.get(job_id)
        if not record:
            return None

        q: asyncio.Queue = asyncio.Queue()
        record.queues.append(q)

        # Replay current status immediately upon connecting
        initial_payload = {
            "status": record.status.value,
            "progress": record.progress.model_dump(),
            "results": [r.model_dump() for r in record.results],
        }
        await q.put({"event": "status", "data": initial_payload})

        if record.status == JobStatus.COMPLETED and record.response:
            await q.put({"event": "completed", "data": record.response.model_dump()})
        elif record.status == JobStatus.FAILED and record.error:
            await q.put({"event": "error", "data": {"error": record.error}})

        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue when client disconnects."""
        record = self._jobs.get(job_id)
        if record and q in record.queues:
            record.queues.remove(q)

    async def _emit_event(self, record: JobRecord, event_type: str, data: Any) -> None:
        """Broadcast an event payload to all active SSE subscribers of a job."""
        dead_queues = []
        for q in record.queues:
            try:
                q.put_nowait({"event": event_type, "data": data})
            except Exception:
                dead_queues.append(q)
        for dq in dead_queues:
            if dq in record.queues:
                record.queues.remove(dq)

    async def update_progress(
        self,
        job_id: str,
        step: str,
        status: JobStatus | None = None,
        completed_items: int | None = None,
        total_items: int | None = None,
        current_key: str | None = None,
        percent: float | None = None,
    ) -> None:
        """Update live progress and emit a progress event to listeners."""
        record = self._jobs.get(job_id)
        if not record:
            return

        if status:
            record.status = status

        if total_items is not None:
            record.progress.total_items = total_items
        if completed_items is not None:
            record.progress.completed_items = completed_items
        record.progress.current_step = step
        record.progress.current_key = current_key

        if percent is not None:
            record.progress.percent = round(percent, 1)
        elif record.progress.total_items > 0:
            # Reserve 0-20% for ingestion, 20-100% for extraction
            base_ratio = record.progress.completed_items / record.progress.total_items
            record.progress.percent = round(20.0 + (base_ratio * 80.0), 1)

        await self._emit_event(
            record,
            "progress",
            {
                "status": record.status.value,
                "progress": record.progress.model_dump(),
            },
        )

    async def append_result(self, job_id: str, result: ExtractionItemResult) -> None:
        """Append an extracted field result and notify subscribers immediately."""
        record = self._jobs.get(job_id)
        if not record:
            return
        record.results.append(result)
        await self._emit_event(record, "item_extracted", result.model_dump())

    async def complete_job(self, job_id: str, response: ExtractionResponse) -> None:
        """Mark job as completed and publish final payload."""
        record = self._jobs.get(job_id)
        if not record:
            return
        record.status = JobStatus.COMPLETED
        record.completed_at = time.time()
        if response and record.created_at:
            proc_ms = round((record.completed_at - record.created_at) * 1000.0, 2)
            response.processing_time_ms = proc_ms
        record.response = response
        if response and response.results:
            record.results = list(response.results)
        record.progress.current_step = "Extraction completed successfully."
        record.progress.percent = 100.0
        record.progress.completed_items = record.progress.total_items
        await self._emit_event(record, "completed", response.model_dump())

    async def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed and publish error event."""
        record = self._jobs.get(job_id)
        if not record:
            return
        record.status = JobStatus.FAILED
        record.completed_at = time.time()
        record.error = error_message
        record.progress.current_step = f"Failed: {error_message}"
        await self._emit_event(record, "error", {"error": error_message})

    def cleanup_expired_jobs(self) -> None:
        """Evict jobs that have exceeded the retention TTL from RAM."""
        ttl_seconds = settings.job_retention_minutes * 60
        now = time.time()
        with self._lock:
            expired_ids = [
                jid for jid, rec in self._jobs.items() if (now - rec.created_at) > ttl_seconds
            ]
            for jid in expired_ids:
                logger.info(f"[JOB CLEANUP] Evicting expired job from RAM: {jid}")
                del self._jobs[jid]

    async def run_extraction_worker(
        self,
        job_id: str,
        files_data: list[tuple[str, bytes]],
        items: list[ExtractionItemQuery],
        enable_reranker: bool | None = None,
    ) -> None:
        """Execute the complete document ingestion, indexing, and extraction workflow
        asynchronously.
        """
        ingestion_service = IngestionService()
        extraction_service = ExtractionService()
        store = SessionVectorStore()

        try:
            total_queries = len(items)
            await self.update_progress(
                job_id=job_id,
                step=f"Parsing {len(files_data)} uploaded document(s)...",
                status=JobStatus.PARSING,
                completed_items=0,
                total_items=total_queries,
                percent=5.0,
            )

            all_chunks: list[DocumentChunk] = []
            with ingestion_service.parser:
                for idx, (filename, content) in enumerate(files_data, 1):
                    await self.update_progress(
                        job_id=job_id,
                        step=f"Parsing document {idx}/{len(files_data)}: {filename}...",
                        percent=5.0 + (idx / len(files_data) * 10.0),
                    )
                    ingest_result = await asyncio.to_thread(
                        ingestion_service.ingest_bytes,
                        content=content,
                        filename=filename,
                    )
                    all_chunks.extend(ingest_result.chunks)

            if not all_chunks:
                await self.fail_job(
                    job_id, "Could not extract any readable text from the uploaded documents."
                )
                return

            await self.update_progress(
                job_id=job_id,
                step=f"Indexing {len(all_chunks)} chunks in RAM (Dense + BM25)...",
                status=JobStatus.INDEXING,
                percent=18.0,
            )
            await asyncio.to_thread(store.add_chunks, all_chunks)

            await self.update_progress(
                job_id=job_id,
                step=f"Beginning field extraction (0/{total_queries})...",
                status=JobStatus.EXTRACTING,
                percent=20.0,
            )

            async def _on_item_extracted(
                item_res: ExtractionItemResult,
                completed_idx: int,
                total_count: int,
            ) -> None:
                await self.append_result(job_id, item_res)
                next_key = items[completed_idx].key if completed_idx < total_count else None
                step_msg = (
                    f"Extracted {completed_idx}/{total_count} fields. Next: '{next_key}'"
                    if next_key
                    else f"Completed extraction of all {total_count} fields."
                )
                await self.update_progress(
                    job_id=job_id,
                    step=step_msg,
                    completed_items=completed_idx,
                    total_items=total_count,
                    current_key=next_key,
                )

            response = await extraction_service.extract_items(
                items=items,
                store=store,
                enable_reranker=enable_reranker,
                on_item_extracted=_on_item_extracted,
            )

            await self.complete_job(job_id, response)
            logger.info(f"[JOB COMPLETE] {job_id} processed in {response.processing_time_ms}ms")

        except Exception as e:
            logger.error(f"[JOB FAILED] {job_id} encountered error: {e}", exc_info=True)
            await self.fail_job(job_id, str(e))

        finally:
            # Guarantee zero-disk privacy: strictly purge all RAM vectors and chunks
            store.clear()


# Global singleton instance
job_manager = JobManager()
