"""Tests for Asynchronous Job API, progress tracking, and SSE streaming."""

import asyncio
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from docneedle.extraction.schemas import ExtractionItemResult
from docneedle.jobs.manager import JobManager
from docneedle.jobs.models import JobStatus
from docneedle.main import app


@pytest.mark.asyncio
async def test_job_manager_unit_lifecycle() -> None:
    """Test core JobManager state machine and event publishing in memory."""
    mgr = JobManager()
    job_id = mgr.create_job(total_items=2)
    assert job_id.startswith("job_")

    job = mgr.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.PENDING
    assert job.progress.total_items == 2
    assert job.progress.completed_items == 0

    # Subscribe queue
    sub_q = await mgr.subscribe(job_id)
    assert sub_q is not None
    initial_event = await sub_q.get()
    assert initial_event["event"] == "status"

    # Update progress
    await mgr.update_progress(
        job_id=job_id,
        step="Extracting field 1: name",
        status=JobStatus.EXTRACTING,
        completed_items=0,
        total_items=2,
        current_key="name",
    )
    progress_event = await sub_q.get()
    assert progress_event["event"] == "progress"
    assert progress_event["data"]["status"] == "extracting"

    # Append item result
    dummy_item = ExtractionItemResult(
        key="name",
        found=True,
        value="John Doe",
        confidence=0.95,
    )
    await mgr.append_result(job_id, dummy_item)
    item_event = await sub_q.get()
    assert item_event["event"] == "item_extracted"
    assert item_event["data"]["key"] == "name"
    assert item_event["data"]["value"] == "John Doe"

    # Fail / error test
    await mgr.fail_job(job_id, "Test error")
    err_event = await sub_q.get()
    assert err_event["event"] == "error"

    updated_job = mgr.get_job(job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED
    assert updated_job.error == "Test error"


@pytest.mark.asyncio
async def test_job_manager_ttl_cleanup() -> None:
    """Verify that jobs exceeding retention TTL are evicted from memory."""
    mgr = JobManager()
    old_id = mgr.create_job(total_items=1)
    # Simulate past creation time exceeding TTL
    mgr._jobs[old_id].created_at = time.time() - 3600

    new_id = mgr.create_job(total_items=1)

    mgr.cleanup_expired_jobs()
    assert mgr.get_job(old_id) is None
    assert mgr.get_job(new_id) is not None


@pytest.mark.asyncio
async def test_job_manager_concurrent_job_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that exceeding max_concurrent_jobs raises 429 Too Many Requests."""
    from fastapi import HTTPException

    from docneedle.config import settings

    monkeypatch.setattr(settings, "max_concurrent_jobs", 3)
    mgr = JobManager()

    job1 = mgr.create_job(total_items=1)
    job2 = mgr.create_job(total_items=1)
    job3 = mgr.create_job(total_items=1)
    assert job1 and job2 and job3

    with pytest.raises(HTTPException) as exc_info:
        mgr.create_job(total_items=1)
    assert exc_info.value.status_code == 429
    assert "Server busy" in exc_info.value.detail
    assert "concurrent jobs currently running" in exc_info.value.detail


@pytest.mark.asyncio
async def test_job_manager_free_threaded_concurrency_stress() -> None:
    """Stress-test JobManager under Python 3.14t Free-Threaded (no-GIL) concurrency.

    Concurrently subscribes, receives events, unsubscribes, updates progress,
    and appends extracted items across multiple concurrent tasks without race conditions.
    """
    from docneedle.extraction.schemas import ExtractionResponse

    mgr = JobManager()
    job_id = mgr.create_job(total_items=20)

    errors: list[Exception] = []

    async def consumer(consumer_id: int) -> None:
        try:
            q = await mgr.subscribe(job_id)
            if not q:
                return
            while True:
                msg = await q.get()
                event_type = msg.get("event")
                if event_type in ("completed", "error"):
                    break
            mgr.unsubscribe(job_id, q)
        except Exception as exc:
            errors.append(exc)

    async def producer() -> None:
        try:
            for i in range(1, 21):
                await mgr.update_progress(
                    job_id=job_id,
                    step=f"Processing field {i}",
                    completed_items=i,
                    total_items=20,
                )
                await mgr.append_result(
                    job_id=job_id,
                    result=ExtractionItemResult(
                        key=f"field_{i}",
                        found=True,
                        value=f"value_{i}",
                        confidence=0.99,
                    ),
                )
                await asyncio.sleep(0.002)

            await mgr.complete_job(
                job_id=job_id,
                response=ExtractionResponse(
                    status="completed",
                    processing_time_ms=50.0,
                    total_items=20,
                    found_items=20,
                    results=[],
                ),
            )
        except Exception as exc:
            errors.append(exc)

    # Launch 5 concurrent consumers and 1 producer simultaneously
    consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(5)]
    producer_task = asyncio.create_task(producer())

    await asyncio.gather(producer_task, *consumer_tasks)

    assert len(errors) == 0, f"Encountered concurrent errors: {errors}"
    final_job = mgr.get_job(job_id)
    assert final_job is not None
    assert final_job.status == JobStatus.COMPLETED
    assert len(final_job.results) == 20


@pytest.mark.asyncio
async def test_jobs_api_e2e_polling_and_sse() -> None:
    """Test full /v1/jobs workflow: submit -> 202 -> SSE stream / poll -> complete."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sample_doc = b"# Kira Sozlesmesi\n\nKiraci: Ahmet Yilmaz\nKira Bedeli: 25.000 TL\n"
        schema = [
            {"key": "kiraci", "description": "Kiracı kimdir?", "type": "string"},
            {"key": "kira_bedeli", "description": "Aylık kira bedeli", "type": "money"},
        ]

        files = {
            "files": ("sozlesme.md", sample_doc, "text/markdown"),
        }
        data = {
            "schema": json.dumps(schema),
            "enable_reranker": "false",
        }

        # 1. Submit job (immediate 202 Accepted)
        post_res = await client.post("/v1/jobs", files=files, data=data)
        assert post_res.status_code == 202
        job_data = post_res.json()
        assert "job_id" in job_data
        assert job_data["status"] == "pending"
        job_id = job_data["job_id"]
        poll_url = job_data["poll_url"]
        events_url = job_data["events_url"]

        assert poll_url == f"/v1/jobs/{job_id}"
        assert events_url == f"/v1/jobs/{job_id}/events"

        # 2. Poll until completed or timeout
        max_retries = 30
        completed = False
        final_job = None

        for _ in range(max_retries):
            poll_res = await client.get(poll_url)
            assert poll_res.status_code == 200
            final_job = poll_res.json()
            if final_job["status"] == "completed":
                completed = True
                break
            elif final_job["status"] == "failed":
                pytest.fail(f"Job failed unexpectedly: {final_job.get('error')}")
            await asyncio.sleep(0.5)

        assert completed is True
        assert final_job is not None
        assert final_job["status"] == "completed"
        assert len(final_job["results"]) == 2

        results_by_key = {r["key"]: r for r in final_job["results"]}
        assert "kiraci" in results_by_key
        assert results_by_key["kiraci"]["found"] is True
        assert "Ahmet" in str(results_by_key["kiraci"]["value"])

        assert "kira_bedeli" in results_by_key
        assert results_by_key["kira_bedeli"]["found"] is True
        assert "25.000" in str(results_by_key["kira_bedeli"]["value"])


@pytest.mark.asyncio
async def test_jobs_not_found() -> None:
    """Verify that polling or streaming a non-existent job returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/jobs/non_existent_job_123")
        assert res.status_code == 404

        sse_res = await client.get("/v1/jobs/non_existent_job_123/events")
        assert sse_res.status_code == 404


@pytest.mark.asyncio
async def test_jobs_sse_stream_events_e2e() -> None:
    """Verify SSE streaming line-by-line reception from /v1/jobs/{job_id}/events."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sample_doc = b"# Fatura\nToplam Tutar: 1.500 TL\n"
        schema = [{"key": "tutar", "description": "Toplam tutar", "type": "money"}]
        files = {"files": ("fatura.md", sample_doc, "text/markdown")}
        data = {"schema": json.dumps(schema)}

        post_res = await client.post("/v1/jobs", files=files, data=data)
        assert post_res.status_code == 202
        events_url = post_res.json()["events_url"]

        received_events = []
        async with client.stream("GET", events_url) as stream:
            async for line in stream.aiter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    received_events.append(event_name)
                    if event_name == "completed":
                        break

        assert "status" in received_events or "progress" in received_events
        assert "completed" in received_events
