import pytest
from httpx import ASGITransport, AsyncClient

from docneedle.main import (
    HealthResponse,
    app,
    check_torch_cuda_status,
    create_app,
    get_system_health,
)


@pytest.mark.asyncio
async def test_root_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "DocNeedle"
        assert "/docs" in data["docs_url"]


@pytest.mark.asyncio
async def test_healthz_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "DocNeedle"
        assert isinstance(data["gil_enabled"], bool)
        assert data["free_threaded"] == (not data["gil_enabled"])
        assert "3.14" in data["python_version"]


@pytest.mark.asyncio
async def test_health_alias_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "DocNeedle"


@pytest.mark.asyncio
async def test_docs_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text


@pytest.mark.asyncio
async def test_unified_app_docs_endpoint() -> None:
    from app import app as unified_app

    transport = ASGITransport(app=unified_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text


def test_create_app_factory() -> None:
    """Verify application factory creates an isolated FastAPI instance with expected routes."""
    fresh_app = create_app()
    assert fresh_app is not app
    paths = set(fresh_app.openapi()["paths"].keys())
    assert "/" in paths
    assert "/healthz" in paths
    assert "/v1/extract" in paths
    assert "/v1/jobs" in paths


def test_get_system_health_model() -> None:
    """Verify typed health diagnostic helper returns a valid HealthResponse model."""
    diag = get_system_health()
    assert isinstance(diag, HealthResponse)
    assert diag.status in ("healthy", "degraded")
    assert diag.service == "DocNeedle"
    assert isinstance(diag.uptime_seconds, float)
    assert diag.uptime_seconds >= 0.0
    assert isinstance(diag.process_memory_mb, float)
    assert diag.process_memory_mb > 0.0
    assert isinstance(diag.free_threaded, bool)
    assert isinstance(diag.gil_enabled, bool)
    assert diag.free_threaded != diag.gil_enabled
    assert diag.hardware.device in ("cuda", "cpu")
    assert diag.hardware.torch_cuda in (True, False) or isinstance(diag.hardware.torch_cuda, str)
    assert isinstance(diag.models.all_ready, bool)
    assert isinstance(diag.models.llm.ready, bool)
    assert isinstance(diag.jobs.active_jobs, int)
    assert isinstance(diag.jobs.total_cached_jobs, int)
    assert isinstance(diag.config.cpu_threads, int)
    assert diag.config.cpu_threads > 0


def test_ui_health_sync() -> None:
    """Verify app.py check_system_health synchronizes dynamically with get_system_health."""
    from app import check_system_health

    ui_data = check_system_health()
    assert isinstance(ui_data, dict)
    assert ui_data["service"] == "DocNeedle"
    assert "uptime_seconds" in ui_data
    assert "process_memory_mb" in ui_data
    assert "models" in ui_data
    assert "jobs" in ui_data
    assert ui_data["models"]["llm"]["ready"] is True
    assert ui_data["models"]["llm"]["size_mb"] is not None


def test_check_torch_cuda_status_non_crashing() -> None:
    """Verify CUDA status check returns a bool or string error without crashing."""
    status = check_torch_cuda_status()
    assert isinstance(status, (bool, str))
