import pytest
from httpx import ASGITransport, AsyncClient

from docneedle.main import (
    HealthModelsInfo,
    HealthResponse,
    ModelStatusItem,
    app,
    check_cuda_backend_status,
    create_app,
    get_system_health,
)


@pytest.fixture
def models_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock local model file existence for weight-free CI test runners."""
    from docneedle import main as m

    def fake_check_models_status() -> HealthModelsInfo:
        item = ModelStatusItem(name="mock.gguf", ready=True, size_mb=123.4)
        return HealthModelsInfo(
            all_ready=True,
            llm=item,
            llm_mmproj=item,
            embedding=item,
            embedding_mmproj=item,
            reranker=item,
        )

    monkeypatch.setattr(m, "check_models_status", fake_check_models_status)


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
async def test_healthz_endpoint(models_ready) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "DocNeedle"
        assert "3.14" in data["python_version"]
        assert data["hardware"]["backend"] == "llama.cpp"
        assert isinstance(data["hardware"]["gpu_offload_supported"], bool)
        assert isinstance(data["hardware"]["cuda_runtime_detected"], bool)


@pytest.mark.asyncio
async def test_healthz_endpoint_degraded_when_models_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify endpoint explicitly signals degraded status when weights are absent."""
    from docneedle import main as m

    def fake_check_missing_models() -> HealthModelsInfo:
        item = ModelStatusItem(name="mock.gguf", ready=False, size_mb=None)
        return HealthModelsInfo(
            all_ready=False,
            llm=item,
            llm_mmproj=item,
            embedding=item,
            embedding_mmproj=item,
            reranker=item,
        )

    monkeypatch.setattr(m, "check_models_status", fake_check_missing_models)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["models"]["all_ready"] is False


@pytest.mark.asyncio
async def test_health_alias_endpoint(models_ready) -> None:
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
        redoc = await client.get("/redoc")
        assert redoc.status_code == 200
        assert "redoc" in redoc.text.lower()


@pytest.mark.asyncio
async def test_launched_gradio_app_docs_and_openapi_routes() -> None:
    """Verify Blocks.launch(app_kwargs=_APP_KWARGS) + _mount_api_routes exposes /docs and /v1/*."""
    from gradio.routes import App

    from app import _APP_KWARGS, _mount_api_routes, demo

    launched_app = App.create_app(demo, app_kwargs=dict(_APP_KWARGS))
    _mount_api_routes(launched_app)

    transport = ASGITransport(app=launched_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        docs = await client.get("/docs")
        assert docs.status_code == 200
        assert "swagger-ui" in docs.text
        assert "DocNeedle" in docs.text

        redoc = await client.get("/redoc")
        assert redoc.status_code == 200

        openapi = await client.get("/openapi.json")
        assert openapi.status_code == 200
        spec = openapi.json()
        assert spec["info"]["title"] == "DocNeedle"
        paths = set(spec["paths"].keys())
        assert "/healthz" in paths
        assert "/v1/extract" in paths
        assert "/v1/jobs" in paths


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
    assert diag.hardware.device in ("cuda", "cpu")
    assert diag.hardware.backend == "llama.cpp"
    assert isinstance(diag.hardware.gpu_offload_supported, bool)
    assert isinstance(diag.hardware.cuda_runtime_detected, bool)
    assert isinstance(diag.models.all_ready, bool)
    assert isinstance(diag.models.llm.ready, bool)
    assert isinstance(diag.jobs.active_jobs, int)
    assert isinstance(diag.jobs.total_cached_jobs, int)
    assert isinstance(diag.config.cpu_threads, int)
    assert diag.config.cpu_threads > 0


def test_ui_health_sync(models_ready) -> None:
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


def test_check_cuda_backend_status_non_crashing() -> None:
    """Verify CUDA status check returns expected dictionary without crashing."""
    status = check_cuda_backend_status()
    assert isinstance(status, dict)
    assert "gpu_offload_supported" in status
    assert "cuda_runtime_detected" in status
