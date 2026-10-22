import pytest
from httpx import ASGITransport, AsyncClient

from docneedle.main import app


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
