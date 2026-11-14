"""Unit tests for health and root endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify that /health returns HTTP 200 and healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert data["services"]["api"] == "online"


@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify that root / returns basic info and links to docs and health."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


def test_settings_load():
    """Verify settings loads defaults without error."""
    settings = get_settings()
    assert settings.EMBEDDING_DIMENSION == 384
    assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
    assert settings.ALLOWED_FILE_TYPES == ["pdf", "txt"]
    assert settings.QDRANT_PORT == 6333
    assert settings.REDIS_PORT == 6379

