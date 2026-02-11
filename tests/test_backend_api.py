"""Tests for the FastAPI backend endpoints.

Uses httpx.AsyncClient with ASGITransport — no real server needed.
"""

import sys
import pathlib
import pytest
import pytest_asyncio

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from httpx import ASGITransport, AsyncClient
from backend.main import app


@pytest_asyncio.fixture
async def client():
    # Manually populate _state so endpoints work without the full lifespan
    from backend.main import _state
    if "coordinator" not in _state:
        import sys as _sys, pathlib as _pathlib
        root = str(_pathlib.Path(__file__).resolve().parent.parent)
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from agents.coordinator_agent import CoordinatorAgent
        from llm.client import FallbackClient
        coordinator = CoordinatorAgent(llm_client=FallbackClient())
        _state["coordinator"] = coordinator
        _state["llm_client"] = None
        _state["llm_validation"] = None
        _state["start_time"] = "2026-02-11T00:00:00"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "services" in body

    @pytest.mark.asyncio
    async def test_health_has_service_details(self, client):
        resp = await client.get("/api/health")
        services = resp.json()["services"]
        assert "llm" in services
        assert "vector_db" in services
        assert "api" in services


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_returns_answer(self, client):
        resp = await client.post("/api/chat", json={"query": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "session_id" in body
        assert len(body["answer"]) > 0

    @pytest.mark.asyncio
    async def test_chat_empty_query_rejected(self, client):
        resp = await client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_with_visualization_flag(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "Show trends", "include_visualization": True},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, client):
        resp = await client.post("/api/search", json={"query": "literacy rate"})
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_empty_rejected(self, client):
        resp = await client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class TestCompareEndpoint:
    @pytest.mark.asyncio
    async def test_compare_two_entities(self, client):
        resp = await client.post(
            "/api/compare",
            json={"entities": ["Kerala", "Bihar"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "comparison" in body
        assert "entities" in body

    @pytest.mark.asyncio
    async def test_compare_with_metric(self, client):
        resp = await client.post(
            "/api/compare",
            json={"entities": ["Delhi", "Mumbai"], "metric": "GER"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Newsletter months
# ---------------------------------------------------------------------------

class TestNewsletterEndpoint:
    @pytest.mark.asyncio
    async def test_months_returns_list(self, client):
        resp = await client.get("/api/newsletter/months")
        assert resp.status_code == 200
        body = resp.json()
        assert "months" in body
        assert len(body["months"]) > 0
        assert "default" in body


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

class TestInputSanitisation:
    @pytest.mark.asyncio
    async def test_script_tags_stripped(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "<script>alert('xss')</script>What is GER?"},
        )
        assert resp.status_code == 200
        assert "<script>" not in resp.json()["answer"]
