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
        import sys as _sys
        import pathlib as _pathlib

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
        _state["sessions"] = {}

    if "rag" not in _state:
        from backend.rag.rag_system import RAGSystem

        rag = RAGSystem()
        rag.load_and_index()
        _state["rag"] = rag

        from backend.api.chat_handler import ChatHandler

        _state["chat_handler"] = ChatHandler(rag_system=rag)

    # Reset rate limiter for tests
    from backend.main import _rate_store
    _rate_store.clear()

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

    @pytest.mark.asyncio
    async def test_health_vector_db_shows_indexed(self, client):
        resp = await client.get("/api/health")
        vdb = resp.json()["services"]["vector_db"]
        assert vdb["connected"] is True
        assert "docs indexed" in vdb["detail"]


# ---------------------------------------------------------------------------
# LLM Status
# ---------------------------------------------------------------------------


class TestLLMStatusEndpoint:
    @pytest.mark.asyncio
    async def test_llm_status_returns_200(self, client):
        resp = await client.get("/api/llm/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "configured" in body


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_returns_structured_response(self, client):
        resp = await client.post("/api/chat", json={"query": "APAAR IDs"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "intent" in body
        assert "key_points" in body
        assert "confidence" in body
        assert "session_id" in body

    @pytest.mark.asyncio
    async def test_chat_empty_query_rejected(self, client):
        resp = await client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_summary_intent(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "Summary for January 2026"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] == "high"
        assert len(body["key_points"]) > 0

    @pytest.mark.asyncio
    async def test_chat_apaar_intent(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "How many APAAR IDs have been generated?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "apaar"
        assert body["data_backed"] is True

    @pytest.mark.asyncio
    async def test_chat_session_tracking(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "Bihar VSK", "session_id": "test123"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test123"

    @pytest.mark.asyncio
    async def test_chat_with_visualization_flag(self, client):
        resp = await client.post(
            "/api/chat",
            json={
                "query": "APAAR IDs trend",
                "include_visualization": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("visualization") is not None


# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------


class TestChatSessionEndpoint:
    @pytest.mark.asyncio
    async def test_session_not_found(self, client):
        resp = await client.get("/api/chat/session/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_session_retrieval(self, client):
        # Create a session first
        await client.post(
            "/api/chat",
            json={"query": "Test session", "session_id": "sess_test"},
        )
        resp = await client.get("/api/chat/session/sess_test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess_test"
        assert len(body["messages"]) > 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, client):
        resp = await client.post(
            "/api/search", json={"query": "attendance integration"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_results_have_month(self, client):
        resp = await client.post(
            "/api/search", json={"query": "APAAR", "top_k": 3}
        )
        body = resp.json()
        for r in body["results"]:
            assert "month" in r

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
    async def test_compare_states_list(self, client):
        resp = await client.get("/api/compare/states")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_months_list(self, client):
        resp = await client.get("/api/compare/months")
        assert resp.status_code == 200
        body = resp.json()
        assert "months" in body
        assert len(body["months"]) > 0


# ---------------------------------------------------------------------------
# Newsletter
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

    @pytest.mark.asyncio
    async def test_newsletter_detail(self, client):
        resp = await client.get("/api/newsletter/January-2026")
        assert resp.status_code == 200
        body = resp.json()
        assert body["month"] == "January 2026"
        assert "highlights" in body
        assert "theme" in body

    @pytest.mark.asyncio
    async def test_newsletter_not_found(self, client):
        resp = await client.get("/api/newsletter/March-2030")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_available_kpis(self, client):
        resp = await client.get("/api/stats/kpis")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TestAdminEndpoint:
    @pytest.mark.asyncio
    async def test_reindex(self, client):
        resp = await client.post("/api/admin/reindex")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["documents_indexed"] > 0


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------


class TestInputSanitisation:
    @pytest.mark.asyncio
    async def test_script_tags_stripped(self, client):
        resp = await client.post(
            "/api/chat",
            json={"query": "<script>alert('xss')</script>What is APAAR?"},
        )
        assert resp.status_code == 200
        assert "<script>" not in resp.json()["answer"]
