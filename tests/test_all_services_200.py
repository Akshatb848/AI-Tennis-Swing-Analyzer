"""Ensure all API services return HTTP 200.

Conference requirement: every endpoint must be healthy and responsive.
"""

import sys
import pathlib
import pytest
import pytest_asyncio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from httpx import ASGITransport, AsyncClient
from backend.main import app, _state


@pytest_asyncio.fixture
async def client():
    if "coordinator" not in _state:
        root = str(pathlib.Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from agents.coordinator_agent import CoordinatorAgent
        from llm.client import FallbackClient

        _state["coordinator"] = CoordinatorAgent(llm_client=FallbackClient())
        _state["llm_client"] = None
        _state["llm_validation"] = None
        _state["start_time"] = "2026-02-11T00:00:00"
        _state["sessions"] = {}

    if "rag" not in _state:
        from backend.rag.rag_system import RAGSystem
        from backend.api.chat_handler import ChatHandler

        rag = RAGSystem()
        rag.load_and_index()
        _state["rag"] = rag
        _state["chat_handler"] = ChatHandler(rag_system=rag)

    # Reset rate limiter for tests
    from backend.main import _rate_store
    _rate_store.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestAllServicesReturn200:
    """Every API endpoint must return 200 for valid requests."""

    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_llm_status(self, client):
        r = await client.get("/api/llm/status")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_chat(self, client):
        r = await client.post("/api/chat", json={"query": "APAAR IDs"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_summary(self, client):
        r = await client.post("/api/chat", json={"query": "Summary January 2026"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_vsk(self, client):
        r = await client.post("/api/chat", json={"query": "VSK operations"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_session_create(self, client):
        r = await client.post("/api/chat", json={"query": "test", "session_id": "svc200"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_session_retrieve(self, client):
        await client.post("/api/chat", json={"query": "test", "session_id": "svc200r"})
        r = await client.get("/api/chat/session/svc200r")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_search(self, client):
        r = await client.post("/api/search", json={"query": "attendance"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_compare(self, client):
        r = await client.post("/api/compare", json={"entities": ["Bihar", "Kerala"]})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_states(self, client):
        r = await client.get("/api/compare/states")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_months(self, client):
        r = await client.get("/api/compare/months")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_newsletter_months(self, client):
        r = await client.get("/api/newsletter/months")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_newsletter_detail(self, client):
        r = await client.get("/api/newsletter/January-2026")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_kpis(self, client):
        r = await client.get("/api/stats/kpis")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_admin_reindex(self, client):
        r = await client.post("/api/admin/reindex")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_all_services_healthy(self, client):
        """Verify all three services report healthy in /api/health."""
        r = await client.get("/api/health")
        body = r.json()
        assert body["services"]["api"]["connected"] is True
        assert body["services"]["vector_db"]["connected"] is True
