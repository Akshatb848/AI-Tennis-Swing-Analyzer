"""Tests for the RAG system and ChatHandler."""

import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from backend.rag.rag_system import RAGSystem
from backend.api.chat_handler import ChatHandler


@pytest.fixture
def rag():
    r = RAGSystem()
    r.load_and_index()
    return r


@pytest.fixture
def handler(rag):
    return ChatHandler(rag_system=rag)


# ---------------------------------------------------------------------------
# RAG System
# ---------------------------------------------------------------------------


class TestRAGSystem:
    def test_load_and_index(self, rag):
        assert rag._indexed is True
        assert len(rag.documents) > 0

    def test_get_all_months(self, rag):
        months = rag.get_all_months()
        assert len(months) == 10
        assert months[0] == "April 2025"
        assert months[-1] == "January 2026"

    def test_get_month_data(self, rag):
        data = rag.get_month_data("January 2026")
        assert data is not None
        assert data["theme"] == "National Conclave Planning & Technical Developments"
        assert data["apaar_ids_total"] == 156737923

    def test_get_month_data_not_found(self, rag):
        assert rag.get_month_data("February 2030") is None

    def test_search_apaar(self, rag):
        results = rag.search("APAAR IDs generated", top_k=3)
        assert len(results) > 0
        assert any("apaar" in r["text"].lower() for r in results)

    def test_search_attendance(self, rag):
        results = rag.search("attendance integration Ladakh", top_k=3)
        assert len(results) > 0
        assert any("ladakh" in r["text"].lower() for r in results)

    def test_search_empty(self, rag):
        results = rag.search("xyznonexistent123")
        assert len(results) == 0

    def test_get_apaar_trend(self, rag):
        trend = rag.get_apaar_trend()
        assert len(trend) > 0
        # Should be sorted chronologically
        for i in range(1, len(trend)):
            assert trend[i]["total"] > trend[i - 1]["total"]

    def test_get_sections_for_month(self, rag):
        sections = rag.get_sections_for_month("January 2026")
        assert "correspondence" in sections
        assert "meetings" in sections
        assert len(sections["correspondence"]) > 0


# ---------------------------------------------------------------------------
# ChatHandler
# ---------------------------------------------------------------------------


class TestChatHandler:
    def test_apaar_intent(self, handler):
        result = handler.handle("How many APAAR IDs?")
        assert result["intent"] == "apaar"
        assert result["confidence"] == "high"
        assert len(result["key_points"]) > 0

    def test_summary_intent(self, handler):
        result = handler.handle("Summary for January 2026")
        assert result["intent"] == "summary"
        assert result["confidence"] == "high"
        assert "January 2026" in result["summary"]

    def test_vsk_intent(self, handler):
        result = handler.handle("VSK operations status")
        assert result["intent"] == "vsk"
        assert result["data_backed"] is True

    def test_dashboard_intent(self, handler):
        result = handler.handle("NIPUN dashboard progress")
        assert result["intent"] == "dashboard"

    def test_state_query(self, handler):
        result = handler.handle("Bihar education updates")
        assert result["intent"] == "state_query"
        assert len(result["key_points"]) > 0

    def test_policy_intent(self, handler):
        result = handler.handle("UDISE+ data quality")
        assert result["intent"] == "policy"

    def test_help_intent(self, handler):
        result = handler.handle("What can you help me with?")
        assert result["intent"] == "help"
        assert result["confidence"] == "high"

    def test_meeting_intent(self, handler):
        result = handler.handle("Capacity Building Workshop details")
        assert result["intent"] == "meeting"

    def test_correspondence_intent(self, handler):
        result = handler.handle("Correspondence and letter updates")
        assert result["intent"] == "correspondence"

    def test_general_fallback(self, handler):
        result = handler.handle("random query about something")
        assert "intent" in result
        assert "summary" in result

    def test_response_structure(self, handler):
        result = handler.handle("APAAR IDs trend")
        required_keys = [
            "intent", "summary", "key_points",
            "sources", "confidence", "data_backed",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_month_extraction(self, handler):
        result = handler.handle("What happened in August 2025?")
        assert result["confidence"] == "high"
        assert "August 2025" in result["summary"]
