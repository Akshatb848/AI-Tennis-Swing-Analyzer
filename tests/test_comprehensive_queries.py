"""Comprehensive query testing for all types of user queries.

Tests that every category of query returns robust, data-backed responses
with proper intent classification, confidence indicators, and sources.
Ensures the system works reliably for conference demonstrations.
"""

import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from backend.rag.rag_system import RAGSystem
from backend.api.chat_handler import ChatHandler


@pytest.fixture
def handler():
    rag = RAGSystem()
    rag.load_and_index()
    return ChatHandler(rag_system=rag)


# ---------------------------------------------------------------------------
# APAAR Queries
# ---------------------------------------------------------------------------


class TestAPAARQueries:
    def test_apaar_count(self, handler):
        r = handler.handle("How many APAAR IDs have been generated?")
        assert r["intent"] == "apaar"
        assert r["confidence"] == "high"
        assert any("156" in kp or "15,67" in kp for kp in r["key_points"])

    def test_apaar_trend(self, handler):
        r = handler.handle("APAAR IDs trend over time")
        assert r["confidence"] == "high"
        assert len(r["key_points"]) >= 5  # multiple months of data

    def test_apaar_latest(self, handler):
        r = handler.handle("Latest APAAR ID count")
        assert r["data_backed"] is True
        assert len(r["key_points"]) > 0


# ---------------------------------------------------------------------------
# Monthly Summary Queries
# ---------------------------------------------------------------------------


class TestSummaryQueries:
    def test_january_2026_summary(self, handler):
        r = handler.handle("Summary for January 2026")
        assert r["intent"] == "summary"
        assert "January 2026" in r["summary"]
        assert r["confidence"] == "high"
        assert len(r["key_points"]) > 0

    def test_august_2025_summary(self, handler):
        r = handler.handle("What happened in August 2025?")
        assert "August 2025" in r["summary"]
        assert any("workshop" in kp.lower() or "capacity" in kp.lower() for kp in r["key_points"])

    def test_october_2025_overview(self, handler):
        r = handler.handle("Overview of October 2025 newsletter")
        assert r["confidence"] == "high"
        assert any("bihar" in kp.lower() for kp in r["key_points"])

    def test_december_2025_highlights(self, handler):
        r = handler.handle("Summary December 2025 newsletter")
        assert r["confidence"] == "high"
        assert len(r["key_points"]) > 0

    def test_default_latest_month(self, handler):
        r = handler.handle("What is the latest newsletter summary?")
        assert r["confidence"] == "high"


# ---------------------------------------------------------------------------
# VSK Queries
# ---------------------------------------------------------------------------


class TestVSKQueries:
    def test_vsk_count(self, handler):
        r = handler.handle("How many VSKs are operational?")
        assert r["intent"] == "vsk"
        # May be medium or high depending on search results
        assert r["confidence"] in ("high", "medium")

    def test_rvsk_portal(self, handler):
        r = handler.handle("RVSK portal status and UAT")
        assert r["intent"] == "vsk"
        assert len(r["key_points"]) > 0

    def test_west_bengal_vsk(self, handler):
        r = handler.handle("West Bengal VSK operationalization")
        assert len(r["key_points"]) > 0

    def test_bihar_vsk(self, handler):
        r = handler.handle("Bihar Vidya Samiksha Kendra")
        assert r["data_backed"] is True


# ---------------------------------------------------------------------------
# Dashboard Queries
# ---------------------------------------------------------------------------


class TestDashboardQueries:
    def test_nipun_dashboard(self, handler):
        r = handler.handle("NIPUN dashboard status")
        assert r["intent"] == "dashboard"
        assert len(r["key_points"]) > 0

    def test_assessment_dashboard(self, handler):
        r = handler.handle("Assessment Dashboard development progress")
        assert r["intent"] == "dashboard"

    def test_pm_shri_dashboard(self, handler):
        r = handler.handle("PM SHRI dashboard updates")
        assert r["intent"] == "dashboard"
        assert r["data_backed"] is True


# ---------------------------------------------------------------------------
# State-Specific Queries
# ---------------------------------------------------------------------------


class TestStateQueries:
    def test_ladakh_integration(self, handler):
        r = handler.handle("Ladakh attendance integration")
        assert r["intent"] == "state_query"
        assert any("ladakh" in kp.lower() for kp in r["key_points"])

    def test_manipur_support(self, handler):
        r = handler.handle("Manipur technical support")
        assert r["intent"] == "state_query"
        assert len(r["key_points"]) > 0

    def test_meghalaya_data(self, handler):
        r = handler.handle("Meghalaya education data integration")
        assert r["intent"] == "state_query"

    def test_chhattisgarh_integration(self, handler):
        r = handler.handle("Chhattisgarh RVSK integration")
        assert len(r["key_points"]) > 0

    def test_andaman_starter_pack(self, handler):
        r = handler.handle("Andaman and Nicobar Islands starter pack")
        assert r["data_backed"] is True


# ---------------------------------------------------------------------------
# Policy Queries
# ---------------------------------------------------------------------------


class TestPolicyQueries:
    def test_udise_data_quality(self, handler):
        r = handler.handle("UDISE+ data quality and name mismatches")
        assert r["intent"] == "policy"
        assert r["data_backed"] is True

    def test_ncte_integration(self, handler):
        r = handler.handle("NCTE VSK integration with RVSK")
        assert len(r["key_points"]) > 0

    def test_nios_launch(self, handler):
        r = handler.handle("NIOS Vidya Samiksha Kendra launch")
        assert r["data_backed"] is True

    def test_bharatnet(self, handler):
        r = handler.handle("BharatNet connectivity progress")
        assert r["intent"] == "policy"


# ---------------------------------------------------------------------------
# Meeting / Workshop Queries
# ---------------------------------------------------------------------------


class TestMeetingQueries:
    def test_capacity_workshop(self, handler):
        r = handler.handle("Capacity Building Workshop 2.0")
        assert r["intent"] == "meeting"
        assert any("workshop" in kp.lower() or "capacity" in kp.lower() for kp in r["key_points"])

    def test_uidai_meeting(self, handler):
        r = handler.handle("UIDAI meeting about Aadhaar for APAAR")
        assert len(r["key_points"]) > 0

    def test_korean_delegation(self, handler):
        r = handler.handle("Korean delegation visit to RVSK")
        assert r["intent"] == "meeting"

    def test_national_conclave(self, handler):
        r = handler.handle("National Conclave February 2026 Telangana")
        assert r["data_backed"] is True


# ---------------------------------------------------------------------------
# Correspondence Queries
# ---------------------------------------------------------------------------


class TestCorrespondenceQueries:
    def test_letters_to_states(self, handler):
        r = handler.handle("Correspondence and letter to NCERT")
        assert r["intent"] == "correspondence"
        assert len(r["key_points"]) > 0

    def test_pm_shri_request(self, handler):
        r = handler.handle("Letter to PM SHRI Bureau requesting data")
        assert r["data_backed"] is True


# ---------------------------------------------------------------------------
# Help & Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_help_query(self, handler):
        r = handler.handle("What can you help me with?")
        assert r["intent"] == "help"
        assert len(r["key_points"]) > 0

    def test_empty_like_query(self, handler):
        r = handler.handle("xyz random query")
        assert "intent" in r
        assert "summary" in r

    def test_response_always_has_required_fields(self, handler):
        queries = [
            "APAAR IDs", "VSK status", "January 2026 summary",
            "NIPUN dashboard", "Bihar education", "UDISE+ data",
            "Capacity workshop", "Letter to NCERT", "help",
        ]
        for q in queries:
            r = handler.handle(q)
            assert "intent" in r, f"Missing 'intent' for query: {q}"
            assert "summary" in r, f"Missing 'summary' for query: {q}"
            assert "key_points" in r, f"Missing 'key_points' for query: {q}"
            assert "sources" in r, f"Missing 'sources' for query: {q}"
            assert "confidence" in r, f"Missing 'confidence' for query: {q}"
            assert "data_backed" in r, f"Missing 'data_backed' for query: {q}"
            assert r["confidence"] in ("high", "medium", "low"), f"Invalid confidence for query: {q}"

    def test_month_context_passed(self, handler):
        r = handler.handle("What is the summary?", current_month="September 2025")
        assert r["confidence"] == "high"
        assert "September 2025" in r["summary"]
