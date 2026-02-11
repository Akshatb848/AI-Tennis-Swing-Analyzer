"""Tests for execution gating: no LLM = no intelligence.

These tests verify that the system fails closed when the LLM is not
validated. Planning, intent analysis, interpretation, and conversational
replies must all refuse to proceed without a real LLM.
"""

import pytest

from agents.coordinator_agent import CoordinatorAgent
from llm.client import FallbackClient, OpenAIClient


# ---------------------------------------------------------------------------
# Intent analysis gating
# ---------------------------------------------------------------------------


class TestIntentAnalysisGating:
    """Coordinator.analyze_user_intent must fail closed without a real LLM."""

    @pytest.mark.asyncio
    async def test_none_client_blocks_intent(self):
        coordinator = CoordinatorAgent(llm_client=None)
        result = await coordinator.analyze_user_intent("analyze data", {})
        assert result["intent"] == "llm_unavailable"
        assert result["blocked"] is True

    @pytest.mark.asyncio
    async def test_fallback_client_blocks_intent(self):
        coordinator = CoordinatorAgent(llm_client=FallbackClient())
        result = await coordinator.analyze_user_intent("train models", {})
        assert result["intent"] == "llm_unavailable"
        assert result["blocked"] is True

    @pytest.mark.asyncio
    async def test_llm_exception_blocks_intent(self, monkeypatch):
        """If the LLM throws during intent analysis, fail closed (no silent fallback)."""
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def exploding_chat_json(*args, **kwargs):
            raise RuntimeError("Connection refused")

        monkeypatch.setattr(client, "chat_json", exploding_chat_json)
        coordinator = CoordinatorAgent(llm_client=client)
        result = await coordinator.analyze_user_intent("do something", {})
        assert result["intent"] == "llm_unavailable"
        assert result["blocked"] is True
        assert "Connection refused" in result["explanation"]


# ---------------------------------------------------------------------------
# Result interpretation gating
# ---------------------------------------------------------------------------


class TestResultInterpretationGating:
    """Coordinator.interpret_results must refuse without a real LLM."""

    @pytest.mark.asyncio
    async def test_none_client_blocks_interpretation(self):
        coordinator = CoordinatorAgent(llm_client=None)
        result = await coordinator.interpret_results("EDAAgent", "full_eda", {"rows": 100})
        assert "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_fallback_client_blocks_interpretation(self):
        coordinator = CoordinatorAgent(llm_client=FallbackClient())
        result = await coordinator.interpret_results("EDAAgent", "full_eda", {"rows": 100})
        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# Conversational reply gating
# ---------------------------------------------------------------------------


class TestConversationalReplyGating:
    """Coordinator.generate_conversational_reply must use fallback without LLM."""

    @pytest.mark.asyncio
    async def test_none_client_uses_fallback_reply(self):
        coordinator = CoordinatorAgent(llm_client=None)
        reply = await coordinator.generate_conversational_reply(
            "hello", context={"has_dataset": False}
        )
        # Should still respond but NOT use LLM intelligence
        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_fallback_client_uses_fallback_reply(self):
        coordinator = CoordinatorAgent(llm_client=FallbackClient())
        reply = await coordinator.generate_conversational_reply(
            "hello", context={"has_dataset": False}
        )
        assert isinstance(reply, str)
        assert len(reply) > 0


# ---------------------------------------------------------------------------
# Workflow planning gating
# ---------------------------------------------------------------------------


class TestWorkflowPlanningGating:
    """Workflow planning falls back to deterministic pipeline without LLM."""

    @pytest.mark.asyncio
    async def test_none_client_uses_default_workflow(self):
        coordinator = CoordinatorAgent(llm_client=None)
        project = coordinator.create_project("Test")
        workflow = await coordinator.plan_workflow(
            project_id=project["id"],
            dataset_info={"target_column": "label", "id": "ds1"},
        )
        # Should get a deterministic workflow, not an LLM-planned one
        assert len(workflow.steps) > 0
        agent_names = [s.agent for s in workflow.steps]
        assert "DataCleanerAgent" in agent_names
