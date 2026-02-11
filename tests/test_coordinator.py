"""Tests for the LLM-powered CoordinatorAgent.

Uses a MockLLMClient (not FallbackClient) so the coordinator treats it
as a real LLM.  FallbackClient is correctly blocked by the fail-closed
architecture — those paths are tested in test_llm_execution_guard.py.
"""

import pytest
import asyncio
import json
from typing import Any, Dict, List, Optional

from agents.coordinator_agent import CoordinatorAgent, Workflow, WorkflowStep
from agents.data_cleaner_agent import DataCleanerAgent
from agents.eda_agent import EDAAgent
from agents.model_trainer_agent import ModelTrainerAgent
from llm.client import LLMClient, FallbackClient, _extract_json


class MockLLMClient(LLMClient):
    """A mock LLM client that uses rule-based logic but is NOT a FallbackClient.

    This allows coordinator tests to exercise the real LLM code path
    (intent analysis, interpretation) without hitting a real API.
    The coordinator's fail-closed guard only blocks None and FallbackClient,
    so this mock passes those checks.

    For chat_json (intent analysis), it delegates to FallbackClient's logic.
    For chat (interpretation/conversation), it raises so the coordinator
    falls back to its template-based _fallback_interpret.
    """

    def __init__(self):
        self._fallback = FallbackClient()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        # Raise to trigger coordinator's _fallback_interpret for interpretation,
        # and _fallback_conversational_reply for conversations
        raise RuntimeError("Mock LLM: no real inference available")

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        # The coordinator sends a formatted prompt via PromptTemplates.intent_analysis.
        # Extract the original user message from 'User message: "..."' and run
        # the fallback's rule-based intent on the raw message instead.
        import re
        last_content = messages[-1]["content"] if messages else ""
        m = re.search(r'User message:\s*"(.+?)"', last_content)
        raw_msg = m.group(1) if m else last_content
        return self._fallback._rule_based_intent(raw_msg)


@pytest.fixture
def coordinator():
    """Coordinator with a mock LLM client and a few agents registered."""
    c = CoordinatorAgent(llm_client=MockLLMClient())
    c.register_agent(DataCleanerAgent())
    c.register_agent(EDAAgent())
    c.register_agent(ModelTrainerAgent())
    return c


class TestConversationalMemory:
    def test_add_to_memory(self, coordinator):
        coordinator.add_to_memory("user", "Hello")
        coordinator.add_to_memory("assistant", "Hi there!")
        assert len(coordinator.conversation_history) == 2
        assert coordinator.conversation_history[0]["role"] == "user"
        assert coordinator.conversation_history[1]["content"] == "Hi there!"

    def test_memory_bounded_at_50(self, coordinator):
        for i in range(60):
            coordinator.add_to_memory("user", f"Message {i}")
        assert len(coordinator.conversation_history) == 50
        # Most recent message should be the last one added
        assert coordinator.conversation_history[-1]["content"] == "Message 59"

    def test_get_llm_messages_format(self, coordinator):
        coordinator.add_to_memory("user", "test")
        msgs = coordinator.get_llm_messages()
        assert msgs == [{"role": "user", "content": "test"}]

    def test_record_analysis(self, coordinator):
        coordinator.record_analysis("EDAAgent", "full_eda", {"rows": 100})
        assert len(coordinator.analysis_history) == 1
        assert coordinator.analysis_history[0]["agent"] == "EDAAgent"


class TestIntentAnalysis:
    @pytest.mark.asyncio
    async def test_eda_intent(self, coordinator):
        result = await coordinator.analyze_user_intent(
            "analyze my data",
            context={"has_dataset": True, "has_target": True, "dataset_summary": "100 rows x 5 cols"},
        )
        assert result.get("intent") == "run_agent"
        assert result.get("agent") == "EDAAgent"

    @pytest.mark.asyncio
    async def test_clean_intent(self, coordinator):
        result = await coordinator.analyze_user_intent("clean the data", context={})
        assert result["intent"] == "run_agent"
        assert result["agent"] == "DataCleanerAgent"

    @pytest.mark.asyncio
    async def test_train_intent(self, coordinator):
        result = await coordinator.analyze_user_intent("train models", context={})
        assert result["intent"] == "run_agent"
        assert result["agent"] == "ModelTrainerAgent"

    @pytest.mark.asyncio
    async def test_pipeline_intent(self, coordinator):
        result = await coordinator.analyze_user_intent("run full analysis", context={})
        assert result["intent"] == "run_pipeline"

    @pytest.mark.asyncio
    async def test_help_intent(self, coordinator):
        result = await coordinator.analyze_user_intent("help", context={})
        assert result["intent"] == "help"

    @pytest.mark.asyncio
    async def test_raw_message_preserved(self, coordinator):
        result = await coordinator.analyze_user_intent("hello world", context={})
        assert result["raw_message"] == "hello world"


class TestResultInterpretation:
    @pytest.mark.asyncio
    async def test_eda_interpretation(self, coordinator):
        result_data = {
            "dataset_info": {
                "shape": {"rows": 100, "columns": 5},
                "missing_values": {"a": 3, "b": 0},
                "duplicates": 2,
            },
            "insights": ["High skewness in column X"],
            "recommendations": ["Consider log transform"],
        }
        interpretation = await coordinator.interpret_results(
            "EDAAgent", "full_eda", result_data
        )
        assert "100" in interpretation
        assert "5" in interpretation

    @pytest.mark.asyncio
    async def test_model_interpretation(self, coordinator):
        result_data = {
            "best_model": "RandomForest",
            "best_metrics": {"accuracy": 0.95, "f1": 0.93, "cv_mean": 0.92, "cv_std": 0.02},
            "task_type": "classification",
            "results": {"RandomForest": {}, "LogisticRegression": {}},
        }
        interpretation = await coordinator.interpret_results(
            "ModelTrainerAgent", "train_models", result_data
        )
        assert "RandomForest" in interpretation
        assert "0.95" in interpretation

    @pytest.mark.asyncio
    async def test_cleaner_interpretation(self, coordinator):
        result_data = {
            "cleaning_report": {
                "original_shape": (100, 5),
                "final_shape": (98, 5),
                "steps": [{"step": "remove_duplicates", "removed": 2}],
            }
        }
        interpretation = await coordinator.interpret_results(
            "DataCleanerAgent", "clean_data", result_data
        )
        assert "100" in interpretation
        assert "98" in interpretation


class TestProjectManagement:
    def test_create_project(self, coordinator):
        project = coordinator.create_project("Test Project", "A test")
        assert project["name"] == "Test Project"
        assert project["id"] in coordinator.projects

    def test_add_dataset_to_project(self, coordinator):
        project = coordinator.create_project("Test")
        result = coordinator.add_dataset_to_project(project["id"], {"name": "data.csv"})
        assert result is True
        assert len(coordinator.projects[project["id"]]["datasets"]) == 1

    def test_add_dataset_to_nonexistent_project(self, coordinator):
        result = coordinator.add_dataset_to_project("fake_id", {"name": "data.csv"})
        assert result is False


class TestWorkflowPlanning:
    @pytest.mark.asyncio
    async def test_default_workflow_with_target(self, coordinator):
        project = coordinator.create_project("Test")
        workflow = await coordinator.plan_workflow(
            project_id=project["id"],
            dataset_info={"target_column": "label", "id": "ds1"},
        )
        assert isinstance(workflow, Workflow)
        agent_names = [s.agent for s in workflow.steps]
        assert "DataCleanerAgent" in agent_names
        assert "EDAAgent" in agent_names
        assert "ModelTrainerAgent" in agent_names

    @pytest.mark.asyncio
    async def test_default_workflow_without_target(self, coordinator):
        project = coordinator.create_project("Test")
        workflow = await coordinator.plan_workflow(
            project_id=project["id"],
            dataset_info={"target_column": None, "id": "ds1"},
        )
        agent_names = [s.agent for s in workflow.steps]
        assert "DataCleanerAgent" in agent_names
        assert "EDAAgent" in agent_names
        # No modeling without a target
        assert "ModelTrainerAgent" not in agent_names


class TestConversationalReply:
    @pytest.mark.asyncio
    async def test_greeting_reply(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "Hello!", context={"has_dataset": False}
        )
        assert "hello" in reply.lower() or "ai data scientist" in reply.lower()

    @pytest.mark.asyncio
    async def test_thanks_reply(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "Thanks a lot!", context={}
        )
        assert "welcome" in reply.lower()

    @pytest.mark.asyncio
    async def test_capabilities_redirects_to_help(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "What can you do?", context={}
        )
        assert "Available Commands" in reply

    @pytest.mark.asyncio
    async def test_no_dataset_suggests_upload(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "I need some analysis", context={"has_dataset": False}
        )
        assert "upload" in reply.lower()

    @pytest.mark.asyncio
    async def test_dataset_loaded_suggests_actions(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "What should I do next?",
            context={
                "has_dataset": True,
                "has_target": True,
                "dataset_summary": "100 rows x 5 columns",
                "completed_analyses": {},
            },
        )
        assert "Analyze" in reply or "Clean" in reply

    @pytest.mark.asyncio
    async def test_with_completed_analyses_suggests_next(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "What now?",
            context={
                "has_dataset": True,
                "has_target": True,
                "dataset_summary": "100 rows x 5 columns",
                "completed_analyses": {"cleaning": {}, "eda": {}},
            },
        )
        assert "cleaning" in reply.lower() or "eda" in reply.lower()
        assert "next" in reply.lower() or "suggest" in reply.lower() or "feature" in reply.lower()

    @pytest.mark.asyncio
    async def test_all_done_congratulates(self, coordinator):
        all_done = {
            "cleaning": {}, "eda": {}, "feature_engineering": {},
            "modeling": {}, "automl": {}, "visualization": {}, "dashboard": {},
        }
        reply = await coordinator.generate_conversational_reply(
            "What now?",
            context={
                "has_dataset": True,
                "has_target": True,
                "dataset_summary": "100 rows x 5 columns",
                "completed_analyses": all_done,
            },
        )
        assert "completed" in reply.lower() or "thorough" in reply.lower()

    @pytest.mark.asyncio
    async def test_ds_question_reply(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "What is overfitting?", context={"has_dataset": False}
        )
        assert "data science" in reply.lower() or "upload" in reply.lower()

    @pytest.mark.asyncio
    async def test_no_target_suggests_setting_one(self, coordinator):
        reply = await coordinator.generate_conversational_reply(
            "What next?",
            context={
                "has_dataset": True,
                "has_target": False,
                "dataset_summary": "100 rows x 5 columns",
                "completed_analyses": {},
            },
        )
        assert "target" in reply.lower()


class TestUIHelpers:
    def test_welcome_message(self, coordinator):
        msg = coordinator.get_welcome_message()
        assert "AI Data Science Platform" in msg

    def test_help_message(self, coordinator):
        msg = coordinator.get_help_message()
        assert "Available Commands" in msg
