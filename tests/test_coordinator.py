"""Tests for the LLM-powered CoordinatorAgent."""

import pytest
import asyncio

from agents.coordinator_agent import CoordinatorAgent, Workflow, WorkflowStep
from agents.data_cleaner_agent import DataCleanerAgent
from agents.eda_agent import EDAAgent
from agents.model_trainer_agent import ModelTrainerAgent
from llm.client import FallbackClient


@pytest.fixture
def coordinator():
    """Coordinator with fallback client and a few agents registered."""
    c = CoordinatorAgent(llm_client=FallbackClient())
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


class TestUIHelpers:
    def test_welcome_message(self, coordinator):
        msg = coordinator.get_welcome_message()
        assert "Data Science Agent Platform" in msg

    def test_help_message(self, coordinator):
        msg = coordinator.get_help_message()
        assert "Available Commands" in msg
