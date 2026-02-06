"""Tests for prompt templates."""

from llm.prompts import PromptTemplates


class TestPromptTemplates:
    def test_coordinator_system_includes_agents(self):
        prompt = PromptTemplates.coordinator_system("- EDAAgent: does EDA\n- ModelTrainer: trains models")
        assert "EDAAgent" in prompt
        assert "ModelTrainer" in prompt
        assert "expert" in prompt.lower()

    def test_intent_analysis_includes_message(self):
        prompt = PromptTemplates.intent_analysis(
            "train some models", has_dataset=True, has_target=True, dataset_summary="100 x 5"
        )
        assert "train some models" in prompt
        assert "Dataset loaded: True" in prompt

    def test_interpret_results_truncates_long_data(self):
        big_data = {"key": "x" * 5000}
        prompt = PromptTemplates.interpret_results(
            "EDAAgent", "full_eda", big_data, "user question"
        )
        assert "truncated" in prompt

    def test_plan_workflow_includes_target(self):
        prompt = PromptTemplates.plan_workflow("100 x 5", "price", "predict house prices")
        assert "price" in prompt
        assert "predict house prices" in prompt

    def test_plan_workflow_unsupervised(self):
        prompt = PromptTemplates.plan_workflow("100 x 5", None, "explore data")
        assert "Not specified" in prompt

    def test_dataset_summary(self):
        info = {
            "shape": {"rows": 150, "columns": 5},
            "dtypes": {"a": "int64", "b": "float64", "c": "object"},
            "missing_values": {"a": 0, "b": 3, "c": 1},
        }
        summary = PromptTemplates.dataset_summary(info)
        assert "150" in summary
        assert "5" in summary
        assert "2 numeric" in summary
        assert "1 categorical" in summary
        assert "4 total missing" in summary
