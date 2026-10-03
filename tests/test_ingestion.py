"""Tests for the ingestion + business-strategy step.

The old ``IngestAgent`` was folded into ``Orchestrator._run_ingest_and_strategy``
(file loading/profiling via ``utils.csv_loader``) followed by ``BusinessStrategyAgent``.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.business_strategy_agent import BusinessStrategyAgent
from agents.orchestrator import Orchestrator


def _write_csv(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _ingest(path, target_col=None, llm_client=None):
    orch = Orchestrator(llm_client=llm_client)
    orch._run_ingest_and_strategy(path, target_col)
    return orch


class TestIngestion:

    def test_ingest_csv_profile(self, tmp_path):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": ["x", "y", "x", "y", "x"],
        })
        orch = _ingest(_write_csv(tmp_path, df))
        result = orch.get_step_result("strategy")
        assert result.success is True
        assert orch.get_pipeline_state()["strategy"] == "completed"
        profile = result.data["profile"]
        assert profile["stats"]["row_count"] == 5
        assert profile["stats"]["column_count"] == 3
        assert "a" in profile["stats"]["dtypes"]
        assert 0.0 <= profile["quality_score"] <= 100.0
        assert result.data["df"].shape == (5, 3)

    def test_target_and_problem_type_detection(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
            "target": [0, 1] * 25,
        })
        orch = _ingest(_write_csv(tmp_path, df))
        profile = orch.get_step_result("strategy").data["profile"]
        assert profile["target_column"] == "target"
        assert profile["problem_type"] == "binary_classification"
        assert orch._context["problem_type"] == "binary_classification"

    def test_explicit_target_overrides_detection(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "price": rng.normal(100, 10, size=60),
            "x": rng.normal(size=60),
            "target": [0, 1] * 30,
        })
        orch = _ingest(_write_csv(tmp_path, df), target_col="price")
        assert orch._context["target_col"] == "price"
        assert orch.get_step_result("strategy").data["profile"]["target_column"] == "price"

    def test_enhanced_profile_fields(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "id": range(30),
            "feature": rng.normal(size=30),
            "target": rng.choice([0, 1], 30),
        })
        profile = _ingest(_write_csv(tmp_path, df)).get_step_result("strategy").data["profile"]
        assert "id" in profile["id_columns"]
        assert "datetime_candidates" in profile
        assert "leakage_columns" in profile

    def test_json_file_is_supported(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})
        path = tmp_path / "data.json"
        df.to_json(path, orient="records")
        orch = _ingest(str(path))
        assert orch.get_step_result("strategy").success is True
        assert orch.get_step_result("strategy").data["profile"]["target_column"] == "label"

    def test_missing_file_fails_and_skips_downstream(self, tmp_path):
        orch = Orchestrator()
        out = orch.run_pipeline(str(tmp_path / "does_not_exist.csv"), use_optuna=False)
        state = out["pipeline_state"]
        assert state["strategy"] == "failed"
        assert orch.get_step_result("strategy").errors
        for step in ("engineering", "exploration", "modeling", "mlops"):
            assert state[step] == "skipped"


def _profile(**overrides):
    profile = {
        "problem_type": "binary_classification",
        "target_column": "churn",
        "quality_score": 85.0,
        "schema": {"age": "numeric", "plan": "categorical", "churn": "numeric"},
        "stats": {"row_count": 300, "column_count": 3, "missing_pct": {"age": 40.0}},
        "id_columns": ["customer_id"],
        "leakage_columns": [],
        "datetime_candidates": [],
        "warnings": [],
    }
    profile.update(overrides)
    return profile


class TestBusinessStrategyAgent:

    def test_rule_based_objective_and_constraints(self):
        result = BusinessStrategyAgent().execute(dataset_profile=_profile())
        assert result.success is True
        assert result.metadata["source"] == "rule_based"
        objective = result.data["objective"]
        assert objective["problem_type"] == "binary_classification"
        assert objective["target"] == "churn"
        assert objective["kpi"] == "accuracy"
        constraints = " ".join(objective["constraints"])
        assert "Small dataset" in constraints
        assert ">30% missing" in constraints
        assert "customer_id" in constraints
        assert any("stratified" in r.lower() for r in result.data["recommendations"])

    @pytest.mark.parametrize("problem_type, kpi", [
        ("regression", "rmse"),
        ("clustering", "silhouette_score"),
        ("something_unknown", "n/a"),
    ])
    def test_kpi_mapping(self, problem_type, kpi):
        result = BusinessStrategyAgent().execute(dataset_profile=_profile(problem_type=problem_type))
        assert result.data["objective"]["kpi"] == kpi

    def test_user_prompt_is_recorded(self):
        result = BusinessStrategyAgent().execute(dataset_profile=_profile(),
                                                 user_prompt="optimise recall")
        assert any("optimise recall" in r for r in result.data["recommendations"])

    def test_disconnected_llm_uses_rules(self):
        llm = MagicMock()
        llm.is_connected.return_value = False
        result = BusinessStrategyAgent().execute(dataset_profile=_profile(), llm_client=llm)
        assert result.metadata["source"] == "rule_based"
        llm.chat.assert_not_called()

    def test_connected_llm_response_becomes_recommendations(self):
        llm = MagicMock()
        llm.is_connected.return_value = True
        llm.chat.return_value = (
            "- Use gradient boosting with class weights\n"
            "- Monitor recall on the churn class closely\n"
            "ok\n"
        )
        result = BusinessStrategyAgent().execute(dataset_profile=_profile(), llm_client=llm)
        assert result.success is True
        assert result.metadata["source"] == "llm"
        assert result.data["recommendations"] == [
            "Use gradient boosting with class weights",
            "Monitor recall on the churn class closely",
        ]
        assert result.data["objective"]["kpi"] == "accuracy"
        llm.chat.assert_called_once()

    def test_llm_exception_falls_back_to_rules(self):
        llm = MagicMock()
        llm.is_connected.return_value = True
        llm.chat.side_effect = RuntimeError("network down")
        result = BusinessStrategyAgent().execute(dataset_profile=_profile(), llm_client=llm)
        assert result.success is True
        assert result.metadata["source"] == "rule_based"
