"""Tests for ModelingMLAgent (replaces the old ModelingAgent tests).

Optuna tuning is disabled in most tests to keep them fast and deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from agents import modeling_ml_agent
from agents.modeling_ml_agent import ModelingMLAgent


@pytest.fixture
def agent():
    return ModelingMLAgent()


def _regression_data(n=100):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(3.0 * X["f1"] + 2.0 * X["f2"] + rng.normal(size=n) * 0.5, name="y")
    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _classification_data(n=120):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="y")
    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _expected_names(base):
    names = set(base)
    if modeling_ml_agent._HAS_XGB:
        names.add("XGBoost")
    if modeling_ml_agent._HAS_LGB:
        names.add("LightGBM")
    return names


class TestModelingMLAgent:

    def test_regression_training(self, agent):
        X_train, y_train, X_test, y_test = _regression_data()
        result = agent.execute(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               problem_type="regression", use_optuna=False)
        assert result.success is True
        data = result.data
        assert data["champion_model"] is not None
        assert data["champion_name"] in data["all_models"]
        # Score for regression is test-set R^2; the signal here is nearly linear.
        assert data["champion_score"] > 0.9
        assert len(data["champion_model"].predict(np.asarray(X_test))) == len(X_test)

    def test_classification_training(self, agent):
        X_train, y_train, X_test, y_test = _classification_data()
        result = agent.execute(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               problem_type="binary_classification", use_optuna=False)
        assert result.success is True
        score = result.data["champion_score"]  # weighted F1
        assert 0.0 <= score <= 1.0
        assert score > 0.8
        assert "cv_score" in result.data["leaderboard"][0]

    def test_leaderboard_contains_all_candidates_sorted(self, agent):
        X_train, y_train, X_test, y_test = _regression_data()
        result = agent.execute(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               problem_type="regression", use_optuna=False)
        assert result.success is True
        leaderboard = result.data["leaderboard"]
        names = {e["name"] for e in leaderboard}
        assert names == _expected_names({"LinearRegression", "RandomForest", "GradientBoosting"})
        scores = [e["score"] for e in leaderboard]
        assert scores == sorted(scores, reverse=True)
        assert result.data["champion_name"] == leaderboard[0]["name"]
        for entry in leaderboard:
            assert {"score", "params", "train_time"} <= set(entry)

    def test_classification_candidates(self, agent):
        names = {n for n, _ in agent._get_candidates(is_classification=True)}
        assert names == _expected_names({"LogisticRegression", "RandomForest", "GradientBoosting"})

    def test_build_model_unknown_name_raises(self, agent):
        with pytest.raises(ValueError):
            agent._build_model("NotAModel", {}, is_classification=False)

    @pytest.mark.skipif(not modeling_ml_agent._HAS_OPTUNA, reason="optuna not installed")
    def test_optuna_adds_tuned_entries(self, agent, monkeypatch):
        """With Optuna enabled, the top-2 models get a '_tuned' leaderboard entry."""
        original = modeling_ml_agent.optuna.create_study

        class _FastStudy:
            def __init__(self, study):
                self._study = study

            def optimize(self, objective, n_trials=30, timeout=None, show_progress_bar=False):
                self._study.optimize(objective, n_trials=2)

            def __getattr__(self, item):
                return getattr(self._study, item)

        monkeypatch.setattr(modeling_ml_agent.optuna, "create_study",
                            lambda **kw: _FastStudy(original(**kw)))
        X_train, y_train, X_test, y_test = _regression_data()
        result = agent.execute(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               problem_type="regression", use_optuna=True)
        assert result.success is True
        tuned = [e["name"] for e in result.data["leaderboard"] if e["name"].endswith("_tuned")]
        assert 1 <= len(tuned) <= 2
