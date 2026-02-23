import numpy as np
import pandas as pd
import pytest

from agents.modeling_agent import ModelingAgent


@pytest.fixture
def agent():
    return ModelingAgent()


def _regression_data(n=100):
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    y = pd.Series(3.0 * X["f1"] + 2.0 * X["f2"] + np.random.randn(n) * 0.5, name="y")
    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _classification_data(n=120):
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="y")
    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestModelingAgent:
    """Tests for training, metrics, and model comparison."""

    def test_regression_training(self, agent):
        """Train regression models and verify champion_model is returned."""
        X_train, y_train, X_test, y_test = _regression_data()
        result = agent.execute(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type="regression",
        )
        assert result.success is True
        assert result.data["champion_model"] is not None
        assert "champion_name" in result.data
        metrics = result.data["champion_metrics"]
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_classification_training(self, agent):
        """Train classification models and verify accuracy/f1 metrics exist."""
        X_train, y_train, X_test, y_test = _classification_data()
        result = agent.execute(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type="binary_classification",
        )
        assert result.success is True
        assert result.data["champion_model"] is not None
        metrics = result.data["champion_metrics"]
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_weighted"] <= 1.0

    def test_model_comparison(self, agent):
        """metrics_comparison table has entries for all trained models."""
        X_train, y_train, X_test, y_test = _regression_data()
        result = agent.execute(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type="regression",
        )
        assert result.success is True
        comparison = result.data["metrics_comparison"]
        expected_models = {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}
        assert expected_models == set(comparison.keys())
        for model_name, metrics in comparison.items():
            assert "rmse" in metrics, f"{model_name} missing rmse"
            assert "r2" in metrics, f"{model_name} missing r2"
