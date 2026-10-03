"""End-to-end test of the 5-step Orchestrator pipeline (no LLM, no Optuna)."""

import os

import numpy as np
import pandas as pd
import pytest

from agents.mlops_deployment_agent import MLOpsDeploymentAgent
from agents.orchestrator import PIPELINE_STEPS, Orchestrator


@pytest.fixture
def classification_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "income": rng.normal(50_000, 12_000, size=n).round(2),
        "plan": rng.choice(["basic", "plus", "pro"], size=n),
    })
    df["churn"] = ((df["age"] < 35) | (df["plan"] == "basic")).astype(int)
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_full_pipeline_completes(classification_csv, tmp_path):
    orch = Orchestrator()
    orch.agents["mlops"].MODELS_DIR = str(tmp_path / "models")

    out = orch.run_pipeline(classification_csv, target_col="churn", use_optuna=False)

    assert out["pipeline_state"] == {step: "completed" for step in PIPELINE_STEPS}
    modeling = orch.get_step_result("modeling")
    assert modeling.data["champion_score"] > 0.7
    exploration = orch.get_step_result("exploration")
    assert exploration.success is True

    mlops = orch.get_step_result("mlops").data
    assert mlops["deployment_ready"] is True
    assert os.path.exists(mlops["model_path"])
    assert mlops["model_path"].startswith(str(tmp_path))
    assert "def predict" in mlops["inference_script"]


def test_mlops_agent_saves_loadable_model(tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
    agent = MLOpsDeploymentAgent()
    agent.MODELS_DIR = str(tmp_path)
    result = agent.execute(champion_model=model, champion_name="Linear Regression",
                           feature_names=["x"], problem_type="regression",
                           model_metrics={"champion_score": 1.0})
    assert result.success is True
    path = result.data["model_path"]
    assert os.path.basename(path).startswith("Linear_Regression_")
    reloaded = joblib.load(path)
    assert reloaded.predict([[3.0]])[0] == pytest.approx(6.0)
    assert isinstance(result.data["monitoring_config"], dict)
    assert result.data["deployment_recommendations"]
