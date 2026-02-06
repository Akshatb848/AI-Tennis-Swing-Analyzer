"""Integration tests — run agents end-to-end with sample data."""

import pytest
import numpy as np
import pandas as pd

from agents.data_cleaner_agent import DataCleanerAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.automl_agent import AutoMLAgent
from agents.data_visualizer_agent import DataVisualizerAgent
from agents.dashboard_builder_agent import DashboardBuilderAgent


class TestDataCleanerIntegration:
    @pytest.mark.asyncio
    async def test_clean_data(self, sample_classification_df):
        agent = DataCleanerAgent()
        result = await agent.run({"action": "clean_data", "dataframe": sample_classification_df})
        assert result.success
        assert "dataframe" in result.data
        assert isinstance(result.data["dataframe"], pd.DataFrame)
        assert "cleaning_report" in result.data


class TestEDAIntegration:
    @pytest.mark.asyncio
    async def test_full_eda(self, sample_classification_df):
        agent = EDAAgent()
        result = await agent.run({
            "action": "full_eda",
            "dataframe": sample_classification_df,
            "target_column": "target",
        })
        assert result.success
        assert "dataset_info" in result.data
        assert "statistical_profile" in result.data
        assert "insights" in result.data

    @pytest.mark.asyncio
    async def test_eda_without_target(self, sample_classification_df):
        agent = EDAAgent()
        result = await agent.run({
            "action": "full_eda",
            "dataframe": sample_classification_df,
        })
        assert result.success


class TestFeatureEngineerIntegration:
    @pytest.mark.asyncio
    async def test_engineer_features(self, sample_classification_df):
        agent = FeatureEngineerAgent()
        result = await agent.run({
            "action": "engineer_features",
            "dataframe": sample_classification_df,
            "target_column": "target",
        })
        assert result.success
        assert "dataframe" in result.data
        # Should have more columns than original after engineering
        assert result.data["dataframe"].shape[1] >= sample_classification_df.shape[1] - 1  # -1 for categorical encoding


class TestModelTrainerIntegration:
    @pytest.mark.asyncio
    async def test_train_classification(self, sample_classification_df):
        # Prepare: drop non-numeric for simplicity
        df = sample_classification_df.drop(columns=["feature_c"])
        agent = ModelTrainerAgent()
        result = await agent.run({
            "action": "train_models",
            "dataframe": df,
            "target_column": "target",
            "cv_folds": 3,
        })
        assert result.success
        assert "best_model" in result.data
        assert "results" in result.data
        assert result.data["task_type"] == "classification"

    @pytest.mark.asyncio
    async def test_train_regression(self, sample_regression_df):
        agent = ModelTrainerAgent()
        result = await agent.run({
            "action": "train_models",
            "dataframe": sample_regression_df,
            "target_column": "target",
            "cv_folds": 3,
        })
        assert result.success
        assert result.data["task_type"] == "regression"
        assert "r2" in result.data["best_metrics"]


class TestAutoMLIntegration:
    @pytest.mark.asyncio
    async def test_auto_select(self, sample_classification_df):
        df = sample_classification_df.drop(columns=["feature_c"])
        agent = AutoMLAgent()
        result = await agent.run({
            "action": "auto_select_models",
            "dataframe": df,
            "target_column": "target",
        })
        assert result.success
        assert "best_model" in result.data


class TestDataVisualizerIntegration:
    @pytest.mark.asyncio
    async def test_generate_visualizations(self, sample_classification_df):
        agent = DataVisualizerAgent()
        result = await agent.run({
            "action": "generate_visualizations",
            "dataframe": sample_classification_df,
        })
        assert result.success
        assert "charts" in result.data
        assert len(result.data["charts"]) > 0


class TestDashboardBuilderIntegration:
    @pytest.mark.asyncio
    async def test_build_dashboard(self, sample_classification_df):
        agent = DashboardBuilderAgent()
        result = await agent.run({
            "action": "build_dashboard",
            "dataframe": sample_classification_df,
        })
        assert result.success
        assert "components" in result.data


class TestPipelineEndToEnd:
    """Test the full pipeline: clean -> EDA -> FE -> train."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_classification_df):
        df = sample_classification_df
        target = "target"

        # Step 1: Clean
        cleaner = DataCleanerAgent()
        result = await cleaner.run({"action": "clean_data", "dataframe": df})
        assert result.success
        df = result.data["dataframe"]

        # Step 2: EDA
        eda = EDAAgent()
        result = await eda.run({"action": "full_eda", "dataframe": df, "target_column": target})
        assert result.success

        # Step 3: Feature Engineering
        fe = FeatureEngineerAgent()
        result = await fe.run({"action": "engineer_features", "dataframe": df, "target_column": target})
        assert result.success
        df = result.data["dataframe"]

        # Step 4: Model Training
        trainer = ModelTrainerAgent()
        result = await trainer.run({
            "action": "train_models",
            "dataframe": df,
            "target_column": target,
            "cv_folds": 3,
        })
        assert result.success
        assert "best_model" in result.data
        assert len(result.data["results"]) >= 1
