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

    @pytest.mark.asyncio
    async def test_clean_data_reports_missing(self, sample_customer_support_df):
        agent = DataCleanerAgent()
        result = await agent.run({"action": "clean_data", "dataframe": sample_customer_support_df})
        assert result.success
        report = result.data["cleaning_report"]
        # Should report missing values filled (Customer Satisfaction Rating has NaN)
        steps = {s["step"]: s for s in report["steps"]}
        assert "handle_missing" in steps
        assert steps["handle_missing"]["values_filled"] > 0


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

    @pytest.mark.asyncio
    async def test_engineer_skips_id_and_text(self, sample_customer_support_df):
        """ID and text columns should be dropped, not encoded."""
        agent = FeatureEngineerAgent()
        result = await agent.run({
            "action": "engineer_features",
            "dataframe": sample_customer_support_df,
            "target_column": "Customer Satisfaction Rating",
        })
        assert result.success
        engineered_df = result.data["dataframe"]
        report = result.data["feature_report"]
        dropped = [d["column"] for d in report.get("dropped_columns", [])]
        # Should have dropped ID and text columns
        assert "Ticket ID" in dropped or "Ticket ID" not in engineered_df.columns
        assert "Customer Email" in dropped or "Customer Email" not in engineered_df.columns
        assert "Ticket Description" in dropped or "Ticket Description" not in engineered_df.columns


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

    @pytest.mark.asyncio
    async def test_train_with_categorical_target(self, sample_categorical_target_df):
        """Model trainer should auto-encode categorical (string) targets."""
        agent = ModelTrainerAgent()
        result = await agent.run({
            "action": "train_models",
            "dataframe": sample_categorical_target_df,
            "target_column": "status",
            "cv_folds": 3,
        })
        assert result.success
        assert result.data["target_encoded"] is True
        assert result.data["task_type"] == "classification"
        assert len(result.data["results"]) >= 1

    @pytest.mark.asyncio
    async def test_train_no_numeric_features(self):
        """Should return helpful error when no numeric features available."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"] * 10,
            "city": ["NY", "LA", "SF"] * 10,
            "target": [0, 1, 0] * 10,
        })
        agent = ModelTrainerAgent()
        result = await agent.run({
            "action": "train_models",
            "dataframe": df,
            "target_column": "target",
        })
        assert not result.success
        assert "No numeric features" in result.error

    @pytest.mark.asyncio
    async def test_train_drops_id_columns(self, sample_customer_support_df):
        """ID columns should be dropped before training."""
        # Add numeric features so training can proceed
        df = sample_customer_support_df.copy()
        df["feature_numeric"] = np.random.randn(len(df))
        agent = ModelTrainerAgent()
        result = await agent.run({
            "action": "train_models",
            "dataframe": df,
            "target_column": "Customer Gender",
            "cv_folds": 2,
        })
        assert result.success
        assert "Ticket ID" in result.data.get("id_columns_dropped", [])


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

    @pytest.mark.asyncio
    async def test_auto_select_with_categorical_target(self, sample_categorical_target_df):
        """AutoML should auto-encode categorical targets."""
        agent = AutoMLAgent()
        result = await agent.run({
            "action": "auto_select_models",
            "dataframe": sample_categorical_target_df,
            "target_column": "status",
        })
        assert result.success
        assert result.data["target_encoded"] is True


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
        section_types = [s["type"] for s in result.data["components"]]
        assert "kpi_section" in section_types
        assert "data_quality_section" in section_types
        assert "chart_section" in section_types

    @pytest.mark.asyncio
    async def test_dashboard_with_model_results(self, sample_classification_df):
        """Dashboard should include model comparison when model results provided."""
        model_results = {
            "results": {
                "RF": {"metrics": {"accuracy": 0.85, "f1": 0.84}},
                "LR": {"metrics": {"accuracy": 0.78, "f1": 0.77}},
            },
            "best_model": "RF",
            "best_metrics": {"accuracy": 0.85, "f1": 0.84},
            "task_type": "classification",
        }
        agent = DashboardBuilderAgent()
        result = await agent.run({
            "action": "build_dashboard",
            "dataframe": sample_classification_df,
            "model_results": model_results,
            "target_column": "target",
        })
        assert result.success
        section_types = [s["type"] for s in result.data["components"]]
        assert "model_comparison_section" in section_types
        assert "target_analysis_section" in section_types


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

    @pytest.mark.asyncio
    async def test_full_pipeline_customer_support(self, sample_customer_support_df):
        """Full pipeline on a realistic dataset with IDs, text, and categorical target."""
        df = sample_customer_support_df
        target = "Ticket Priority"

        # Clean
        cleaner = DataCleanerAgent()
        result = await cleaner.run({"action": "clean_data", "dataframe": df})
        assert result.success
        df = result.data["dataframe"]

        # Feature Engineering
        fe = FeatureEngineerAgent()
        result = await fe.run({"action": "engineer_features", "dataframe": df, "target_column": target})
        assert result.success
        df = result.data["dataframe"]

        # Model Training (categorical target)
        trainer = ModelTrainerAgent()
        result = await trainer.run({
            "action": "train_models",
            "dataframe": df,
            "target_column": target,
            "cv_folds": 2,
        })
        assert result.success
        assert result.data["target_encoded"] is True
        assert len(result.data["results"]) >= 1
