"""Tests for DataEngineeringAgent (replaces the old CleaningAgent tests)."""

import numpy as np
import pandas as pd
import pytest

from agents.data_engineering_agent import DataEngineeringAgent


@pytest.fixture
def agent():
    return DataEngineeringAgent()


def _make_df_with_missing():
    return pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


class TestDataEngineeringAgent:
    """Cleaning, encoding and splitting behaviour of the engineering pipeline."""

    def test_imputation_removes_missing_values(self, agent):
        result = agent.execute(df=_make_df_with_missing(), target_col="target",
                               problem_type="binary_classification")
        assert result.success is True
        assert result.data["X_train"].isnull().sum().sum() == 0
        assert result.data["X_test"].isnull().sum().sum() == 0
        imputed = result.data["engineering_report"]["imputation"]["columns_imputed"]
        assert set(imputed) == {"feat1", "feat2"}

    def test_train_test_split(self, agent):
        df = _make_df_with_missing()
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        data = result.data
        for key in ("X_train", "X_test", "y_train", "y_test"):
            assert key in data, f"Missing key: {key}"
        assert len(data["X_train"]) + len(data["X_test"]) == len(df)
        assert len(data["y_train"]) + len(data["y_test"]) == len(df)
        assert len(data["X_test"]) == 2  # test_size=0.2
        assert "target" not in data["X_train"].columns

    def test_high_missing_column_dropped(self, agent):
        """Columns above the default 60% missing threshold are dropped."""
        n = 20
        df = pd.DataFrame({
            "good_col": np.arange(n, dtype=float),
            "bad_col": [np.nan] * 14 + list(range(6)),  # 70% missing
            "target": [0, 1] * 10,
        })
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        assert "bad_col" in result.data["engineering_report"]["dropped_high_missing"]
        assert "bad_col" not in result.data["feature_names"]

    def test_id_and_leakage_columns_dropped(self, agent):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "row_id": range(40),
            "leak": rng.normal(size=40),
            "x": rng.normal(size=40),
            "target": [0, 1] * 20,
        })
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification",
                               id_columns=["row_id"], leakage_columns=["leak"])
        assert result.success is True
        dropped = result.data["engineering_report"]["dropped_id_leakage"]
        assert set(dropped) == {"row_id", "leak"}
        assert not {"row_id", "leak"} & set(result.data["feature_names"])

    def test_categorical_encoding(self, agent):
        """Low-cardinality categoricals are one-hot encoded to numeric columns."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "num_feat": rng.normal(size=30),
            "cat_feat": rng.choice(["a", "b", "c"], size=30),
            "target": rng.choice([0, 1], size=30),
        })
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        X_train = result.data["X_train"]
        for col in X_train.columns:
            assert pd.api.types.is_numeric_dtype(X_train[col]), f"{col} is not numeric"
        encoded = result.data["engineering_report"]["encoding"]["encoded"]
        assert encoded["cat_feat"]["method"] == "onehot"

    def test_string_target_is_label_encoded(self, agent):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x1": rng.normal(size=30),
            "x2": rng.normal(size=30),
            "target": ["yes", "no", "maybe"] * 10,
        })
        result = agent.execute(df=df, target_col="target",
                               problem_type="multiclass_classification")
        assert result.success is True
        report = result.data["engineering_report"]
        assert report["target_label_encoded"] is True
        assert report["target_classes"] == ["maybe", "no", "yes"]
        assert pd.api.types.is_integer_dtype(result.data["y_train"])

    def test_skewed_positive_feature_gets_box_cox(self, agent):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "skewed": rng.exponential(scale=5.0, size=200) + 1.0,
            "x": rng.normal(size=200),
            "target": rng.normal(size=200),
        })
        result = agent.execute(df=df, target_col="target", problem_type="regression")
        assert result.success is True
        corrections = result.data["engineering_report"]["skewness_correction"]["corrections"]
        assert corrections["skewed"]["method"] == "box_cox"
        assert abs(corrections["skewed"]["new_skewness"]) < abs(
            corrections["skewed"]["original_skewness"]
        )

    def test_features_are_scaled(self, agent):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "a": rng.normal(100, 20, size=100),
            "b": rng.normal(-5, 3, size=100),
            "target": rng.normal(size=100),
        })
        result = agent.execute(df=df, target_col="target", problem_type="regression")
        assert result.success is True
        assert result.data["engineering_report"]["scaling"]["method"] in ("standard", "robust")
        X_all = pd.concat([result.data["X_train"], result.data["X_test"]])
        assert abs(X_all["a"].mean()) < 1e-6

    def test_without_target_returns_unsplit_features(self, agent):
        df = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) * 2})
        result = agent.execute(df=df, target_col=None, problem_type="clustering")
        assert result.success is True
        assert len(result.data["X_train"]) == 20
        assert len(result.data["X_test"]) == 0

    def test_invalid_input_returns_failed_result(self, agent):
        result = agent.execute(df=None, target_col="target")
        assert result.success is False
        assert result.errors
