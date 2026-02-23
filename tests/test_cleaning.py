import numpy as np
import pandas as pd
import pytest

from agents.cleaning_agent import CleaningAgent


@pytest.fixture
def agent():
    return CleaningAgent()


def _make_df_with_missing():
    np.random.seed(42)
    return pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


class TestBasicCleaning:
    """Tests for the CleaningAgent data preparation pipeline."""

    def test_basic_cleaning(self, agent):
        """DataFrame with missing values produces output with no missing values."""
        df = _make_df_with_missing()
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        X_train = result.data["X_train"]
        X_test = result.data["X_test"]
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_train_test_split(self, agent):
        """Output contains X_train, X_test, y_train, y_test with correct shapes."""
        df = _make_df_with_missing()
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        data = result.data
        for key in ("X_train", "X_test", "y_train", "y_test"):
            assert key in data, f"Missing key: {key}"
        assert len(data["X_train"]) + len(data["X_test"]) == len(df)
        assert len(data["y_train"]) + len(data["y_test"]) == len(df)

    def test_high_missing_column_dropped(self, agent):
        """A column with >50% missing values is dropped during cleaning."""
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "good_col": np.arange(n, dtype=float),
            "bad_col": [np.nan] * 12 + list(range(8)),
            "target": [0, 1] * 10,
        })
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        assert "bad_col" in result.data["dropped_columns"]
        assert "bad_col" not in result.data["feature_names"]

    def test_categorical_encoding(self, agent):
        """Categorical columns are encoded to numeric types after cleaning."""
        np.random.seed(42)
        df = pd.DataFrame({
            "num_feat": np.random.randn(30),
            "cat_feat": np.random.choice(["a", "b", "c"], size=30),
            "target": np.random.choice([0, 1], size=30),
        })
        result = agent.execute(df=df, target_col="target", problem_type="binary_classification")
        assert result.success is True
        X_train = result.data["X_train"]
        for col in X_train.columns:
            assert pd.api.types.is_numeric_dtype(X_train[col]), (
                f"Column '{col}' is not numeric after encoding"
            )
