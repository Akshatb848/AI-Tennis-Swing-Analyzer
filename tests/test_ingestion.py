import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from agents.ingestion_agent import IngestAgent


@pytest.fixture
def agent():
    return IngestAgent()


def _make_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


class TestIngestCSV:
    """Tests for basic CSV ingestion and profiling."""

    def test_ingest_csv(self, agent):
        """Ingest a simple CSV and verify profile row/col counts, dtypes, and quality_score."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": ["x", "y", "x", "y", "x"],
        })
        path = _make_csv(df)
        try:
            result = agent.execute(file_path=path, file_type="csv")
            assert result.success is True
            profile = result.data["profile"]
            assert profile["stats"]["row_count"] == 5
            assert profile["stats"]["column_count"] == 3
            assert "a" in profile["stats"]["dtypes"]
            assert isinstance(profile["quality_score"], float)
            assert 0.0 <= profile["quality_score"] <= 100.0
        finally:
            os.unlink(path)

    def test_ingest_with_target_detection(self, agent):
        """Create CSV with a column named 'target' and verify it is detected."""
        df = pd.DataFrame({
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
            "target": np.random.choice([0, 1], size=20),
        })
        path = _make_csv(df)
        try:
            result = agent.execute(file_path=path, file_type="csv")
            assert result.success is True
            profile = result.data["profile"]
            assert profile["target_column"] == "target"
        finally:
            os.unlink(path)

    def test_problem_type_detection(self, agent):
        """Binary classification is detected when target has exactly 2 unique values."""
        df = pd.DataFrame({
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "target": [0, 1] * 25,
        })
        path = _make_csv(df)
        try:
            result = agent.execute(file_path=path, file_type="csv")
            assert result.success is True
            profile = result.data["profile"]
            assert profile["target_column"] == "target"
            assert profile["problem_type"] == "binary_classification"
        finally:
            os.unlink(path)

    def test_ingest_invalid_file(self, agent):
        """Ingesting a non-existent file returns a failed result with errors."""
        result = agent.execute(file_path="/tmp/nonexistent_file_xyz.csv", file_type="csv")
        assert result.success is False
        assert len(result.errors) > 0

    def test_ingest_has_enhanced_profile(self, agent):
        """Profile includes id_columns, datetime_candidates, and leakage_columns."""
        df = pd.DataFrame({
            "id": range(30),
            "feature": np.random.randn(30),
            "target": np.random.choice([0, 1], 30),
        })
        path = _make_csv(df)
        try:
            result = agent.execute(file_path=path, file_type="csv")
            assert result.success is True
            profile = result.data["profile"]
            assert "id_columns" in profile
            assert "datetime_candidates" in profile
            assert "leakage_columns" in profile
        finally:
            os.unlink(path)
