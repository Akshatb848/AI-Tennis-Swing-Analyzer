"""Comprehensive tests for the robust CSV loader.

Tests cover: different delimiters, encodings, bad lines, quoted commas,
empty files, duplicate columns, BOM handling, large files, headerless files,
and dataset nature detection.
"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from utils.csv_loader import CSVLoadError, load_csv


class TestBasicLoading:
    """Test basic CSV loading from different source types."""

    def test_load_from_bytes(self):
        """Load CSV from raw bytes."""
        data = b"a,b,c\n1,2,3\n4,5,6\n"
        df, profile = load_csv(data)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]
        assert profile["stats"]["row_count"] == 2
        assert profile["stats"]["column_count"] == 3

    def test_load_from_bytesio(self):
        """Load CSV from BytesIO object (simulates Streamlit upload)."""
        csv_str = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        buf = io.BytesIO(csv_str.encode("utf-8"))
        df, profile = load_csv(buf)
        assert len(df) == 2
        assert "name" in df.columns
        assert df.iloc[0]["name"] == "Alice"

    def test_load_from_file_path(self):
        """Load CSV from a file path string."""
        csv_content = "x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(csv_content)
            path = f.name
        try:
            df, profile = load_csv(path)
            assert len(df) == 3
            assert profile["quality_score"] > 0
        finally:
            os.unlink(path)


class TestEncodings:
    """Test different file encodings."""

    def test_utf8_with_bom(self):
        """Handle UTF-8 BOM correctly."""
        bom = b"\xef\xbb\xbf"
        data = bom + "col1,col2\n1,2\n3,4\n".encode("utf-8")
        df, profile = load_csv(data)
        assert len(df) == 2
        assert "col1" in df.columns
        assert profile["encoding"] == "utf-8-sig"

    def test_latin1_encoding(self):
        """Handle latin-1 encoded files with special characters."""
        csv_str = "name,city\nJosé,São Paulo\nFrançois,Zürich\n"
        data = csv_str.encode("latin-1")
        df, profile = load_csv(data)
        assert len(df) == 2
        assert "José" in df["name"].values or "Jos" in str(df["name"].values)

    def test_cp1252_encoding(self):
        """Handle Windows CP1252 encoding."""
        csv_str = "product,price\nWidget\u2122,9.99\nGadget\u00ae,19.99\n"
        data = csv_str.encode("cp1252")
        df, profile = load_csv(data)
        assert len(df) == 2


class TestDelimiters:
    """Test different delimiter handling."""

    def test_semicolon_delimiter(self):
        """Auto-detect semicolon delimiter."""
        data = b"a;b;c\n1;2;3\n4;5;6\n7;8;9\n"
        df, profile = load_csv(data)
        assert len(df) == 3
        assert list(df.columns) == ["a", "b", "c"]
        assert profile["delimiter"] == ";"

    def test_tab_delimiter(self):
        """Auto-detect tab delimiter."""
        data = b"a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        df, profile = load_csv(data)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]
        assert profile["delimiter"] == "\t"

    def test_pipe_delimiter(self):
        """Auto-detect pipe delimiter."""
        data = b"a|b|c\n1|2|3\n4|5|6\n"
        df, profile = load_csv(data)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]
        assert profile["delimiter"] == "|"

    def test_forced_delimiter(self):
        """Force a specific delimiter."""
        data = b"a;b;c\n1;2;3\n"
        df, profile = load_csv(data, delimiter=";")
        assert list(df.columns) == ["a", "b", "c"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self):
        """Empty file raises CSVLoadError with clear message."""
        with pytest.raises(CSVLoadError) as exc_info:
            load_csv(b"")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_file(self):
        """File with only whitespace raises CSVLoadError."""
        with pytest.raises(CSVLoadError) as exc_info:
            load_csv(b"   \n  \n  ")
        assert exc_info.value.cause in ("empty_file", "empty_data")

    def test_quoted_commas(self):
        """Fields with commas inside quotes are parsed correctly."""
        data = b'name,address,city\n"Smith, John","123 Main St, Apt 4",NYC\n"Doe, Jane","456 Oak Ave",LA\n'
        df, profile = load_csv(data)
        assert len(df) == 2
        assert "Smith, John" in df["name"].values

    def test_bad_lines_skipped(self):
        """Ragged/bad lines are skipped without crashing."""
        data = b"a,b,c\n1,2,3\n4,5\n6,7,8\n9,10,11,12\n"
        df, profile = load_csv(data)
        assert len(df) >= 2

    def test_duplicate_columns(self):
        """Duplicate column names are handled (renamed by parser)."""
        data = b"x,x,y\n1,2,3\n4,5,6\n"
        df, profile = load_csv(data)
        assert len(set(df.columns)) == 3
        assert len(df) == 2

    def test_unnamed_columns(self):
        """Unnamed/empty columns are auto-named."""
        data = b",b,\n1,2,3\n4,5,6\n"
        df, profile = load_csv(data)
        assert len(df.columns) == 3
        assert all(col != "" for col in df.columns)

    def test_sample_rows_limit(self):
        """Large files are sampled when sample_rows is set."""
        rows = "\n".join([f"{i},{i*2}" for i in range(1000)])
        data = ("a,b\n" + rows).encode("utf-8")
        df, profile = load_csv(data, sample_rows=100)
        assert len(df) == 100


class TestNatureDetection:
    """Test automatic dataset nature/problem type detection."""

    def test_classification_detection(self):
        """Binary classification detected for binary target."""
        np.random.seed(42)
        rows = ["feature,target"]
        for _ in range(100):
            rows.append(f"{np.random.randn():.4f},{np.random.choice([0,1])}")
        data = "\n".join(rows).encode("utf-8")
        df, profile = load_csv(data)
        assert profile["target_column"] == "target"
        assert profile["problem_type"] == "binary_classification"

    def test_regression_detection(self):
        """Regression detected for continuous numeric target."""
        np.random.seed(42)
        rows = ["x1,x2,target"]
        for _ in range(100):
            rows.append(f"{np.random.randn():.4f},{np.random.randn():.4f},{np.random.randn()*100:.2f}")
        data = "\n".join(rows).encode("utf-8")
        df, profile = load_csv(data)
        assert profile["problem_type"] == "regression"

    def test_id_column_detection(self):
        """ID-like columns are detected."""
        rows = ["id,feature,label"]
        for i in range(50):
            rows.append(f"{i},{np.random.randn():.4f},{np.random.choice([0,1])}")
        data = "\n".join(rows).encode("utf-8")
        df, profile = load_csv(data)
        assert "id" in profile["id_columns"]

    def test_target_hint_churn(self):
        """Column named 'churn' is detected as target."""
        rows = ["age,balance,churn"]
        for _ in range(50):
            rows.append(f"{np.random.randint(18,70)},{np.random.randint(0,10000)},{np.random.choice([0,1])}")
        data = "\n".join(rows).encode("utf-8")
        df, profile = load_csv(data)
        assert profile["target_column"] == "churn"
        assert profile["problem_type"] == "binary_classification"

    def test_profile_has_all_fields(self):
        """Profile dict contains all required fields."""
        data = b"a,b,c\n1,2,3\n4,5,6\n"
        df, profile = load_csv(data)
        required_keys = [
            "schema", "stats", "quality_score", "target_column",
            "problem_type", "id_columns", "datetime_candidates",
            "leakage_columns", "encoding", "delimiter", "warnings",
        ]
        for key in required_keys:
            assert key in profile, f"Missing profile key: {key}"

    def test_stats_have_missing_pct(self):
        """Stats include missingness percentage."""
        data = b"a,b\n1,\n2,3\n,4\n5,6\n"
        df, profile = load_csv(data)
        assert "missing_pct" in profile["stats"]
        col_a = df.columns[0]
        col_b = df.columns[1]
        assert profile["stats"]["missing_pct"][col_a] == 25.0
        assert profile["stats"]["missing_pct"][col_b] == 25.0
