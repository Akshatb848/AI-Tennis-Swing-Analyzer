"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df():
    """Simple classification dataset."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "feature_c": np.random.choice(["x", "y", "z"], n),
        "target": np.random.choice([0, 1], n),
    })
    return df


@pytest.fixture
def sample_regression_df():
    """Simple regression dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.uniform(0, 10, n),
    })
    df["target"] = 3 * df["x1"] + 2 * df["x2"] - df["x3"] + np.random.randn(n) * 0.5
    return df
