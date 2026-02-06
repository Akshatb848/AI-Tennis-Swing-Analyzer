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


@pytest.fixture
def sample_customer_support_df():
    """Realistic customer support dataset with ID, text, and categorical columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Ticket ID": range(1000, 1000 + n),
        "Customer Name": [f"Customer_{i}" for i in range(n)],
        "Customer Email": [f"user{i}@example.com" for i in range(n)],
        "Customer Age": np.random.randint(18, 75, n),
        "Customer Gender": np.random.choice(["Male", "Female", "Other"], n),
        "Product Purchased": np.random.choice(["Laptop", "Phone", "Tablet", "Watch", "Camera"], n),
        "Ticket Type": np.random.choice(["Bug", "Feature Request", "Question"], n),
        "Ticket Subject": [f"Issue about feature {i % 50}" for i in range(n)],
        "Ticket Description": [f"Long description text about the problem that the customer is experiencing, detail #{i}" for i in range(n)],
        "Ticket Status": np.random.choice(["Open", "Closed", "Pending"], n),
        "Ticket Priority": np.random.choice(["Low", "Medium", "High", "Critical"], n),
        "Customer Satisfaction Rating": np.random.choice([1, 2, 3, 4, 5, np.nan], n),
    })
    return df


@pytest.fixture
def sample_categorical_target_df():
    """Dataset with a purely categorical (string) target."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.uniform(20000, 150000, n),
        "score": np.random.randn(n),
        "category": np.random.choice(["A", "B", "C"], n),
        "status": np.random.choice(["Active", "Inactive", "Pending"], n),
    })
    return df


@pytest.fixture
def sample_stringdtype_df():
    """Dataset with pandas 3.x StringDtype columns to test numpy compatibility."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.uniform(20000, 150000, n),
        "score": np.random.randn(n),
        "category": pd.array(np.random.choice(["A", "B", "C"], n), dtype="string"),
        "product": pd.array(np.random.choice(["Laptop", "Phone", "Tablet", "Watch", "Camera"], n), dtype="string"),
        "status": pd.array(np.random.choice(["Active", "Inactive", "Pending"], n), dtype="string"),
        "target": np.random.choice([0, 1], n),
    })
    return df
