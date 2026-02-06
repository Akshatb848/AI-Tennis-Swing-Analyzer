"""
Helper utilities for the Data Science Agent Platform
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

# Disable pandas 3.x StringDtype at the earliest import point
pd.set_option("future.infer_string", False)

logger = logging.getLogger(__name__)


def _sanitize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas 3.x StringDtype columns to plain object for numpy compat."""
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
            df[col] = df[col].astype(object)
    return df


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load dataframe from various file formats."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(filepath)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif suffix == '.json':
        df = pd.read_json(filepath)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif suffix == '.feather':
        df = pd.read_feather(filepath)
    elif suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return _sanitize_dtypes(df)


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save dataframe to various file formats."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=False, **kwargs)
    elif suffix == '.json':
        df.to_json(filepath, **kwargs)
    elif suffix == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif suffix == '.feather':
        df.to_feather(filepath, **kwargs)
    elif suffix in ['.pkl', '.pickle']:
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Saved dataframe to {filepath}")


def generate_sample_data(sample_type: str = "random", n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample datasets for testing."""
    np.random.seed(42)
    
    if sample_type == "iris":
        try:
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return _sanitize_dtypes(df)
        except ImportError:
            pass

    elif sample_type == "boston" or sample_type == "housing":
        df = pd.DataFrame({
            'rooms': np.random.uniform(3, 10, n_samples),
            'age': np.random.uniform(1, 100, n_samples),
            'distance': np.random.uniform(1, 12, n_samples),
            'tax': np.random.uniform(150, 750, n_samples),
            'crime_rate': np.random.exponential(3, n_samples),
        })
        df['price'] = df['rooms'] * 30000 + df['age'] * (-200) + np.random.normal(0, 20000, n_samples)
        return _sanitize_dtypes(df)

    elif sample_type == "titanic":
        n = min(n_samples, 891)
        df = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 14, n).clip(0, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.05, 0.03, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.09, 0.02]),
            'Fare': np.random.exponential(32, n),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]),
            'Survived': np.random.choice([0, 1], n, p=[0.62, 0.38])
        })
        return _sanitize_dtypes(df)

    elif sample_type == "classification":
        df = pd.DataFrame({f'feature_{i}': np.random.randn(n_samples) for i in range(10)})
        df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        df['target'] = (df['feature_0'] + df['feature_1'] > 0).astype(int)
        return _sanitize_dtypes(df)

    elif sample_type == "regression":
        df = pd.DataFrame({f'feature_{i}': np.random.randn(n_samples) for i in range(10)})
        df['target'] = 3 * df['feature_0'] + 2 * df['feature_1'] - df['feature_2'] + np.random.randn(n_samples) * 0.5
        return _sanitize_dtypes(df)

    # Default: random data
    df = pd.DataFrame({f'feature_{i}': np.random.randn(n_samples) for i in range(10)})
    df['category'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    df['target'] = np.random.choice([0, 1], n_samples)
    return _sanitize_dtypes(df)
