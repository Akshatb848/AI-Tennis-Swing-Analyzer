import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ======================================================
# COLUMN PROFILE
# ======================================================

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    unique_ratio: float
    is_numeric: bool
    is_datetime: bool
    variance: Optional[float]
    monotonic: bool


class DatasetProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self) -> Dict[str, ColumnProfile]:
        profiles = {}

        for col in self.df.columns:
            series = self.df[col]

            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_ratio=series.isnull().mean(),
                unique_ratio=series.nunique() / max(len(series), 1),
                is_numeric=pd.api.types.is_numeric_dtype(series),
                is_datetime=pd.api.types.is_datetime64_any_dtype(series),
                variance=series.var() if pd.api.types.is_numeric_dtype(series) else None,
                monotonic=series.is_monotonic_increasing if pd.api.types.is_numeric_dtype(series) else False,
            )

        return profiles


# ======================================================
# SEMANTIC CLASSIFIER
# ======================================================

class SemanticModel:
    def __init__(self):
        self.time_columns: List[str] = []
        self.measures: List[str] = []
        self.dimensions: List[str] = []
        self.identifiers: List[str] = []
        self.rates: List[str] = []


class SemanticClassifier:
    def __init__(self, profiles: Dict[str, ColumnProfile]):
        self.profiles = profiles
        self.model = SemanticModel()

    def classify(self) -> SemanticModel:
        for col, p in self.profiles.items():

            name = col.lower()

            if p.is_datetime or "date" in name or "time" in name:
                self.model.time_columns.append(col)

            elif p.unique_ratio > 0.9 and not p.is_numeric:
                self.model.identifiers.append(col)

            elif "%" in name or "margin" in name or "rate" in name:
                self.model.rates.append(col)

            elif p.is_numeric and p.variance and p.variance > 0:
                self.model.measures.append(col)

            else:
                self.model.dimensions.append(col)

        return self.model


# ======================================================
# METRIC INTELLIGENCE
# ======================================================

class MetricIntelligenceEngine:
    def __init__(self, df: pd.DataFrame, semantic: SemanticModel):
        self.df = df
        self.semantic = semantic

    def discover_kpis(self) -> List[str]:
        kpis = []

        for col in self.semantic.measures:
            series = self.df[col]

            if (
                series.var() > 0
                and series.nunique() > 10
                and series.isnull().mean() < 0.5
            ):
                kpis.append(col)

        return kpis


# ======================================================
# FORECAST ELIGIBILITY ENGINE
# ======================================================

class ForecastEligibilityEngine:
    def __init__(self, df, time_col, metric):
        self.df = df
        self.time_col = time_col
        self.metric = metric

    def check(self) -> Tuple[bool, str]:
        ts = self.df[[self.time_col, self.metric]].dropna()

        if len(ts) < 15:
            return False, "Insufficient historical data"

        mean = ts[self.metric].mean()
        std = ts[self.metric].std()

        if mean == 0:
            return False, "Zero mean series"

        if std / mean > 2.5:
            return False, "Highly volatile series"

        return True, "Forecast eligible"


# ======================================================
# SEMANTIC CATALOG
# ======================================================

@dataclass
class MetricDefinition:
    name: str
    column: str
    aggregation: str = "sum"
    format: str = "number"
    description: str = ""
    unit: str = ""


@dataclass
class SemanticCatalog:
    metrics: List[MetricDefinition] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    time_column: Optional[str] = None
    hierarchies: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SemanticCatalog":
        metrics_payload = payload.get("metrics", [])
        metrics = [
            MetricDefinition(
                name=item.get("name", item.get("column", "")),
                column=item.get("column", ""),
                aggregation=item.get("aggregation", "sum"),
                format=item.get("format", "number"),
                description=item.get("description", ""),
                unit=item.get("unit", "")
            )
            for item in metrics_payload
        ]
        return cls(
            metrics=metrics,
            dimensions=payload.get("dimensions", []),
            time_column=payload.get("time_column"),
            hierarchies=payload.get("hierarchies", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": [
                {
                    "name": m.name,
                    "column": m.column,
                    "aggregation": m.aggregation,
                    "format": m.format,
                    "description": m.description,
                    "unit": m.unit
                }
                for m in self.metrics
            ],
            "dimensions": self.dimensions,
            "time_column": self.time_column,
            "hierarchies": self.hierarchies
        }

    def validate(self, df: pd.DataFrame) -> List[str]:
        issues = []
        columns = set(df.columns)
        for metric in self.metrics:
            if not metric.column:
                issues.append(f"Metric '{metric.name}' is missing a column mapping.")
            elif metric.column not in columns:
                issues.append(f"Metric '{metric.name}' column '{metric.column}' not found.")
        for dim in self.dimensions:
            if dim not in columns:
                issues.append(f"Dimension '{dim}' not found in dataset.")
        if self.time_column and self.time_column not in columns:
            issues.append(f"Time column '{self.time_column}' not found in dataset.")
        return issues

    def metric_columns(self) -> List[str]:
        return [m.column for m in self.metrics if m.column]
