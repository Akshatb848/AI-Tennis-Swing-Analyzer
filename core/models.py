"""Pydantic schemas and data models for the AI Data Science Platform.

Defines all shared data structures used across agents, services, and the dashboard.
Uses Pydantic v2 BaseModel for serializable schemas and dataclasses for fields
that hold non-serializable objects (e.g. numpy arrays).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    """Supported ML problem types."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    EDA_ONLY = "eda_only"


class StepStatus(str, Enum):
    """Status of an individual pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProjectObjective(BaseModel):
    """High-level objective for a data-science project."""

    problem_type: str = Field(default="eda_only", description="ML problem type")
    target_column: Optional[str] = Field(default=None, description="Target variable column name")
    kpi_metric: str = Field(default="accuracy", description="Primary evaluation metric")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints such as max training time or model size")
    business_context: str = Field(default="", description="Free-text business context for the project")


class DataProfile(BaseModel):
    """Statistical profile of an ingested dataset."""

    shape: tuple[int, int] = Field(default=(0, 0), description="(rows, columns)")
    dtypes: Dict[str, str] = Field(default_factory=dict, description="Column name to pandas dtype mapping")
    missing_pct: Dict[str, float] = Field(default_factory=dict, description="Column name to missing-value percentage")
    cardinality: Dict[str, float] = Field(default_factory=dict, description="Column name to cardinality percentage")
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall data quality score 0-100")
    col_schema: Dict[str, str] = Field(default_factory=dict, alias="schema", description="Column name to inferred semantic type")
    target_column: Optional[str] = Field(default=None, description="Detected or user-specified target column")
    problem_type: str = Field(default="eda_only", description="Detected or user-specified problem type")
    id_columns: List[str] = Field(default_factory=list, description="Columns identified as IDs")
    datetime_candidates: List[str] = Field(default_factory=list, description="Columns that may be datetime")
    leakage_columns: List[str] = Field(default_factory=list, description="Columns suspected of target leakage")
    warnings: List[str] = Field(default_factory=list, description="Data quality warnings")

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}


@dataclass
class EngineeredDataset:
    """Holds train/test splits and feature engineering metadata.

    Uses a dataclass because numpy arrays are not natively serializable
    by Pydantic without custom validators.
    """

    X_train: np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    X_test: np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    y_train: np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    y_test: np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    feature_names: List[str] = dc_field(default_factory=list)
    encoding_map: Dict[str, Any] = dc_field(default_factory=dict)
    scaling_method: Optional[str] = None
    dropped_columns: List[str] = dc_field(default_factory=list)
    engineering_report: Dict[str, Any] = dc_field(default_factory=dict)


class VisualizationSpec(BaseModel):
    """Specification for a single chart or visualization."""

    chart_type: str = Field(description="Type of chart (bar, scatter, histogram, heatmap, box, line, etc.)")
    x_col: Optional[str] = Field(default=None, description="Column for X axis")
    y_col: Optional[str] = Field(default=None, description="Column for Y axis")
    color_by: Optional[str] = Field(default=None, description="Column used for colour encoding")
    title: str = Field(default="", description="Chart title")
    description: str = Field(default="", description="Human-readable description of what the chart shows")


class ExplorationReport(BaseModel):
    """Results of automated exploratory data analysis."""

    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Pairwise correlation coefficients")
    statistical_tests: Dict[str, Any] = Field(default_factory=dict, description="Results of statistical tests (normality, chi-sq, etc.)")
    viz_specs: List[VisualizationSpec] = Field(default_factory=list, description="Recommended visualizations")
    summary: str = Field(default="", description="Natural-language summary of key findings")


class ModelCandidate(BaseModel):
    """A single trained model with its evaluation scores."""

    name: str = Field(description="Unique display name for the model")
    model_type: str = Field(description="Algorithm type (e.g. RandomForest, XGBoost, LogisticRegression)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters used")
    train_score: float = Field(default=0.0, description="Score on training set")
    val_score: float = Field(default=0.0, description="Score on validation / test set")
    train_time_seconds: float = Field(default=0.0, ge=0.0, description="Wall-clock training time in seconds")


class ModelLeaderboard(BaseModel):
    """Ranked collection of model candidates."""

    candidates: List[ModelCandidate] = Field(default_factory=list, description="All evaluated model candidates")
    champion_name: Optional[str] = Field(default=None, description="Name of the best model")
    champion_model: Optional[Any] = Field(default=None, description="Reference to the fitted champion model object")
    problem_type: str = Field(default="eda_only", description="Problem type the models were trained for")
    best_metric_name: str = Field(default="", description="Name of the metric used to rank models")
    best_metric_value: float = Field(default=0.0, description="Best metric value achieved")

    model_config = {"arbitrary_types_allowed": True}


class DeploymentInfo(BaseModel):
    """Metadata for deploying a trained model."""

    model_path: str = Field(default="", description="Path to the serialized model file")
    requirements: List[str] = Field(default_factory=list, description="Python package requirements")
    inference_script: str = Field(default="", description="Generated inference script content")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring and alerting configuration")


class ChatMessage(BaseModel):
    """A single message in the chat interface."""

    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message text content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")


class PipelineState(BaseModel):
    """Tracks the status of each stage in the ML pipeline."""

    steps: Dict[str, StepStatus] = Field(
        default_factory=lambda: {
            "ingestion": StepStatus.PENDING,
            "profiling": StepStatus.PENDING,
            "cleaning": StepStatus.PENDING,
            "exploration": StepStatus.PENDING,
            "feature_engineering": StepStatus.PENDING,
            "modeling": StepStatus.PENDING,
            "evaluation": StepStatus.PENDING,
            "deployment": StepStatus.PENDING,
        },
        description="Pipeline step name to current status mapping",
    )

    def mark(self, step: str, status: StepStatus) -> None:
        """Update the status of a pipeline step.

        Args:
            step: Name of the pipeline step.
            status: New status to set.
        """
        if step in self.steps:
            self.steps[step] = status

    def is_complete(self) -> bool:
        """Return True if all steps are completed or skipped."""
        return all(
            s in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps.values()
        )

    def current_step(self) -> Optional[str]:
        """Return the name of the currently running step, if any."""
        for name, status in self.steps.items():
            if status == StepStatus.RUNNING:
                return name
        return None
