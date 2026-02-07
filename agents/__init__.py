"""
Data Science Agent Platform - Agents Module
"""

import pandas as pd

# CRITICAL: Disable pandas 3.x StringDtype globally.
# Without this, pd.read_csv() creates StringDtype columns which crash
# numpy operations (corr, get_dummies, factorize) across all agents.
pd.set_option("future.infer_string", False)

from .base_agent import (
    BaseAgent,
    AgentState,
    AgentMessage,
    MessageType,
    TaskResult,
    generate_uuid,
    is_valid_uuid,
    get_numeric_cols,
    get_categorical_cols,
    get_datetime_cols,
    get_numeric_df,
    _sanitize_dataframe,
)
from .coordinator_agent import CoordinatorAgent, Workflow, WorkflowStep
from .data_cleaner_agent import DataCleanerAgent
from .eda_agent import EDAAgent
from .feature_engineer_agent import FeatureEngineerAgent
from .model_trainer_agent import ModelTrainerAgent
from .automl_agent import AutoMLAgent
from .dashboard_builder_agent import DashboardBuilderAgent
from .data_visualizer_agent import DataVisualizerAgent
from .forecast_agent import ForecastAgent
from .insights_agent import InsightsAgent
from .report_generator_agent import ReportGeneratorAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "AgentMessage",
    "MessageType",
    "TaskResult",
    "generate_uuid",
    "is_valid_uuid",
    "CoordinatorAgent",
    "Workflow",
    "WorkflowStep",
    "DataCleanerAgent",
    "EDAAgent",
    "FeatureEngineerAgent",
    "ModelTrainerAgent",
    "AutoMLAgent",
    "DashboardBuilderAgent",
    "DataVisualizerAgent",
    "ForecastAgent",
    "InsightsAgent",
    "ReportGeneratorAgent",
]
