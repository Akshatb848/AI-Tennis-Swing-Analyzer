"""
Forecast Agent - Time Series Forecasting with Prophet

Provides:
- Automatic time column & metric detection
- Prophet-based forecasting with configurable seasonality
- Forecast summary with trend direction and confidence intervals
- Forecast-eligible metric discovery via SemanticEngine
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import (
    BaseAgent, TaskResult, _sanitize_dataframe,
    get_numeric_cols, get_datetime_cols,
)

logger = logging.getLogger(__name__)


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the best date/time column in the DataFrame."""
    # First check actual datetime columns
    for col in get_datetime_cols(df):
        return col
    # Then try to parse object columns that look like dates
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            name = col.lower()
            if any(kw in name for kw in ["date", "time", "timestamp", "created", "updated"]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > len(parsed) * 0.5:
                        return col
                except Exception:
                    pass
    return None


class ForecastAgent(BaseAgent):
    """Agent for time series forecasting using Prophet."""

    def __init__(self):
        super().__init__(
            name="ForecastAgent",
            description="Time series forecasting and trend analysis",
            capabilities=["time_series_forecast", "trend_analysis", "seasonality_detection"]
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Time Series Forecasting Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "forecast")

        try:
            if action == "forecast":
                return await self._forecast(task)
            elif action == "check_eligibility":
                return await self._check_eligibility(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return TaskResult(success=False, error=str(e))

    async def _forecast(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        df = _sanitize_dataframe(df.copy())

        date_col = task.get("date_column") or _detect_date_column(df)
        target_col = task.get("target_column")
        periods = task.get("periods", 30)
        freq = task.get("freq", "D")

        if not date_col:
            return TaskResult(success=False, error="No date column found. Provide 'date_column' or ensure a datetime column exists.")

        if not target_col:
            # Pick the first numeric column that isn't the date
            numeric = [c for c in get_numeric_cols(df) if c != date_col]
            if not numeric:
                return TaskResult(success=False, error="No numeric column available for forecasting.")
            target_col = numeric[0]

        if target_col not in df.columns:
            return TaskResult(success=False, error=f"Target column '{target_col}' not found.")

        # Prepare Prophet data
        try:
            ts = df[[date_col, target_col]].dropna()
            ts.columns = ["ds", "y"]
            ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
            ts = ts.dropna()
            ts = ts.groupby("ds")["y"].mean().reset_index()
            ts = ts.sort_values("ds")
        except Exception as e:
            return TaskResult(success=False, error=f"Failed to prepare time series: {e}")

        if len(ts) < 15:
            return TaskResult(
                success=False,
                error=f"Need at least 15 data points for forecasting, got {len(ts)}.",
            )

        # Train Prophet
        try:
            from prophet import Prophet
        except ImportError:
            return TaskResult(success=False, error="Prophet is not installed. Run: pip install prophet")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95,
            )
            model.fit(ts)

            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)

        # Build summary
        future_only = forecast[forecast["ds"] > ts["ds"].max()]
        trend_start = float(forecast["trend"].iloc[0])
        trend_end = float(forecast["trend"].iloc[-1])
        trend_dir = "upward" if trend_end > trend_start else ("downward" if trend_end < trend_start else "flat")

        summary = {
            "date_column": date_col,
            "target_column": target_col,
            "historical_points": len(ts),
            "forecast_periods": periods,
            "frequency": freq,
            "trend_direction": trend_dir,
            "predicted_mean": float(future_only["yhat"].mean()) if len(future_only) > 0 else None,
            "predicted_min": float(future_only["yhat_lower"].min()) if len(future_only) > 0 else None,
            "predicted_max": float(future_only["yhat_upper"].max()) if len(future_only) > 0 else None,
            "confidence_interval_width": float(
                (future_only["yhat_upper"] - future_only["yhat_lower"]).mean()
            ) if len(future_only) > 0 else None,
        }

        # Forecast data for plotting
        forecast_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
        forecast_data["ds"] = forecast_data["ds"].astype(str)

        return TaskResult(
            success=True,
            data={
                "summary": summary,
                "forecast": forecast_data.to_dict(orient="records"),
                "historical": ts.assign(ds=ts["ds"].astype(str)).to_dict(orient="records"),
            },
            metrics={"forecast_periods": periods, "trend": trend_dir},
        )

    async def _check_eligibility(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        df = _sanitize_dataframe(df.copy())
        date_col = task.get("date_column") or _detect_date_column(df)

        if not date_col:
            return TaskResult(
                success=True,
                data={"eligible": False, "reason": "No date column found", "eligible_metrics": []},
            )

        eligible = []
        for col in get_numeric_cols(df):
            if col == date_col:
                continue
            ts = df[[date_col, col]].dropna()
            if len(ts) < 15:
                continue
            mean = ts[col].mean()
            std = ts[col].std()
            if mean == 0:
                continue
            if std / abs(mean) > 2.5:
                continue
            eligible.append(col)

        return TaskResult(
            success=True,
            data={
                "eligible": len(eligible) > 0,
                "date_column": date_col,
                "eligible_metrics": eligible,
                "reason": f"{len(eligible)} metrics eligible for forecasting" if eligible else "No stable metrics found",
            },
        )
