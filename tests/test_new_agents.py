"""Tests for ForecastAgent, InsightsAgent, and ReportGeneratorAgent."""

import pytest
import numpy as np
import pandas as pd

from agents.forecast_agent import ForecastAgent
from agents.insights_agent import InsightsAgent
from agents.report_generator_agent import ReportGeneratorAgent


# =========================================================================
# ForecastAgent
# =========================================================================

class TestForecastAgent:
    @pytest.mark.asyncio
    async def test_check_eligibility(self, sample_timeseries_df):
        agent = ForecastAgent()
        result = await agent.run({
            "action": "check_eligibility",
            "dataframe": sample_timeseries_df,
        })
        assert result.success
        eligible = result.data.get("eligible_metrics", [])
        # revenue and units_sold should be eligible (60 rows > 15 threshold)
        assert len(eligible) > 0

    @pytest.mark.asyncio
    async def test_forecast_execution(self, sample_timeseries_df):
        """Forecast should either succeed with Prophet or fail gracefully without it."""
        agent = ForecastAgent()
        result = await agent.run({
            "action": "forecast",
            "dataframe": sample_timeseries_df,
            "target_column": "revenue",
        })
        # If Prophet is not installed, it should fail gracefully
        # If installed, it should succeed
        # Either way, it should not crash
        if result.success:
            assert "forecast" in result.data
            assert "summary" in result.data
            assert result.data["summary"]["trend_direction"] in ("upward", "downward", "flat")
        else:
            assert "prophet" in result.error.lower() or "error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_forecast_auto_detect_date(self, sample_timeseries_df):
        """Should auto-detect the date column."""
        agent = ForecastAgent()
        result = await agent.run({
            "action": "check_eligibility",
            "dataframe": sample_timeseries_df,
        })
        assert result.success
        assert result.data.get("date_column") == "date"

    @pytest.mark.asyncio
    async def test_forecast_insufficient_data(self):
        """Should fail gracefully with too little data."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        agent = ForecastAgent()
        result = await agent.run({
            "action": "check_eligibility",
            "dataframe": df,
        })
        assert result.success
        # value should NOT be eligible (only 5 rows < 15 threshold)
        eligible = result.data.get("eligible_metrics", [])
        assert "value" not in eligible

    @pytest.mark.asyncio
    async def test_forecast_no_date_column(self):
        """Should handle missing date column."""
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randn(50),
        })
        agent = ForecastAgent()
        result = await agent.run({
            "action": "forecast",
            "dataframe": df,
            "target_column": "x",
        })
        assert not result.success
        assert "date" in result.error.lower() or "time" in result.error.lower()

    @pytest.mark.asyncio
    async def test_forecast_no_dataframe(self):
        agent = ForecastAgent()
        result = await agent.run({"action": "forecast"})
        assert not result.success


# =========================================================================
# InsightsAgent
# =========================================================================

class TestInsightsAgent:
    @pytest.mark.asyncio
    async def test_generate_insights(self, sample_classification_df):
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_classification_df,
        })
        assert result.success
        assert "insights" in result.data
        assert isinstance(result.data["insights"], list)

    @pytest.mark.asyncio
    async def test_insights_on_regression_data(self, sample_regression_df):
        """InsightsAgent should produce insights on regression-style data."""
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_regression_df,
        })
        assert result.success
        # Should produce at least recommendations even if no strong insights
        recs = result.data.get("recommendations", [])
        assert len(recs) > 0

    @pytest.mark.asyncio
    async def test_insights_recommendations(self, sample_customer_support_df):
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_customer_support_df,
        })
        assert result.success
        recs = result.data.get("recommendations", [])
        assert len(recs) > 0
        for r in recs:
            assert "action" in r
            assert "detail" in r

    @pytest.mark.asyncio
    async def test_insights_executive_summary(self, sample_classification_df):
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_classification_df,
        })
        assert result.success
        summary = result.data.get("executive_summary", "")
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_insights_with_target(self, sample_classification_df):
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_classification_df,
            "target_column": "target",
        })
        assert result.success
        assert len(result.data["insights"]) > 0

    @pytest.mark.asyncio
    async def test_insights_no_dataframe(self):
        agent = InsightsAgent()
        result = await agent.run({"action": "generate_insights"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_insights_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": df,
        })
        assert result.success

    @pytest.mark.asyncio
    async def test_insights_with_stringdtype(self, sample_stringdtype_df):
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_stringdtype_df,
        })
        assert result.success
        assert "insights" in result.data

    @pytest.mark.asyncio
    async def test_insights_timeseries(self, sample_timeseries_df):
        """InsightsAgent should detect trends in time series data."""
        agent = InsightsAgent()
        result = await agent.run({
            "action": "generate_insights",
            "dataframe": sample_timeseries_df,
        })
        assert result.success
        assert len(result.data["insights"]) > 0


# =========================================================================
# ReportGeneratorAgent
# =========================================================================

class TestReportGeneratorAgent:
    @pytest.mark.asyncio
    async def test_generate_all_formats(self, sample_classification_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "format": "all",
        })
        assert result.success
        assert "markdown" in result.data
        assert "html" in result.data
        assert "csv_summary" in result.data

    @pytest.mark.asyncio
    async def test_generate_markdown_only(self, sample_classification_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "format": "markdown",
        })
        assert result.success
        md = result.data["markdown"]
        assert "# " in md  # Has heading
        assert "Records" in md
        assert "Features" in md

    @pytest.mark.asyncio
    async def test_generate_html_only(self, sample_classification_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "format": "html",
        })
        assert result.success
        html = result.data["html"]
        assert "<!DOCTYPE html>" in html
        assert "Dataset Overview" in html

    @pytest.mark.asyncio
    async def test_generate_csv_only(self, sample_classification_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "format": "csv",
        })
        assert result.success
        csv = result.data["csv_summary"]
        assert "category,priority,title,detail" in csv

    @pytest.mark.asyncio
    async def test_report_with_insights(self, sample_classification_df):
        """Report should include insights and recommendations when provided."""
        insights = [
            {"title": "High correlation", "narrative": "Features x and y are correlated", "priority": "high", "category": "correlation"},
            {"title": "Skewed distribution", "narrative": "Feature z is skewed", "priority": "medium", "category": "distribution"},
        ]
        recommendations = [
            {"action": "Transform features", "detail": "Apply log transform", "priority": "high"},
        ]
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "insights": insights,
            "recommendations": recommendations,
            "format": "all",
        })
        assert result.success
        md = result.data["markdown"]
        assert "High correlation" in md or "Features x and y" in md
        assert "Transform features" in md

    @pytest.mark.asyncio
    async def test_report_with_model_results(self, sample_classification_df):
        """Report should include model comparison table."""
        model_results = {
            "results": {
                "RF": {"metrics": {"accuracy": 0.85, "f1": 0.84, "cv_mean": 0.83}},
                "LR": {"metrics": {"accuracy": 0.78, "f1": 0.77, "cv_mean": 0.76}},
            },
            "best_model": "RF",
            "task_type": "classification",
        }
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "model_results": model_results,
            "format": "markdown",
        })
        assert result.success
        md = result.data["markdown"]
        assert "Model Performance" in md
        assert "RF" in md
        assert "BEST" in md

    @pytest.mark.asyncio
    async def test_report_no_dataframe(self):
        agent = ReportGeneratorAgent()
        result = await agent.run({"action": "generate_report"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_report_with_stringdtype(self, sample_stringdtype_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_stringdtype_df,
            "format": "all",
        })
        assert result.success
        assert "markdown" in result.data
        assert "html" in result.data

    @pytest.mark.asyncio
    async def test_report_custom_title(self, sample_classification_df):
        agent = ReportGeneratorAgent()
        result = await agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "title": "My Custom Report",
            "format": "markdown",
        })
        assert result.success
        assert "My Custom Report" in result.data["markdown"]


# =========================================================================
# New Agents + FallbackClient Routing
# =========================================================================

class TestFallbackClientRouting:
    """Test that FallbackClient routes forecast/insights/report intents."""

    @pytest.mark.asyncio
    async def test_forecast_intent(self):
        from llm.client import FallbackClient
        client = FallbackClient()
        result = await client.chat_json(
            messages=[{"role": "user", "content": "forecast future trends"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "ForecastAgent"

    @pytest.mark.asyncio
    async def test_insights_intent(self):
        from llm.client import FallbackClient
        client = FallbackClient()
        result = await client.chat_json(
            messages=[{"role": "user", "content": "generate business insights"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "InsightsAgent"

    @pytest.mark.asyncio
    async def test_report_intent(self):
        from llm.client import FallbackClient
        client = FallbackClient()
        result = await client.chat_json(
            messages=[{"role": "user", "content": "generate report for download"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "ReportGeneratorAgent"


# =========================================================================
# Integration: Pipeline with new agents
# =========================================================================

class TestNewAgentsPipeline:
    @pytest.mark.asyncio
    async def test_insights_then_report(self, sample_classification_df):
        """InsightsAgent output should feed into ReportGeneratorAgent."""
        # Step 1: Generate insights
        insights_agent = InsightsAgent()
        insights_result = await insights_agent.run({
            "action": "generate_insights",
            "dataframe": sample_classification_df,
        })
        assert insights_result.success

        # Step 2: Generate report with those insights
        report_agent = ReportGeneratorAgent()
        report_result = await report_agent.run({
            "action": "generate_report",
            "dataframe": sample_classification_df,
            "insights": insights_result.data.get("insights", []),
            "recommendations": insights_result.data.get("recommendations", []),
            "format": "all",
        })
        assert report_result.success
        assert "markdown" in report_result.data
        assert "html" in report_result.data
