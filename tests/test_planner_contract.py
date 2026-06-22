import unittest
from unittest.mock import patch

from src.domain.langchain.schema import AnalysisPlan, HistogramChartSpec, LineChartSpec
from src.planners.langchain.pipeline import PlannerIntentAmbiguityError


class PlannerContractTests(unittest.TestCase):
    """Test the hardened planner contract: canonical-or-fail with explicit KPI clarification."""

    def test_missing_metric_triggers_clarification(self) -> None:
        """User provides no metric entities -> clarification, not silent failure."""
        with patch("src.planners.langchain.pipeline.plan_chain") as mock_chain:
            mock_chain.invoke.return_value = '{"charts": []}'

            with self.assertRaises(PlannerIntentAmbiguityError) as ctx:
                from src.planners.langchain.pipeline import generate_analysis_plan
                generate_analysis_plan(
                    question="Show me something",
                    entities={},
                    language="en",
                    max_retries=1,
                    debug=False,
                )

            self.assertIn("which KPI metric", str(ctx.exception))

    def test_metric_present_but_planner_fails_raises_strict_error(self) -> None:
        """User provides metric but planner output is invalid -> strict ValueError, no clarification."""
        with patch("src.planners.langchain.pipeline.plan_chain") as mock_chain:
            mock_chain.invoke.return_value = '{"charts": []}'

            with self.assertRaises(ValueError) as ctx:
                from src.planners.langchain.pipeline import generate_analysis_plan
                generate_analysis_plan(
                    question="Show me DTN with invalid planner payload",
                    entities={"metric": ["DTN"]},
                    language="en",
                    max_retries=1,
                    debug=False,
                )

            error_text = str(ctx.exception)
            self.assertNotIn("which KPI metric", error_text)
            self.assertIn("failed to produce", error_text)
            self.assertNotIsInstance(ctx.exception, PlannerIntentAmbiguityError)

    def test_valid_plan_succeeds(self) -> None:
        """Valid canonical plan output -> success."""
        valid_plan_json = """
        {
            "charts": [
                {
                    "chartType": "HISTOGRAM",
                    "xAxis": {"kind": "numeric_metric", "metric": "DTN"},
                    "yAxis": {"kind": "count"}
                }
            ]
        }
        """

        with patch("src.planners.langchain.pipeline.plan_chain") as mock_chain:
            mock_chain.invoke.return_value = valid_plan_json

            from src.planners.langchain.pipeline import generate_analysis_plan
            result = generate_analysis_plan(
                question="Show me DTN as histogram default",
                entities={"metric": ["DTN"]},
                language="en",
                max_retries=1,
                debug=False,
            )

            self.assertIsInstance(result, AnalysisPlan)
            self.assertEqual(len(result.charts), 1)
            self.assertIsInstance(result.charts[0], HistogramChartSpec)

    def test_distribution_defaults_to_histogram_when_chart_type_not_explicit(self) -> None:
        """Without explicit chart type, distribution-as-line output is rejected."""
        line_distribution_json = """
        {
            "charts": [
                {
                    "chartType": "LINE",
                    "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 20}},
                    "yAxes": {"y1": {"kind": "count"}},
                    "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}]
                }
            ]
        }
        """

        with patch("src.planners.langchain.pipeline.plan_chain") as mock_chain:
            mock_chain.invoke.return_value = line_distribution_json

            with self.assertRaises(ValueError) as ctx:
                from src.planners.langchain.pipeline import generate_analysis_plan
                generate_analysis_plan(
                    question="Show me DTN distribution with unspecified chart type",
                    entities={"metric": ["DTN"]},
                    language="en",
                    max_retries=1,
                    debug=False,
                )

            self.assertIn("failed to produce", str(ctx.exception))
            self.assertNotIn("which KPI metric", str(ctx.exception))

    def test_explicit_line_chart_request_allows_line_distribution(self) -> None:
        """Explicit chart_type=LINE preserves user override for distribution chart type."""
        line_distribution_json = """
        {
            "charts": [
                {
                    "chartType": "LINE",
                    "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 20}},
                    "yAxes": {"y1": {"kind": "count"}},
                    "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}]
                }
            ]
        }
        """

        with patch("src.planners.langchain.pipeline.plan_chain") as mock_chain:
            mock_chain.invoke.return_value = line_distribution_json

            from src.planners.langchain.pipeline import generate_analysis_plan
            result = generate_analysis_plan(
                question="Show me a line chart of DTN",
                entities={"metric": ["DTN"], "chart_type": ["LINE"]},
                language="en",
                max_retries=1,
                debug=False,
            )

            self.assertIsInstance(result, AnalysisPlan)
            self.assertEqual(len(result.charts), 1)
            self.assertIsInstance(result.charts[0], LineChartSpec)


if __name__ == "__main__":
    unittest.main()
