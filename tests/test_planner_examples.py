import importlib.util
import json
import unittest
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


examples = load_module("action_planner_examples_under_test", "src/planners/langchain/examples.py")
schema = load_module("action_schema_under_test", "src/domain/langchain/schema.py")


class PlannerExamplesTests(unittest.TestCase):
    def test_few_shot_examples_include_monthly_trend_and_distribution(self) -> None:
        few_shots = examples.get_few_shot_examples()
        users = [item["user"] for item in few_shots]

        self.assertTrue(any("monthly trend of dtn" in user.lower() for user in users))
        self.assertTrue(any("distribution of dtn by month" in user.lower() for user in users))
        self.assertTrue(any("box plot of dtn by sex" in user.lower() for user in users))
        self.assertTrue(any("dtn over time" in user.lower() for user in users), "Missing 'dtn over time' TIME_SERIES example")

    def test_monthly_examples_have_expected_xaxis_shapes(self) -> None:
        few_shots = examples.get_few_shot_examples()

        def parse_plan(user_phrase: str) -> schema.AnalysisPlan:
            for item in few_shots:
                if user_phrase in item["user"].lower():
                    return schema.AnalysisPlan.model_validate(json.loads(item["assistant"]))
            raise AssertionError(f"Missing few-shot example containing: {user_phrase}")

        trend_plan = parse_plan("monthly trend of dtn")
        distribution_plan = parse_plan("distribution of dtn by month")
        box_plan = parse_plan("box plot of dtn by sex")
        over_time_plan = parse_plan("dtn over time")

        trend_chart = trend_plan.charts[0]
        distribution_chart = distribution_plan.charts[0]
        box_chart = box_plan.charts[0]
        over_time_chart = over_time_plan.charts[0]

        self.assertEqual(type(trend_chart).__name__, "LineChartSpec")
        self.assertEqual(type(distribution_chart).__name__, "HistogramChartSpec")
        self.assertEqual(type(box_chart).__name__, "LineChartSpec")
        self.assertEqual(type(over_time_chart).__name__, "LineChartSpec")

        self.assertEqual(type(trend_chart.x_axes["x1"]).__name__, "TimeXAxis")
        self.assertEqual(type(distribution_chart.x_axis).__name__, "NumericMetricXAxis")
        self.assertEqual(type(box_chart.x_axes["x1"]).__name__, "CategoryXAxis")
        self.assertEqual(type(over_time_chart.x_axes["x1"]).__name__, "TimeXAxis")

    def test_distribution_examples_use_numeric_metric_xaxis(self) -> None:
        """Histogram examples must use explicit NumericMetricXAxis in the plan domain."""
        few_shots = examples.get_few_shot_examples()
        for item in few_shots:
            try:
                plan = schema.AnalysisPlan.model_validate(json.loads(item["assistant"]))
            except Exception:
                # Ignore unrelated malformed examples; this test only enforces
                # the DISTRIBUTION+GroupByTime constraint on valid examples.
                continue
            for chart in (plan.charts or []):
                if chart.chart_type == "HISTOGRAM":
                    self.assertEqual(
                        type(chart.x_axis).__name__,
                        "NumericMetricXAxis",
                        msg=f"Example '{item['user'][:80]}' uses HISTOGRAM without NumericMetricXAxis",
                    )


if __name__ == "__main__":
    unittest.main()
