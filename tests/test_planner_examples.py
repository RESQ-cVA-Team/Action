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

    def test_monthly_examples_have_expected_analysis_modes(self) -> None:
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

        self.assertEqual(trend_plan.charts[0].analysis_mode, "TIME_SERIES")
        self.assertEqual(distribution_plan.charts[0].analysis_mode, "DISTRIBUTION")
        self.assertEqual(box_plan.charts[0].analysis_mode, "SUMMARY")
        self.assertEqual(over_time_plan.charts[0].analysis_mode, "TIME_SERIES")

    def test_distribution_examples_have_no_time_groupby(self) -> None:
        """DISTRIBUTION+GroupByTime is rejected by the semantic adapter; examples must not teach it."""
        few_shots = examples.get_few_shot_examples()
        for item in few_shots:
            plan = schema.AnalysisPlan.model_validate(json.loads(item["assistant"]))
            for chart in (plan.charts or []):
                if chart.analysis_mode == "DISTRIBUTION":
                    for gb in (chart.group_by or []):
                        self.assertNotIsInstance(
                            gb,
                            schema.GroupByTime,
                            msg=f"Example '{item['user'][:80]}' has DISTRIBUTION+GroupByTime which is not supported",
                        )


if __name__ == "__main__":
    unittest.main()
