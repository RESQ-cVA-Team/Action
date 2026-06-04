import importlib.util
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


schema = load_module("action_schema_under_test", "src/domain/langchain/schema.py")
metric_request_factory = load_module("action_metric_request_factory_under_test", "src/executors/planning/metric_request_factory.py")
chart_types = load_module("action_chart_types_under_test", "src/domain/dto/charts/types.py")
semantic_adapter = load_module("action_semantic_adapter_under_test", "src/planners/langchain/semantic_adapter.py")
chart_builder = load_module("action_chart_builder_under_test", "src/executors/mapping/chart_builder.py")
query_compiler = load_module("action_query_compiler_under_test", "src/executors/planning/query_compiler.py")


class VisualizationSemanticsTests(unittest.TestCase):
    def test_chart_spec_normalizes_analysis_mode(self) -> None:
        chart = schema.ChartSpec(
            chart_type="line",
            analysis_mode="distribution",
            metrics=[schema.MetricSpec(metric="DTN")],
        )

        self.assertEqual(chart.chart_type, "LINE")
        self.assertEqual(chart.analysis_mode, "DISTRIBUTION")

    def test_distribution_mode_builds_distribution_request(self) -> None:
        chart = schema.ChartSpec(
            chart_type="LINE",
            analysis_mode="DISTRIBUTION",
            metrics=[schema.MetricSpec(metric="DTN")],
        )

        derived_axes_calls: list[tuple[str, int, int]] = []

        def derive_defaults(metric_code: str) -> tuple[int, int, int]:
            self.assertEqual(metric_code, "DTN")
            return 18, 5, 95

        def axis_from_meta(metric_code: str, lower: int, upper: int):
            derived_axes_calls.append((metric_code, lower, upper))
            return chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")

        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=derive_defaults,
            axis_from_meta_fn=axis_from_meta,
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_distribution)
        self.assertFalse(metric_requests[0].include_stats)
        self.assertIsNotNone(metric_requests[0].distribution_options)
        self.assertEqual(metric_requests[0].distribution_options.bin_count, 18)
        self.assertEqual(metric_requests[0].distribution_options.lower_bound, 5)
        self.assertEqual(metric_requests[0].distribution_options.upper_bound, 95)
        self.assertEqual(derived_axes_calls, [("DTN", 5, 95)])
        self.assertEqual(derived_axes, (chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")))

    def test_time_grouping_keeps_stats_request(self) -> None:
        plan = schema.AnalysisPlan(
            charts=[
                schema.ChartSpec(
                    chart_type="LINE",
                    analysis_mode="TIME_SERIES",
                    group_by=[schema.GroupByTime(grain="MONTH")],
                    metrics=[schema.MetricSpec(metric="DTN")],
                )
            ],
            statistical_tests=None,
        )

        chart = semantic_adapter.validate_analysis_plan_semantics(plan).charts[0]

        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=lambda metric_code: (12, 0, 100),
            axis_from_meta_fn=lambda metric_code, lower, upper: (chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")),
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_stats)
        self.assertFalse(metric_requests[0].include_distribution)
        self.assertIsNone(metric_requests[0].distribution_options)
        self.assertIsNone(derived_axes)

    def test_time_series_without_groupby_gets_default_month_window(self) -> None:
        plan = schema.AnalysisPlan(
            charts=[
                schema.ChartSpec(
                    chart_type="LINE",
                    analysis_mode="TIME_SERIES",
                    metrics=[schema.MetricSpec(metric="DTN")],
                )
            ],
            statistical_tests=None,
        )

        normalized = semantic_adapter.validate_analysis_plan_semantics(plan)
        chart = normalized.charts[0]

        self.assertEqual(chart.analysis_mode, "TIME_SERIES")
        self.assertEqual(len(chart.group_by or []), 1)
        time_group = chart.group_by[0]
        self.assertEqual(type(time_group).__name__, "GroupByTime")
        self.assertEqual(time_group.grain, "MONTH")
        self.assertIsNotNone(time_group.window)
        self.assertEqual(time_group.window.last_n, 24)
        self.assertEqual(time_group.window.unit, "MONTH")

        compiled = query_compiler.compile_chart_grouping(chart)
        self.assertTrue(compiled.batches[0].batched_time_enabled)
        self.assertGreater(len(compiled.batches[0].batched_time_periods), 0)

    def test_schema_rejects_auto_analysis_mode(self) -> None:
        with self.assertRaises(ValidationError):
            schema.AnalysisPlan.model_validate(
                {
                    "charts": [
                        {
                            "chart_type": "LINE",
                            "analysisMode": "AUTO",
                            "metrics": [{"metric": "DTN"}],
                        }
                    ],
                    "statistical_tests": None,
                }
            )

    def test_grouped_chart_infers_comparison_mode(self) -> None:
        plan = schema.AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "LINE",
                        "analysisMode": "COMPARISON",
                        "group_by": [{"categories": None}],
                        "metrics": [{"metric": "DTN"}],
                    }
                ],
                "statistical_tests": None,
            }
        )

        normalized = semantic_adapter.validate_analysis_plan_semantics(plan)
        self.assertEqual(normalized.charts[0].analysis_mode, "COMPARISON")

    def test_grouped_distribution_stays_on_distribution_path(self) -> None:
        plan = schema.AnalysisPlan(
            charts=[
                schema.ChartSpec(
                    chart_type="HISTOGRAM",
                    analysis_mode="DISTRIBUTION",
                    group_by=[schema.GroupBySex(categories=["MALE", "FEMALE"])],
                    metrics=[schema.MetricSpec(metric="DTN")],
                )
            ],
            statistical_tests=None,
        )

        normalized = semantic_adapter.validate_analysis_plan_semantics(plan)
        self.assertEqual(normalized.charts[0].analysis_mode, "DISTRIBUTION")
        self.assertEqual(normalized.charts[0].chart_type, "BAR")

        metric_requests, _, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=normalized.charts[0],
            derive_defaults_fn=lambda metric_code: (18, 5, 95),
            axis_from_meta_fn=lambda metric_code, lower, upper: (chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")),
        )

        self.assertTrue(metric_requests[0].include_distribution)
        self.assertFalse(metric_requests[0].include_stats)

    def test_grouped_box_chart_emits_one_entry_per_series(self) -> None:
        plan = schema.ChartSpec(
            chart_type="BOX",
            analysis_mode="SUMMARY",
            metrics=[schema.MetricSpec(metric="DTN")],
        )
        series = [
            chart_types.ChartSeries(name="Males", data=[chart_types.ChartPoint(x=1, y=10), chart_types.ChartPoint(x=2, y=14)]),
            chart_types.ChartSeries(name="Females", data=[chart_types.ChartPoint(x=1, y=20), chart_types.ChartPoint(x=2, y=24)]),
        ]

        chart = chart_builder.build_chart_dto(
            plan_chart=plan,
            dimensions=[],
            series=series,
            derived_axes=None,
        )

        self.assertEqual(chart.type, chart_types.ChartType.BOX)
        self.assertEqual(len(chart.data), 2)
        self.assertEqual([entry.name for entry in chart.data], ["Males", "Females"])


if __name__ == "__main__":
    unittest.main()
