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
examples = load_module("action_examples_under_test", "src/planners/langchain/examples.py")


def _line_chart_payload(metric: str = "DTN") -> dict:
    return {
        "chartType": "LINE",
        "xAxes": {"x1": {"kind": "time", "grain": "MONTH"}},
        "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
        "series": [{"metric": metric, "xAxis": "x1", "yAxis": "y1"}],
    }


class VisualizationSemanticsTests(unittest.TestCase):
    def test_origin_scope_rejects_legacy_group_id(self) -> None:
        with self.assertRaises(ValidationError):
            schema.OriginScopeSpec(scopeType="group_id", value=1)

    def test_origin_scope_accepts_canonical_provider_group_id(self) -> None:
        spec = schema.OriginScopeSpec(scopeType="provider_group_id", value=1)
        self.assertEqual(spec.scope_type, "provider_group_id")

    def test_line_chart_requires_explicit_axes(self) -> None:
        with self.assertRaises(ValidationError):
            schema.LineChartSpec(chartType="LINE", xAxes={}, yAxes={}, series=[])

    def test_histogram_builds_distribution_request(self) -> None:
        chart = metric_request_factory.S.HistogramChartSpec(
            chartType="HISTOGRAM",
            xAxis={"kind": "numeric_metric", "metric": "DTN", "bins": 18, "minValue": 5, "maxValue": 95},
            yAxis={"kind": "count"},
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

    def test_histogram_uses_ssot_defaults_when_bins_and_range_are_omitted(self) -> None:
        chart = metric_request_factory.S.HistogramChartSpec(
            chartType="HISTOGRAM",
            xAxis={"kind": "numeric_metric", "metric": "DTN"},
            yAxis={"kind": "count"},
        )

        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=lambda metric_code: (24, 0, 1440),
            axis_from_meta_fn=lambda metric_code, lower, upper: (chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")),
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_distribution)
        self.assertIsNotNone(metric_requests[0].distribution_options)
        self.assertEqual(metric_requests[0].distribution_options.bin_count, 24)
        self.assertEqual(metric_requests[0].distribution_options.lower_bound, 0)
        self.assertEqual(metric_requests[0].distribution_options.upper_bound, 1440)
        self.assertIsNotNone(derived_axes)

    def test_time_xaxis_keeps_stats_request(self) -> None:
        chart = metric_request_factory.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {"x1": {"kind": "time", "grain": "MONTH", "window": {"last_n": 24, "unit": "MONTH"}},},
                "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
            }
        )

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

    def test_line_numeric_xaxis_with_count_builds_distribution_request(self) -> None:
        chart = metric_request_factory.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 12}},
                "yAxes": {"y1": {"kind": "count"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
            }
        )

        derived_axes_calls: list[tuple[str, int, int]] = []

        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=lambda metric_code: (12, 0, 1440),
            axis_from_meta_fn=lambda metric_code, lower, upper: (
                derived_axes_calls.append((metric_code, lower, upper)) or chart_types.ChartAxis(label="x"),
                chart_types.ChartAxis(label="y"),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_distribution)
        self.assertFalse(metric_requests[0].include_stats)
        self.assertIsNotNone(metric_requests[0].distribution_options)
        self.assertEqual(metric_requests[0].distribution_options.bin_count, 12)
        self.assertEqual(metric_requests[0].distribution_options.lower_bound, 0)
        self.assertEqual(metric_requests[0].distribution_options.upper_bound, 1440)
        self.assertEqual(derived_axes_calls, [("DTN", 0, 1440)])
        self.assertIsNotNone(derived_axes)

    def test_line_numeric_xaxis_uses_ssot_bucket_default_when_bins_omitted(self) -> None:
        chart = metric_request_factory.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN"}},
                "yAxes": {"y1": {"kind": "count"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
            }
        )

        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=lambda metric_code: (24, 0, 1440),
            axis_from_meta_fn=lambda metric_code, lower, upper: (chart_types.ChartAxis(label="x"), chart_types.ChartAxis(label="y")),
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_distribution)
        self.assertIsNotNone(metric_requests[0].distribution_options)
        self.assertEqual(metric_requests[0].distribution_options.bin_count, 24)
        self.assertEqual(metric_requests[0].distribution_options.lower_bound, 0)
        self.assertEqual(metric_requests[0].distribution_options.upper_bound, 1440)
        self.assertIsNotNone(derived_axes)

    def test_line_series_split_on_numeric_metric_compiles_one_distribution_request(self) -> None:
        # Split semantics are explicit via seriesSplit. A numeric_metric/count LINE
        # compiles into one distribution request; grouped fan-out happens in query compilation.
        chart = metric_request_factory.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 52}},
                "yAxes": {"y1": {"kind": "count"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                "seriesSplit": {"categories": ["MALE", "FEMALE"]},
            }
        )

        derived_axes_calls: list[tuple[str, int, int]] = []
        metric_requests, derived_axes, _, _ = metric_request_factory.build_metric_requests(
            plan_chart=chart,
            derive_defaults_fn=lambda metric_code: (52, 0, 520),
            axis_from_meta_fn=lambda metric_code, lower, upper: (
                derived_axes_calls.append((metric_code, lower, upper)) or chart_types.ChartAxis(label="x"),
                chart_types.ChartAxis(label="y"),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        self.assertTrue(metric_requests[0].include_distribution)
        self.assertFalse(metric_requests[0].include_stats)
        self.assertIsNotNone(metric_requests[0].distribution_options)
        self.assertEqual(metric_requests[0].distribution_options.bin_count, 52)
        self.assertEqual(metric_requests[0].distribution_options.lower_bound, 0)
        self.assertEqual(metric_requests[0].distribution_options.upper_bound, 520)
        self.assertEqual(derived_axes_calls, [("DTN", 0, 520)])
        self.assertIsNotNone(derived_axes)

    def test_line_chart_rejects_category_multi_series_without_series_split(self) -> None:
        with self.assertRaises(ValidationError):
            schema.LineChartSpec.model_validate(
                {
                    "chartType": "LINE",
                    "xAxes": {"x1": {"kind": "category", "groupBy": {"categories": ["MALE", "FEMALE"]}}},
                    "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                    "series": [
                        {"metric": "DTN", "xAxis": "x1", "yAxis": "y1", "label": "A"},
                        {"metric": "DTN", "xAxis": "x1", "yAxis": "y1", "label": "B"},
                    ],
                }
            )

    def test_line_chart_rejects_category_axis_with_series_split(self) -> None:
        with self.assertRaises(ValidationError):
            schema.LineChartSpec.model_validate(
                {
                    "chartType": "LINE",
                    "xAxes": {"x1": {"kind": "category", "groupBy": {"field": "FIRST_CONTACT_PLACE"}}},
                    "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                    "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                    "seriesSplit": {"categories": ["MALE", "FEMALE"]},
                }
            )

    def test_line_chart_rejects_orphan_axis_keys(self) -> None:
        with self.assertRaises(ValidationError):
            schema.LineChartSpec.model_validate(
                {
                    "chartType": "LINE",
                    "xAxes": {
                        "x1": {"kind": "time", "grain": "MONTH"},
                        "x2": {"kind": "time", "grain": "WEEK"},
                    },
                    "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                    "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                }
            )

    def test_line_chart_rejects_empty_groupby_object(self) -> None:
        with self.assertRaises(ValidationError):
            schema.AnalysisPlan.model_validate(
                {
                    "charts": [
                        {
                            "chartType": "LINE",
                            "xAxes": {"x1": {"kind": "category", "groupBy": {}}},
                            "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                            "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                        }
                    ],
                }
            )

    def test_analysis_plan_accepts_without_schema_version(self) -> None:
        plan = schema.AnalysisPlan.model_validate({"charts": [_line_chart_payload()]})
        self.assertIsNotNone(plan)

    def test_analysis_plan_requires_content(self) -> None:
        with self.assertRaises(ValidationError):
            schema.AnalysisPlan.model_validate({})

    def test_few_shot_examples_include_metric_only_kpi_default(self) -> None:
        few_shots = examples.get_few_shot_examples()
        metric_only = next((item for item in few_shots if "USER_UTTERANCE:\nShow me DTN\n" in item["user"]), None)

        self.assertIsNotNone(metric_only)
        assert metric_only is not None
        plan = schema.AnalysisPlan.model_validate_json(metric_only["assistant"])
        self.assertEqual(len(plan.charts), 1)
        chart = plan.charts[0]
        self.assertIsInstance(chart, schema.HistogramChartSpec)
        self.assertEqual(chart.chart_type, "HISTOGRAM")
        self.assertEqual(chart.x_axis.metric, "DTN")
        self.assertEqual(chart.y_axis.kind, "count")

    def test_semantics_allow_line_time_axis_when_explicit(self) -> None:
        plan = schema.AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chartType": "LINE",
                        "xAxes": {"x1": {"kind": "time", "grain": "MONTH"}},
                        "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                        "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                    }
                ],
            }
        )

        self.assertIsNotNone(plan)

    def test_semantics_allow_line_numeric_distribution_when_explicit(self) -> None:
        plan = schema.AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chartType": "LINE",
                        "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 20}},
                        "yAxes": {"y1": {"kind": "count"}},
                        "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                    }
                ],
            }
        )

        self.assertIsNotNone(plan)

    def test_semantics_reject_line_numeric_distribution_with_metric_value_yaxis(self) -> None:
        with self.assertRaisesRegex(ValidationError, "requires count y-axis"):
            schema.AnalysisPlan.model_validate(
                {
                    "charts": [
                        {
                            "chartType": "LINE",
                            "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 20}},
                            "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                            "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                        }
                    ],
                }
            )

    def test_semantics_reject_legacy_sex_filters_without_series_split(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must declare seriesSplit explicitly"):
            schema.AnalysisPlan.model_validate(
                {
                    "charts": [
                        {
                            "chartType": "LINE",
                            "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 20}},
                            "yAxes": {"y1": {"kind": "count"}},
                            "series": [
                                {
                                    "metric": "DTN",
                                    "xAxis": "x1",
                                    "yAxis": "y1",
                                    "filters": {
                                        "op": "predicate",
                                        "field": "SEX",
                                        "operator": "EQ",
                                        "value": "MALE",
                                    },
                                },
                                {
                                    "metric": "DTN",
                                    "xAxis": "x1",
                                    "yAxis": "y1",
                                    "filters": {
                                        "op": "predicate",
                                        "field": "SEX",
                                        "operator": "EQ",
                                        "value": "FEMALE",
                                    },
                                },
                            ],
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
