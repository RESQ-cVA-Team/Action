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
chart_builder = load_module("action_chart_builder_under_test", "src/executors/mapping/chart_builder.py")


class VisualizationSemanticsTests(unittest.TestCase):
    def test_origin_scope_rejects_legacy_group_id(self) -> None:
        with self.assertRaises(ValidationError):
            schema.OriginScopeSpec(scopeType="group_id", value=1)

    def test_origin_scope_accepts_canonical_provider_group_id(self) -> None:
        spec = schema.OriginScopeSpec(scopeType="provider_group_id", value=1)
        self.assertEqual(spec.scope_type, "provider_group_id")

    def test_chart_spec_requires_explicit_axes(self) -> None:
        with self.assertRaises(ValidationError):
            schema.ChartSpec(chart_type="LINE")

    def test_numeric_xaxis_builds_distribution_request(self) -> None:
        chart = schema.ChartSpec(
            chart_type="HISTOGRAM",
            xAxis={"metric": "DTN", "bins": 18, "minValue": 5, "maxValue": 95},
            yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "COUNT"}],
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

    def test_time_xaxis_keeps_stats_request(self) -> None:
        chart = schema.ChartSpec(
            chart_type="LINE",
            xAxis={"grain": "MONTH", "window": {"last_n": 24, "unit": "MONTH"}},
            yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "MEAN"}],
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

    def test_mixed_unit_yaxis_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            schema.ChartSpec(
                chart_type="LINE",
                xAxis={"grain": "MONTH"},
                yAxes=[
                    {
                        "metrics": [
                            {"metric": "DTN"},
                            {"metric": "AGE"},
                        ],
                        "statistic": "MEAN",
                    }
                ],
            )

    def test_histogram_requires_numeric_xaxis(self) -> None:
        with self.assertRaises(ValidationError):
            schema.ChartSpec(
                chart_type="HISTOGRAM",
                xAxis={"grain": "MONTH"},
                yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "COUNT"}],
            )

    def test_pie_requires_category_xaxis(self) -> None:
        with self.assertRaises(ValidationError):
            schema.ChartSpec(
                chart_type="PIE",
                xAxis={"grain": "MONTH"},
                yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "COUNT"}],
            )

    def test_scatter_is_explicitly_unsupported_in_v1(self) -> None:
        with self.assertRaises(ValidationError):
            schema.ChartSpec(
                chart_type="SCATTER",
                xAxis={"grain": "MONTH"},
                yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "MEAN"}],
            )

    def test_grouped_box_chart_emits_one_entry_per_series(self) -> None:
        plan = schema.ChartSpec(
            chart_type="BOX",
            xAxis={"groupBy": {"categories": ["MALE", "FEMALE"]}},
            yAxes=[{"metrics": [{"metric": "DTN"}], "statistic": "MEDIAN"}],
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
