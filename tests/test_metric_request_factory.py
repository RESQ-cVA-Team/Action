import unittest
from typing import Any, cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DistributionOptions
from src.domain.langchain.schema import ChartSpec, GroupByStrokeType, MetricSpec, NumericResolutionSpec
from src.executors.planning.metric_request_factory import build_metric_requests


class MetricRequestFactoryTests(unittest.TestCase):
    def test_histogram_defaults_distribution_when_missing(self) -> None:
        plan_chart = ChartSpec(chart_type="HISTOGRAM", metrics=[MetricSpec(metric="DTN")])

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (20, 0, 200),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 20)
        self.assertEqual(distribution_options.lower_bound, 0)
        self.assertEqual(distribution_options.upper_bound, 200)
        self.assertTrue(request.include_stats)
        self.assertIsNotNone(derived_axes)
        derived_axes_value = cast(tuple[ChartAxis, ChartAxis], derived_axes)
        self.assertIsInstance(derived_axes_value[0], ChartAxis)
        self.assertIsInstance(derived_axes_value[1], ChartAxis)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_line_chart_requests_distribution_by_default(self) -> None:
        plan_chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="DTN")])

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (15, 10, 90),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 15)
        self.assertEqual(distribution_options.lower_bound, 10)
        self.assertEqual(distribution_options.upper_bound, 90)
        self.assertTrue(request.include_stats)
        self.assertIsNone(derived_axes)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_line_chart_applies_value_domain_override_as_bounds(self) -> None:
        plan_chart = ChartSpec(
            chart_type="LINE",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 10, "upperBound": 180}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (15, 0, 240),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 15)
        self.assertEqual(distribution_options.lower_bound, 10)
        self.assertEqual(distribution_options.upper_bound, 180)
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, 10)
        self.assertEqual(metric_options_value.upper_boundary, 180)
        self.assertIsNone(derived_axes)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_grouped_chart_requests_distribution_by_default(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="DTN")],
            group_by=[GroupByStrokeType()],
        )

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (25, 5, 105),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 25)
        self.assertEqual(distribution_options.lower_bound, 5)
        self.assertEqual(distribution_options.upper_bound, 105)
        self.assertTrue(request.include_stats)
        self.assertIsNone(derived_axes)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_histogram_merges_numeric_resolution_overrides(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate(
                {
                    "valueDomain": {"lowerBound": 25},
                    "bucketing": {"bucketCount": 8},
                }
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (20, 0, 200),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 8)
        self.assertEqual(distribution_options.lower_bound, 25)
        self.assertEqual(distribution_options.upper_bound, 200)
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, 25)
        self.assertEqual(metric_options_value.upper_boundary, 200)
        self.assertIsNotNone(derived_axes)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_histogram_computes_bucket_count_from_bucket_size(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate(
                {
                    "valueDomain": {"lowerBound": 0, "upperBound": 95},
                    "bucketing": {"bucketSize": 10},
                }
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (20, 0, 200),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 10)

    def test_bar_chart_merges_partial_value_domain_override_in_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"upperBound": 130}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(
            plan_chart=plan_chart,
            derive_defaults_fn=lambda metric: (30, 5, 150),
            axis_from_meta_fn=lambda metric, lower, upper: (
                ChartAxis(label=str(lower)),
                ChartAxis(label=str(upper)),
            ),
        )

        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 30)
        self.assertEqual(distribution_options.lower_bound, 5)
        self.assertEqual(distribution_options.upper_bound, 130)
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, 5)
        self.assertEqual(metric_options_value.upper_boundary, 130)


if __name__ == "__main__":
    unittest.main()
