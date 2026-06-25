import unittest
from typing import cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DistributionOptions
from src.domain.langchain.schema import ChartSpec, DistributionSpec, GroupByStrokeType, MetricSpec
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

    def test_histogram_uses_explicit_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[
                MetricSpec(
                    metric="DTN",
                    distribution=DistributionSpec(
                        num_buckets=12,
                        min_value=5,
                        max_value=65,
                    ),
                )
            ],
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
        self.assertEqual(distribution_options.bin_count, 12)
        self.assertEqual(distribution_options.lower_bound, 5)
        self.assertEqual(distribution_options.upper_bound, 65)
        self.assertTrue(request.include_stats)
        self.assertIsNotNone(derived_axes)
        derived_axes_value = cast(tuple[ChartAxis, ChartAxis], derived_axes)
        self.assertIsInstance(derived_axes_value[0], ChartAxis)
        self.assertIsInstance(derived_axes_value[1], ChartAxis)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_line_chart_defaults_distribution_when_missing(self) -> None:
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

    def test_grouped_chart_defaults_distribution_when_missing(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
