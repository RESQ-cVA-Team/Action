import unittest
from typing import Any, cast

from src.domain.graphql.request import DistributionOptions
from src.domain.langchain.schema import AnalysisSemanticsSpec, ChartSpec, MeasureSemanticsSpec, MetricSpec, NumericResolutionSpec, SplitSpec
from src.executors.planning.metric_request_factory import build_metric_requests
from src.executors.planning.ssot_metric_defaults import get_distribution_defaults

# DTN SSOT defaults used as baseline for assertions. Tests that apply explicit
# numericResolution overrides assert on the override values, not the SSOT baseline.
_DTN_BINS, _DTN_MIN, _DTN_MAX = get_distribution_defaults("DTN")


class MetricRequestFactoryTests(unittest.TestCase):
    def test_line_chart_uses_ssot_defaults_for_distribution(self) -> None:
        plan_chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="DTN")])

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, _DTN_BINS)
        self.assertEqual(distribution_options.lower_bound, _DTN_MIN)
        self.assertEqual(distribution_options.upper_bound, _DTN_MAX)
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
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
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

    def test_grouped_chart_uses_ssot_defaults_for_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            semantics=AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=MeasureSemanticsSpec(type="DISTRIBUTION"),
                splits=[SplitSpec(kind="STROKE_TYPE")],
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, _DTN_BINS)
        self.assertEqual(distribution_options.lower_bound, _DTN_MIN)
        self.assertEqual(distribution_options.upper_bound, _DTN_MAX)
        self.assertTrue(request.include_stats)
        self.assertIsNone(derived_axes)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_bar_chart_ignores_numeric_resolution_bucketing_for_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            numericResolution=NumericResolutionSpec.model_validate(
                {
                    "valueDomain": {"upperBound": 130},
                    "bucketing": {"bucketCount": 9},
                }
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, _DTN_BINS)
        self.assertEqual(distribution_options.lower_bound, _DTN_MIN)
        self.assertEqual(distribution_options.upper_bound, 130)
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, _DTN_MIN)
        self.assertEqual(metric_options_value.upper_boundary, 130)


if __name__ == "__main__":
    unittest.main()
