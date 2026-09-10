import unittest
from typing import Any, cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DistributionOptions
from src.domain.langchain.schema import AnalysisSemanticsSpec, ChartSpec, MeasureSemanticsSpec, MetricSpec, NumericResolutionSpec, SplitSpec
from src.executors.planning.metric_request_factory import build_metric_requests
from src.executors.planning.ssot_metric_defaults import get_distribution_defaults, resolve_minutes_distribution_layout, resolve_score_distribution_layout

# DTN SSOT defaults used as baseline for assertions. Tests that apply explicit
# numericResolution overrides assert on the override values, not the SSOT baseline.
_DTN_BINS, _DTN_MIN, _DTN_MAX = get_distribution_defaults("DTN")
_DTN_PRETTY_LAYOUT = resolve_minutes_distribution_layout(_DTN_MIN, _DTN_MAX)


class MetricRequestFactoryTests(unittest.TestCase):
    def test_histogram_uses_ssot_defaults_for_distribution(self) -> None:
        plan_chart = ChartSpec(chart_type="HISTOGRAM", metrics=[MetricSpec(metric="DTN")])

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, _DTN_PRETTY_LAYOUT.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(_DTN_PRETTY_LAYOUT.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(_DTN_PRETTY_LAYOUT.upper_bound))
        self.assertTrue(request.include_stats)
        self.assertIsNotNone(derived_axes)
        derived_axes_value = cast(tuple[ChartAxis, ChartAxis], derived_axes)
        self.assertIsInstance(derived_axes_value[0], ChartAxis)
        self.assertIsInstance(derived_axes_value[1], ChartAxis)
        self.assertEqual(metric_data_origins, [None])
        self.assertEqual(metric_scope_labels, [None])

    def test_line_chart_uses_ssot_defaults_for_distribution(self) -> None:
        plan_chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="DTN")])

        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(
            plan_chart=plan_chart,
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, _DTN_PRETTY_LAYOUT.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(_DTN_PRETTY_LAYOUT.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(_DTN_PRETTY_LAYOUT.upper_bound))
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
        expected_layout = resolve_minutes_distribution_layout(10, 180)
        self.assertEqual(distribution_options.bin_count, expected_layout.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(expected_layout.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(expected_layout.upper_bound))
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, int(expected_layout.lower_bound))
        self.assertEqual(metric_options_value.upper_boundary, int(expected_layout.upper_bound))
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
        self.assertEqual(distribution_options.bin_count, _DTN_PRETTY_LAYOUT.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(_DTN_PRETTY_LAYOUT.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(_DTN_PRETTY_LAYOUT.upper_bound))
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
        )

        self.assertEqual(len(metric_requests), 1)
        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 8)
        self.assertEqual(distribution_options.lower_bound, 25)
        self.assertEqual(distribution_options.upper_bound, _DTN_MAX)  # upper not overridden
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, 25)
        self.assertEqual(metric_options_value.upper_boundary, _DTN_MAX)
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

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 10)

    def test_bar_chart_merges_partial_value_domain_override_in_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"upperBound": 130}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        self.assertTrue(request.include_distribution)
        distribution_options = cast(DistributionOptions, request.distribution_options)
        expected_layout = resolve_minutes_distribution_layout(_DTN_MIN, 130)
        self.assertEqual(distribution_options.bin_count, expected_layout.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(expected_layout.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(expected_layout.upper_bound))
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, int(expected_layout.lower_bound))
        self.assertEqual(metric_options_value.upper_boundary, int(expected_layout.upper_bound))

    def test_explicit_bucket_count_override_still_wins_over_pretty_defaults(self) -> None:
        plan_chart = ChartSpec(
            chart_type="LINE",
            numericResolution=NumericResolutionSpec.model_validate({"bucketing": {"bucketCount": 9}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        self.assertEqual(distribution_options.bin_count, 9)

    def test_score_metrics_use_one_bin_per_score_by_default(self) -> None:
        plan_chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="ADMISSION_NIHSS")])

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        expected_layout = resolve_score_distribution_layout(0, 42)
        self.assertEqual(distribution_options.bin_count, expected_layout.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(expected_layout.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(expected_layout.upper_bound))

    def test_minutes_metrics_use_fixed_width_five_by_default(self) -> None:
        plan_chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="DTN")])

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        expected_layout = resolve_minutes_distribution_layout(_DTN_MIN, _DTN_MAX)
        self.assertEqual(distribution_options.bin_count, expected_layout.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(expected_layout.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(expected_layout.upper_bound))

    def test_minutes_value_domain_is_snapped_to_width_five_without_explicit_bucketing(self) -> None:
        plan_chart = ChartSpec(
            chart_type="LINE",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 12, "upperBound": 131}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, _, _, _ = build_metric_requests(plan_chart=plan_chart)

        request = metric_requests[0]
        distribution_options = cast(DistributionOptions, request.distribution_options)
        expected_layout = resolve_minutes_distribution_layout(12, 131)
        self.assertEqual(distribution_options.bin_count, expected_layout.bin_count)
        self.assertEqual(distribution_options.lower_bound, int(expected_layout.lower_bound))
        self.assertEqual(distribution_options.upper_bound, int(expected_layout.upper_bound))


if __name__ == "__main__":
    unittest.main()
