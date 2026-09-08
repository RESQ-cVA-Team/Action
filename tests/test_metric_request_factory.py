import unittest
from typing import Any, cast

from src.domain.graphql.request import DistributionOptions
from src.domain.langchain.schema import AnalysisSemanticsSpec, ChartSpec, MeasureSemanticsSpec, MetricSpec, NumericResolutionSpec, SplitSpec
from src.executors.planning.metric_request_factory import (
    HistogramBinningStats,
    build_metric_requests,
    resolve_adaptive_histogram_bin_count,
    resolve_freedman_diaconis_bin_count,
    should_defer_histogram_binning,
)
from src.executors.planning.ssot_metric_defaults import get_distribution_defaults

# DTN SSOT defaults used as baseline for assertions. Tests that apply explicit
# numericResolution overrides assert on the override values, not the SSOT baseline.
_DTN_BINS, _DTN_MIN, _DTN_MAX = get_distribution_defaults("DTN")


class MetricRequestFactoryTests(unittest.TestCase):
    def test_histogram_requires_explicit_or_precomputed_bins_for_distribution(self) -> None:
        plan_chart = ChartSpec(chart_type="HISTOGRAM", metrics=[MetricSpec(metric="DTN")])

        with self.assertRaisesRegex(ValueError, "must be explicitly specified or precomputed"):
            build_metric_requests(plan_chart=plan_chart)

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
        self.assertEqual(distribution_options.bin_count, 9)
        self.assertEqual(distribution_options.lower_bound, _DTN_MIN)
        self.assertEqual(distribution_options.upper_bound, 130)
        self.assertIsNotNone(request.metric_options)
        metric_options = request.metric_options
        self.assertIsNotNone(metric_options)
        metric_options_value = cast(Any, metric_options)
        self.assertEqual(metric_options_value.lower_boundary, _DTN_MIN)
        self.assertEqual(metric_options_value.upper_boundary, 130)

    def test_ungrouped_bar_chart_requires_explicit_or_precomputed_bins_for_distribution(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        with self.assertRaisesRegex(ValueError, "must be explicitly specified or precomputed"):
            build_metric_requests(plan_chart=plan_chart)

    def test_histogram_without_distribution_keeps_bounds_for_stats_preflight(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        metric_requests, derived_axes, _, _ = build_metric_requests(
            plan_chart=plan_chart,
            include_distribution=False,
        )

        request = metric_requests[0]
        self.assertFalse(request.include_distribution)
        self.assertIsNone(request.distribution_options)
        self.assertTrue(request.include_stats)
        self.assertIsNone(derived_axes)
        self.assertIsNotNone(request.metric_options)
        metric_options = cast(Any, request.metric_options)
        self.assertEqual(metric_options.lower_boundary, 0)
        self.assertEqual(metric_options.upper_boundary, 100)

    def test_should_defer_histogram_binning_for_single_numeric_histogram_without_bucketing(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        self.assertTrue(should_defer_histogram_binning(plan_chart))

    def test_should_not_defer_histogram_binning_when_bucket_count_is_explicit(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate(
                {
                    "valueDomain": {"lowerBound": 0, "upperBound": 100},
                    "bucketing": {"bucketCount": 8},
                }
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        self.assertFalse(should_defer_histogram_binning(plan_chart))

    def test_should_defer_histogram_binning_for_single_numeric_ungrouped_bar_without_bucketing(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        self.assertTrue(should_defer_histogram_binning(plan_chart))

    def test_should_not_defer_histogram_binning_for_grouped_bar_chart(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            semantics=AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=MeasureSemanticsSpec(type="DISTRIBUTION"),
                splits=[SplitSpec(kind="STROKE_TYPE")],
            ),
            metrics=[MetricSpec(metric="DTN")],
        )

        self.assertFalse(should_defer_histogram_binning(plan_chart))

    def test_resolve_freedman_diaconis_bin_count_uses_iqr_and_chart_bounds(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        resolved_bins = resolve_freedman_diaconis_bin_count(
            plan_chart=plan_chart,
            metric_code="DTN",
            stats=HistogramBinningStats(cohort_size=64, interquartile_range=16.0),
        )

        self.assertEqual(resolved_bins, 13)

    def test_resolve_freedman_diaconis_bin_count_returns_raw_unclamped_result(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        resolved_bins = resolve_freedman_diaconis_bin_count(
            plan_chart=plan_chart,
            metric_code="DTN",
            stats=HistogramBinningStats(cohort_size=64, interquartile_range=60.0),
        )

        self.assertEqual(resolved_bins, 4)

    def test_resolve_adaptive_histogram_bin_count_falls_back_to_rice_when_iqr_is_unavailable(self) -> None:
        plan_chart = ChartSpec(
            chart_type="HISTOGRAM",
            numericResolution=NumericResolutionSpec.model_validate({"valueDomain": {"lowerBound": 0, "upperBound": 100}}),
            metrics=[MetricSpec(metric="DTN")],
        )

        resolved_bins = resolve_adaptive_histogram_bin_count(
            plan_chart=plan_chart,
            metric_code="DTN",
            stats=HistogramBinningStats(cohort_size=64, interquartile_range=None),
        )

        self.assertEqual(resolved_bins, 8)


if __name__ == "__main__":
    unittest.main()
