import unittest

from src.executors.planning.ssot_metric_defaults import (
    format_distribution_bin_label,
    format_distribution_point_label,
    get_distribution_defaults,
    resolve_implicit_distribution_layout,
    resolve_minutes_distribution_layout,
    resolve_pretty_distribution_bins,
    resolve_score_distribution_layout,
)


class PrettyDistributionBinsTests(unittest.TestCase):
    def test_pretty_bins_choose_clean_divisor_close_to_target(self) -> None:
        bins = resolve_pretty_distribution_bins(31, 89, target_bins=6)

        self.assertEqual(bins.lower_bound, 30.0)
        self.assertEqual(bins.upper_bound, 90.0)
        self.assertEqual(bins.bin_width, 10.0)
        self.assertEqual(bins.bin_count, 6)

    def test_pretty_bins_allow_clean_decimal_steps(self) -> None:
        bins = resolve_pretty_distribution_bins(0, 15, target_bins=6)

        self.assertEqual(bins.bin_width, 2.5)
        self.assertEqual(bins.bin_count, 6)
        self.assertEqual(format_distribution_bin_label(0.0, 2.5), "0-2.5")
        self.assertEqual(format_distribution_bin_label(2.5, 5.0), "2.5-5")

    def test_pretty_bins_enforce_resolution_floor(self) -> None:
        bins = resolve_pretty_distribution_bins(0, 120, target_bins=2, minimum_bins=6, maximum_bins=16)

        self.assertGreaterEqual(bins.bin_count, 6)
        self.assertEqual(bins.bin_width, 20.0)
        self.assertEqual(bins.bin_count, 6)

    def test_distribution_bin_label_strips_floating_artifacts(self) -> None:
        self.assertEqual(format_distribution_bin_label(2.0, 3.9999999998), "2-4")

    def test_implicit_layout_targets_higher_resolution_with_nice_width(self) -> None:
        bins = resolve_implicit_distribution_layout(0, 520)

        self.assertEqual(bins.lower_bound, 0.0)
        self.assertEqual(bins.upper_bound, 525.0)
        self.assertEqual(bins.bin_width, 25.0)
        self.assertEqual(bins.bin_count, 21)

    def test_score_layout_uses_one_bin_per_score(self) -> None:
        bins = resolve_score_distribution_layout(0, 6)

        self.assertEqual(bins.lower_bound, 0.0)
        self.assertEqual(bins.upper_bound, 7.0)
        self.assertEqual(bins.bin_width, 1.0)
        self.assertEqual(bins.bin_count, 7)

    def test_score_metric_distribution_labels_use_single_values(self) -> None:
        self.assertEqual(format_distribution_point_label("ADMISSION_NIHSS", 0.0, 1.0), "0")
        self.assertEqual(format_distribution_point_label("ADMISSION_NIHSS", 6.0, 7.0), "6")
        self.assertEqual(format_distribution_point_label("DTN", 0.0, 25.0), "0-25")

    def test_minutes_layout_uses_fixed_width_five(self) -> None:
        bins = resolve_minutes_distribution_layout(12, 131)

        self.assertEqual(bins.lower_bound, 10.0)
        self.assertEqual(bins.upper_bound, 135.0)
        self.assertEqual(bins.bin_width, 5.0)
        self.assertEqual(bins.bin_count, 25)

    def test_minutes_metric_defaults_round_upper_bound_up_to_five(self) -> None:
        _, lower, upper = get_distribution_defaults("DTN")

        self.assertEqual(lower, 0)
        self.assertEqual(upper % 5, 0)


if __name__ == "__main__":
    unittest.main()