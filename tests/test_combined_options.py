import unittest
from types import SimpleNamespace

from src.domain.langchain.schema import MetricSpec
from src.executors.mapping.series_mapper import map_metrics_payload_to_series
from src.executors.planning.request_plan import CategoryView

LABELS = ["CT", "CT CTA", "CT CTA perfusion", "MR DWI/FLAIR", "MR DWI/FLAIR MRA", "MR DWI/FLAIR MRA perfusion"]
COUNTS = [10, 5, 3, 20, 2, 0]
CT = CategoryView(combined=("ct", "ct_cta", "ct_cta_perfusion"))


def _kpi(time_period=None):
    return SimpleNamespace(
        kpi1=SimpleNamespace(median=None, mean=None, case_count=list(COUNTS), d1=None, cohort_size=40, percents=None),
        grouped_by=None,
        time_period=time_period,
        data_origin=None,
    )


def _map(kpi, value_mode, **kwargs):
    return map_metrics_payload_to_series(
        metrics_payload={"metric_IMAGING_TYPE": SimpleNamespace(kpi_group=[kpi], labels=list(LABELS))},
        label_parts=[],
        include_metric_alias=False,
        group_by_field=None,
        value_mode=value_mode,
        **kwargs,
    )


class CombinedOptionsTests(unittest.TestCase):
    def test_a_share_over_time_counts_the_options_together_out_of_all_cases(self) -> None:
        kpi = _kpi(SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31"))
        series = _map(kpi, "percent", add_time_period_labels=True, category_views={"IMAGING_TYPE": CT})
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].name, "brain imaging type — ct, ct with cta, ct with cta and perfusion")
        self.assertEqual(series[0].data[0].y, 45.0)

    def test_without_time_the_combined_options_stand_against_the_rest(self) -> None:
        series = _map(_kpi(), "count", add_time_period_labels=False, category_views={"IMAGING_TYPE": CT})
        self.assertEqual([(point.x, point.y) for point in series[0].data], [("ct, ct with cta, ct with cta and perfusion", 18.0), ("other", 22.0)])

    def test_an_option_missing_from_the_payload_is_an_error(self) -> None:
        view = CategoryView(combined=("ct", "pet"))
        with self.assertRaises(ValueError):
            _map(_kpi(), "count", add_time_period_labels=False, category_views={"IMAGING_TYPE": view})


class CategoriesSpecTests(unittest.TestCase):
    def test_option_keys_are_checked_and_normalised(self) -> None:
        self.assertEqual(MetricSpec(metric="IMAGING_TYPE", categories=["CT", "ct_cta", "ct"]).categories, ["ct", "ct_cta"])
        with self.assertRaisesRegex(ValueError, "no option"):
            MetricSpec(metric="IMAGING_TYPE", categories=["pet"])
        with self.assertRaisesRegex(ValueError, "has no options"):
            MetricSpec(metric="DTN", categories=["ct"])
        with self.assertRaisesRegex(ValueError, "every option"):
            MetricSpec(metric="SEX", categories=["male", "female", "unknown_other"])
        with self.assertRaisesRegex(ValueError, "not both"):
            MetricSpec(metric="AA_STROKE_UNIT", categories=["yes"], include_category_detail=True)


if __name__ == "__main__":
    unittest.main()
