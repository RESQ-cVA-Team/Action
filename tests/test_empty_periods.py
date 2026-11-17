import unittest
from types import SimpleNamespace

from src.executors.mapping.series_mapper import map_metrics_payload_to_series

QUARTERS = [("2026-04-01", "2026-06-30"), ("2026-07-01", "2026-09-30")]


def _period(start: str, end: str, case_count, median=None, mean=None):
    return SimpleNamespace(
        kpi1=SimpleNamespace(median=median, mean=mean, case_count=case_count, d1=None, cohort_size=None, percents=None),
        grouped_by=None,
        time_period=SimpleNamespace(start_date=start, end_date=end),
        data_origin=None,
    )


def _quarterly(value_mode: str):
    # The second quarter has no cases, as the backend returns it: caseCount [0], no statistics.
    kpis = [_period(*QUARTERS[0], case_count=[1], median=49.0, mean=49.0), _period(*QUARTERS[1], case_count=[0])]
    series = map_metrics_payload_to_series(
        metrics_payload={"metric_DTN": SimpleNamespace(kpi_group=kpis, labels=None)},
        label_parts=[],
        include_metric_alias=False,
        group_by_field=None,
        add_time_period_labels=True,
        value_mode=value_mode,
    )
    return [point.y for item in series for point in item.data]


class EmptyPeriodTests(unittest.TestCase):
    def test_median_of_a_period_without_cases_is_a_gap(self) -> None:
        self.assertEqual(_quarterly("median"), [49.0, None])

    def test_mean_of_a_period_without_cases_is_a_gap(self) -> None:
        self.assertEqual(_quarterly("mean"), [49.0, None])

    def test_case_count_chart_plots_counts_and_keeps_zero(self) -> None:
        self.assertEqual(_quarterly("count"), [1.0, 0.0])

    def test_grouped_row_without_cases_is_a_gap(self) -> None:
        kpi = _period(*QUARTERS[1], case_count=[0])
        kpi.time_period = None
        kpi.grouped_by = SimpleNamespace(group_item_name="FEMALE")
        series = map_metrics_payload_to_series(
            metrics_payload={"metric_DTN": SimpleNamespace(kpi_group=[kpi], labels=None)},
            label_parts=[],
            include_metric_alias=False,
            group_by_field="SEX",
            add_time_period_labels=False,
            value_mode="median",
        )
        self.assertIsNone(series[0].data[0].y)

    def test_grouped_row_with_cases_but_no_statistic_is_rejected(self) -> None:
        kpi = _period(*QUARTERS[1], case_count=[5])
        kpi.time_period = None
        kpi.grouped_by = SimpleNamespace(group_item_name="FEMALE")
        with self.assertRaises(ValueError):
            map_metrics_payload_to_series(
                metrics_payload={"metric_DTN": SimpleNamespace(kpi_group=[kpi], labels=None)},
                label_parts=[],
                include_metric_alias=False,
                group_by_field="SEX",
                add_time_period_labels=False,
                value_mode="median",
            )


class EmptyCohortShareTests(unittest.TestCase):
    def test_share_of_a_period_without_cases_is_a_gap(self) -> None:
        def aa(start, end, count, cohort, percents):
            kpi = _period(start, end, case_count=[count])
            kpi.kpi1.cohort_size = cohort
            kpi.kpi1.percents = percents
            return kpi

        kpis = [aa(*QUARTERS[0], 1, 4, [25.0]), aa(*QUARTERS[1], 0, 0, [0.0])]
        series = map_metrics_payload_to_series(
            metrics_payload={"metric_AA_DTN_LE60": SimpleNamespace(kpi_group=kpis, labels=None)},
            label_parts=[],
            include_metric_alias=False,
            group_by_field=None,
            add_time_period_labels=True,
            value_mode="percent",
        )
        self.assertEqual([point.y for item in series for point in item.data], [25.0, None])

    def test_category_share_of_an_empty_group_is_a_gap(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(median=None, mean=None, case_count=[0, 0], d1=None, cohort_size=0, percents=[0.0, 0.0]),
            grouped_by=SimpleNamespace(group_item_name="FEMALE"),
            time_period=None,
            data_origin=None,
        )
        series = map_metrics_payload_to_series(
            metrics_payload={"metric_ARRIVAL_MODE": SimpleNamespace(kpi_group=[kpi], labels=["ems", "private"])},
            label_parts=[],
            include_metric_alias=False,
            group_by_field="SEX",
            add_time_period_labels=False,
            value_mode="percent",
        )
        self.assertEqual([point.y for item in series for point in item.data], [None, None])


if __name__ == "__main__":
    unittest.main()
