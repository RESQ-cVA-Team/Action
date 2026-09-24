from types import SimpleNamespace
import unittest

from src.executors.mapping.series_mapper import map_metrics_payload_to_series


class SeriesMapperDistributionLabelTests(unittest.TestCase):
    def test_numeric_distribution_points_include_pretty_bin_labels(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=None,
                mean=None,
                case_count=[],
                d1=SimpleNamespace(edges=[30, 40, 50], case_count=[8, 13, 14]),
                cohort_size=None,
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi], labels=None)}

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=[],
            include_metric_alias=True,
            group_by_field=None,
            add_time_period_labels=False,
        )

        self.assertEqual(len(series), 1)
        self.assertEqual([point.x for point in series[0].data], [30, 40, 50])
        self.assertEqual([point.label for point in series[0].data], ["30-40", "40-50", "50-60"])

    def test_numeric_distribution_labels_strip_float_artifacts(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=None,
                mean=None,
                case_count=[],
                d1=SimpleNamespace(edges=[0, 2, 3.9999999998], case_count=[2, 4, 1]),
                cohort_size=None,
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi], labels=None)}

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=[],
            include_metric_alias=False,
            group_by_field=None,
            add_time_period_labels=False,
        )

        self.assertEqual([point.label for point in series[0].data], ["0-2", "2-4", "4-6"])

    def test_score_metric_distribution_points_use_single_value_labels(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=None,
                mean=None,
                case_count=[],
                d1=SimpleNamespace(edges=[0, 1, 2], case_count=[5, 3, 1]),
                cohort_size=None,
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_ADMISSION_NIHSS": SimpleNamespace(kpi_group=[kpi], labels=None)}

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=[],
            include_metric_alias=False,
            group_by_field=None,
            add_time_period_labels=False,
        )

        self.assertEqual([point.label for point in series[0].data], ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()