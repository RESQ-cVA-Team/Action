import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.langchain.schema import AnalysisSemanticsSpec, ChartSpec, MeasureSemanticsSpec, MetricSpec
from src.executors.mapping.chart_builder import build_chart_dto
from src.executors.mapping.series_mapper import map_metrics_payload_to_series, merge_series_by_name
from src.executors.orchestration.plan_executor import _chart_value_mode
from src.planners.langchain.request_orchestrator import _decision_stage, _infer_stroke_type_share_metric


def _kpi(**kpi1):
    base = {"median": None, "mean": None, "case_count": [], "d1": None, "cohort_size": None, "percents": None}
    base.update(kpi1)
    return SimpleNamespace(kpi1=SimpleNamespace(**base), grouped_by=None, time_period=None, data_origin=None)


def _map(payload, value_mode, **kwargs):
    args = {"label_parts": [], "include_metric_alias": False, "group_by_field": None, "add_time_period_labels": False}
    args.update(kwargs)
    return map_metrics_payload_to_series(metrics_payload=payload, value_mode=value_mode, **args)


class PercentSeriesMappingTests(unittest.TestCase):
    def test_categorical_metric_uses_backend_percents(self) -> None:
        payload = {"metric_ARRIVAL_MODE": SimpleNamespace(kpi_group=[_kpi(case_count=[30, 70], percents=[30.0, 70.0])], labels=["ems", "private"])}
        self.assertEqual([p.y for p in _map(payload, "count")[0].data], [30.0, 70.0])
        self.assertEqual([p.y for p in _map(payload, "percent")[0].data], [30.0, 70.0])

    def test_categorical_metric_derives_percents_from_counts_when_backend_omits_them(self) -> None:
        payload = {"metric_ARRIVAL_MODE": SimpleNamespace(kpi_group=[_kpi(case_count=[1, 3])], labels=["ems", "private"])}
        self.assertEqual([p.y for p in _map(payload, "percent")[0].data], [25.0, 75.0])

    def test_distribution_uses_bin_percents(self) -> None:
        d1 = SimpleNamespace(edges=[30, 40, 50], case_count=[8, 12, 20], percents=[20.0, 30.0, 50.0])
        payload = {"metric_DTN": SimpleNamespace(kpi_group=[_kpi(d1=d1)], labels=None)}
        self.assertEqual([p.y for p in _map(payload, "count")[0].data], [8.0, 12.0, 20.0])
        self.assertEqual([p.y for p in _map(payload, "percent")[0].data], [20.0, 30.0, 50.0])

    def test_grouped_numeric_row_plots_the_planned_statistic(self) -> None:
        kpi = _kpi(median=42.0, mean=51.5, case_count=[120])
        kpi.time_period = SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31")
        payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi], labels=None)}
        self.assertEqual(_map(payload, "count", add_time_period_labels=True)[0].data[0].y, 42.0)
        self.assertEqual(_map(payload, "mean", add_time_period_labels=True)[0].data[0].y, 51.5)

    def test_grouped_row_uses_first_backend_percent(self) -> None:
        # One quarter of an Angels Awards metric: the first percent is the "yes" share.
        kpi = _kpi(case_count=[80, 20], percents=[80.0, 20.0], cohort_size=100)
        kpi.time_period = SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31")
        payload = {"metric_AA_DTN_LE60": SimpleNamespace(kpi_group=[kpi], labels=["yes", "no"])}
        counts = _map(payload, "count", add_time_period_labels=True)
        shares = _map(payload, "percent", add_time_period_labels=True)
        self.assertEqual(counts[0].data[0].y, 80.0)
        self.assertEqual(shares[0].data[0].y, 80.0)

    def test_aa_default_selects_yes_by_label_not_category_order(self) -> None:
        payload = {
            "metric_AA_DTN_LE60": SimpleNamespace(
                kpi_group=[_kpi(case_count=[20, 80], percents=[20.0, 80.0], cohort_size=100)],
                labels=["no", "yes"],
            )
        }

        percent_points = _map(payload, "percent")[0].data
        count_points = _map(payload, "count")[0].data

        self.assertEqual([(point.x, point.y) for point in percent_points], [("yes", 80.0)])
        self.assertEqual([(point.x, point.y) for point in count_points], [("yes", 80.0)])

    def test_aa_category_detail_includes_yes_and_no(self) -> None:
        payload = {
            "metric_AA_DTN_LE60": SimpleNamespace(
                kpi_group=[_kpi(case_count=[80, 20], percents=[80.0, 20.0], cohort_size=100)],
                labels=["yes", "no"],
            )
        }

        series = _map(payload, "percent", category_detail_metric_codes=["AA_DTN_LE60"])

        self.assertEqual([(point.x, point.y) for point in series[0].data], [("yes", 80.0), ("no", 20.0)])

    def test_aa_grouped_category_detail_creates_yes_and_no_series(self) -> None:
        first = _kpi(case_count=[80, 20], percents=[80.0, 20.0], cohort_size=100)
        first.time_period = SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31")
        second = _kpi(case_count=[60, 40], percents=[60.0, 40.0], cohort_size=100)
        second.time_period = SimpleNamespace(start_date="2026-04-01", end_date="2026-06-30")
        payload = {"metric_AA_DTN_LE60": SimpleNamespace(kpi_group=[first, second], labels=["yes", "no"])}

        series = _map(
            payload,
            "percent",
            add_time_period_labels=True,
            category_detail_metric_codes=["AA_DTN_LE60"],
        )

        self.assertEqual(len(series), 4)
        yes_series = [item for item in series if item.name.endswith("— yes")]
        no_series = [item for item in series if item.name.endswith("— no")]
        self.assertEqual([point.y for item in yes_series for point in item.data], [80.0, 60.0])
        self.assertEqual([point.y for item in no_series for point in item.data], [20.0, 40.0])

    def test_aa_grouped_empty_time_bucket_maps_missing_categories(self) -> None:
        populated = _kpi(case_count=[80, 20], percents=[80.0, 20.0], cohort_size=100)
        populated.time_period = SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31")
        empty = _kpi(case_count=[])
        empty.time_period = SimpleNamespace(start_date="2026-04-01", end_date="2026-06-30")
        payload = {
            "metric_AA_DTN_LE60": SimpleNamespace(
                kpi_group=[populated, empty],
                labels=["yes", "no"],
            )
        }

        # The executor merges the per-quarter series by name, as it does for every request.
        series = merge_series_by_name(_map(payload, "percent", add_time_period_labels=True))

        self.assertEqual([(point.x, point.y) for point in series[0].data], [("2026-Q1", 80.0), ("2026-Q2", None)])

    def test_grouped_row_falls_back_to_count_over_cohort(self) -> None:
        kpi = _kpi(case_count=[25], cohort_size=200)
        kpi.time_period = SimpleNamespace(start_date="2026-01-01", end_date="2026-03-31")
        payload = {"metric_AA_DTN_LE60": SimpleNamespace(kpi_group=[kpi], labels=None)}
        self.assertEqual(_map(payload, "percent", add_time_period_labels=True)[0].data[0].y, 12.5)


class PercentAxisAndModeTests(unittest.TestCase):
    def _chart(self, measure: str) -> ChartSpec:
        return ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="ARRIVAL_MODE")],
            semantics=AnalysisSemanticsSpec(intent="DISTRIBUTION", measure=MeasureSemanticsSpec(type=measure)),
        )

    def test_value_mode_follows_the_measure(self) -> None:
        self.assertEqual(_chart_value_mode(self._chart("RATE")), "percent")
        self.assertEqual(_chart_value_mode(self._chart("DISTRIBUTION")), "count")
        self.assertEqual(_chart_value_mode(self._chart("MEAN")), "mean")
        self.assertEqual(_chart_value_mode(self._chart("MEDIAN")), "count")
        self.assertEqual(_chart_value_mode(ChartSpec(chart_type="BAR", metrics=[MetricSpec(metric="DTN")])), "count")
        self.assertEqual(_chart_value_mode(self._chart_for_aa("RATE")), "percent")
        self.assertEqual(_chart_value_mode(self._chart_for_aa("COUNT")), "count")

    def _chart_for_aa(self, measure: str) -> ChartSpec:
        return ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="AA_DTN_LE60")],
            semantics=AnalysisSemanticsSpec(intent="DISTRIBUTION", measure=MeasureSemanticsSpec(type=measure)),
        )

    def test_aa_axis_follows_the_measure(self) -> None:
        series = [ChartSeries(name="AA", data=[ChartPoint(x="yes", y=80.0)])]

        rate = build_chart_dto(plan_chart=self._chart_for_aa("RATE"), dimensions=[], series=series, derived_axes=None)
        count = build_chart_dto(plan_chart=self._chart_for_aa("COUNT"), dimensions=[], series=series, derived_axes=None)

        self.assertEqual(rate.metadata.y_axis.label, "Share of cases (%)")
        self.assertEqual(count.metadata.y_axis.label, "Cases")

    def test_enum_counts_are_labelled_as_cases(self) -> None:
        series = [ChartSeries(name="arrival mode", data=[ChartPoint(x="ems", y=30.0)])]
        counts = build_chart_dto(plan_chart=self._chart("DISTRIBUTION"), dimensions=[], series=series, derived_axes=None)
        self.assertEqual(counts.metadata.y_axis.label, "Cases")

    def test_numeric_value_axis_names_the_plotted_statistic(self) -> None:
        series = [ChartSeries(name="DTN", data=[ChartPoint(x="2026-Q1", y=42.0)])]
        labels = {}
        for measure in ("MEDIAN", "MEAN"):
            chart = ChartSpec(
                chart_type="LINE",
                metrics=[MetricSpec(metric="DTN")],
                semantics=AnalysisSemanticsSpec(intent="TREND", measure=MeasureSemanticsSpec(type=measure)),
            )
            labels[measure] = build_chart_dto(plan_chart=chart, dimensions=[], series=series, derived_axes=None).metadata.y_axis.label
        self.assertTrue(labels["MEDIAN"].startswith("Median "), labels)
        self.assertTrue(labels["MEAN"].startswith("Mean "), labels)

    def test_percent_chart_labels_its_y_axis_as_a_share(self) -> None:
        series = [ChartSeries(name="arrival mode", data=[ChartPoint(x="ems", y=30.0), ChartPoint(x="private", y=70.0)])]
        percent = build_chart_dto(plan_chart=self._chart("RATE"), dimensions=[], series=series, derived_axes=None)
        counts = build_chart_dto(plan_chart=self._chart("DISTRIBUTION"), dimensions=[], series=series, derived_axes=None)
        self.assertEqual(percent.metadata.y_axis.label, "Share of cases (%)")
        self.assertNotEqual(counts.metadata.y_axis.label, "Share of cases (%)")


class BareMetricDefaultChartTests(unittest.TestCase):
    def test_missing_chart_type_alone_proceeds_to_a_default_chart(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "clarify", "reason": "missing_required_fields", "missing_fields": ["chart_type"], "message": "Which chart type?"},
        ):
            outcome = _decision_stage(question="What is my door to needle time?", entities={"metric": "DTN"}, language="en")
        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "all_required_fields_present")

    def test_missing_metric_still_asks(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "clarify", "reason": "missing_required_fields", "missing_fields": ["metric"], "message": "Which metric?"},
        ):
            outcome = _decision_stage(question="Show me a line chart", entities={"chart_type": "LINE"}, language="en")
        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["metric"])

    def test_statistical_test_requests_never_needed_a_chart_type(self) -> None:
        # chart_type is optional for statistical tests already; the new
        # safeguard leaves that path alone and the existing one still proceeds.
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "clarify", "reason": "missing_required_fields", "missing_fields": ["chart_type"], "message": "Which chart type?"},
        ):
            outcome = _decision_stage(question="Run a mann-whitney test on dtn", entities={"metric": "DTN", "statistical_test_type": "MANN_WHITNEY"}, language="en")
        self.assertEqual(outcome.decision, "proceed")


class StrokeShareMetricInferenceTests(unittest.TestCase):
    def test_percentage_of_a_stroke_type_becomes_the_stroke_type_metric(self) -> None:
        out = _infer_stroke_type_share_metric("Make a line chart of percentage of ischemic strokes per quarter", {"chart_type": "LINE", "stroke_type": "ISCHEMIC", "group_by": "QUARTER"})
        self.assertEqual(out["metric"], "STROKE_TYPE")

    def test_a_present_metric_is_never_overridden(self) -> None:
        entities = {"metric": "DTN", "stroke_type": "ISCHEMIC"}
        self.assertEqual(_infer_stroke_type_share_metric("percentage of ischemic strokes with dtn", entities), entities)

    def test_a_plain_stroke_type_filter_is_left_alone(self) -> None:
        entities = {"stroke_type": "ISCHEMIC", "chart_type": "BAR"}
        self.assertEqual(_infer_stroke_type_share_metric("show ischemic strokes in a bar chart", entities), entities)


if __name__ == "__main__":
    unittest.main()
