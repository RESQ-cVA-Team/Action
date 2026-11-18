import asyncio
import unittest
from datetime import date
from typing import Any, Dict
from unittest.mock import patch

from src.domain.dto.charts.types import ChartMetric
from src.domain.dto.value import ValueResult
from src.domain.graphql.response import MetricsQueryResponse
from src.domain.langchain.schema import AnalysisPlan, ValueSpec
from src.executors.orchestration import plan_executor
from src.executors.planning.value_plan import value_chart, value_period
from src.planners.langchain import value_answer
from src.planners.langchain.pipeline import _assert_value_dates_are_stated, _plain_plan_mismatch


_MY_HOSPITAL = {"providerId": [279]}


def _value(**fields: Any) -> ValueSpec:
    return ValueSpec.model_validate({"metric": "DTN", "measure": {"type": "MEDIAN"}, "dataOrigin": _MY_HOSPITAL, **fields})


def _response(metric: str, kpi1: Dict[str, Any], labels: Any = None) -> MetricsQueryResponse:
    group = {"kpi1": kpi1, "timePeriod": {"startDate": "2025-01-01", "endDate": "2025-12-31"}}
    return MetricsQueryResponse.model_validate({"data": {"getMetrics": {f"metric_{metric}": {"labels": labels, "kpiGroup": [group]}}}})


def _execute(value: ValueSpec, response: MetricsQueryResponse) -> ValueResult:
    plan = AnalysisPlan(values=[value])
    with (
        patch.object(plan_executor, "resolve_plan_metric_origins", side_effect=lambda plan, **_: plan),
        patch.object(plan_executor.client, "query", return_value=response),
    ):
        result = asyncio.run(plan_executor.execute_plan_async(plan, user_sub="user", trace_id="trace"))
    assert not result.charts
    (answer,) = result.values
    return answer


class ValueSpecTests(unittest.TestCase):
    def test_measures_follow_the_data_type(self) -> None:
        self.assertEqual(_value().measure.type, "MEDIAN")
        with self.assertRaisesRegex(ValueError, "measured as one of"):
            _value(measure={"type": "RATE"})
        with self.assertRaisesRegex(ValueError, "measured as one of"):
            ValueSpec.model_validate({"metric": "THROMBOLYSIS", "measure": {"type": "MEDIAN"}})

    def test_an_enum_value_names_its_options_unless_it_is_yes_no(self) -> None:
        ValueSpec.model_validate({"metric": "THROMBOLYSIS", "measure": {"type": "COUNT"}})
        with self.assertRaisesRegex(ValueError, "categories"):
            ValueSpec.model_validate({"metric": "IMAGING_TYPE", "measure": {"type": "COUNT"}})
        ValueSpec.model_validate({"metric": "IMAGING_TYPE", "categories": ["ct"], "measure": {"type": "COUNT"}})

    def test_dates_go_in_the_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "period"):
            _value(filters={"type": "DateFilter", "operator": "GE", "value": "2025-01-01"})

    def test_a_plan_has_values_or_charts(self) -> None:
        chart = {"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}}}
        with self.assertRaisesRegex(ValueError, "either with values"):
            AnalysisPlan.model_validate({"values": [{"metric": "DTN", "measure": {"type": "MEDIAN"}}], "charts": [chart]})


class ValuePeriodTests(unittest.TestCase):
    def test_a_range_is_kept_and_a_window_ends_today(self) -> None:
        self.assertEqual(value_period(_value(period={"start_date": "2025-01-01", "end_date": "2025-12-31"})), ("2025-01-01", "2025-12-31"))
        today = date.today()
        last_year = value_period(_value(period={"last_n": 1, "unit": "YEAR"}))
        self.assertEqual(last_year, value_period(_value(period={"last_n": 12, "unit": "MONTH"})))
        start, end = last_year
        self.assertEqual(end, today.isoformat())
        first = date.fromisoformat(start)
        self.assertEqual(first.day, 1)
        self.assertEqual((today.year - first.year) * 12 + today.month - first.month, 11)
        start, end = value_period(_value(period={"last_n": 2, "unit": "WEEK"}))
        self.assertEqual((date.fromisoformat(end) - date.fromisoformat(start)).days, 13)

    def test_a_value_is_a_chart_over_its_one_period(self) -> None:
        chart = value_chart(_value(period={"start_date": "2026-01-01", "end_date": "2026-03-31"}))
        self.assertEqual([period.model_dump() for period in chart.semantics.time.periods], [{"start_date": "2026-01-01", "end_date": "2026-03-31"}])
        self.assertEqual(chart.metrics[0].metric, "DTN")


class ValueExecutionTests(unittest.TestCase):
    def test_a_median_comes_with_the_patients_it_is_based_on(self) -> None:
        result = _execute(
            _value(period={"start_date": "2025-01-01", "end_date": "2025-12-31"}),
            _response("DTN", {"caseCount": [39], "cohortSize": 39, "percents": [100.0], "median": 161.0, "mean": 152.3}),
        )
        self.assertEqual((result.value, result.patients, result.statistic), (161.0, 39, "MEDIAN"))
        self.assertEqual((result.start_date, result.end_date), ("2025-01-01", "2025-12-31"))

    def test_a_count_of_chosen_options_is_out_of_the_cohort(self) -> None:
        labels = ["CT", "CT CTA", "CT CTA perfusion", "MR DWI/FLAIR", "MR DWI/FLAIR MRA", "MR DWI/FLAIR MRA perfusion"]
        value = ValueSpec.model_validate(
            {
                "metric": "IMAGING_TYPE",
                "categories": ["ct", "ct_cta", "ct_cta_perfusion"],
                "measure": {"type": "COUNT"},
                "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
                "dataOrigin": _MY_HOSPITAL,
            }
        )
        result = _execute(value, _response("IMAGING_TYPE", {"caseCount": [35, 37, 48, 14, 57, 24], "cohortSize": 219}, labels))
        self.assertEqual((result.value, result.patients), (120.0, 219))
        self.assertEqual(result.metric, ChartMetric(metric="IMAGING_TYPE", categories=["ct", "ct_cta", "ct_cta_perfusion"]))


class ValueAnswerTests(unittest.TestCase):
    def _result(self, **fields: Any) -> ValueResult:
        base: Dict[str, Any] = {"metric": ChartMetric(metric="DTN"), "statistic": "MEDIAN", "value": 161.0, "patients": 39, "start_date": "2025-01-01", "end_date": "2025-12-31", "scope": "My Hospital"}
        return ValueResult(**{**base, **fields})

    def test_numbers_and_periods_are_written_the_same_way_every_time(self) -> None:
        self.assertEqual(value_answer.format_number(29.37, "RATE", "cs"), "29,4")
        self.assertEqual(value_answer.format_number(161.0, "MEDIAN", "en"), "161")
        self.assertEqual(value_answer.format_number(41.6, "COUNT", "en"), "42")
        self.assertEqual(value_answer.period_label("2025-01-01", "2025-12-31"), "2025")
        self.assertEqual(value_answer.period_label("2026-01-01", "2026-03-31"), "2026-Q1")
        self.assertEqual(value_answer.period_label("2025-12-01", "2026-11-17"), "2025-12-01 – 2026-11-17")

    def test_a_sentence_keeps_the_value_and_adds_no_number(self) -> None:
        facts = value_answer.value_facts(self._result(), "en")
        self.assertTrue(value_answer.sentence_keeps_the_numbers("Your median door-to-needle time in 2025 was 161 minutes, based on 39 patients.", facts))
        self.assertFalse(value_answer.sentence_keeps_the_numbers("Your median door-to-needle time was about 160 minutes.", facts))
        self.assertFalse(value_answer.sentence_keeps_the_numbers("161 minutes in 2025, for 39 of 222 patients.", facts))

    def test_a_changed_number_sends_the_fixed_sentence(self) -> None:
        with patch.object(value_answer, "_write_now", return_value=None):
            (sentence,) = asyncio.run(value_answer.answer_values([self._result()], "cs"))
        self.assertEqual(sentence, "Medián – door-to-needle, 2025, My Hospital: 161 minut (počet pacientů: 39).")

    def test_a_period_without_cases_says_so_without_the_model(self) -> None:
        with patch.object(value_answer, "_write_now", side_effect=AssertionError("no model call")):
            (sentence,) = asyncio.run(value_answer.answer_values([self._result(value=None, patients=0)], "en"))
        self.assertEqual(sentence, "Median – door to needle, 2025, My Hospital: no cases.")


class ValuePlanCheckTests(unittest.TestCase):
    def test_value_dates_come_from_the_question(self) -> None:
        plan = AnalysisPlan(values=[_value(period={"start_date": "2022-01-01", "end_date": "2022-12-31"})])
        with self.assertRaisesRegex(ValueError, "TimeWindow"):
            _assert_value_dates_are_stated(plan, "What was our average door to needle time last year?")
        _assert_value_dates_are_stated(plan, "What was our average door to needle time in 2022?")

    def test_a_plain_request_counts_value_metrics(self) -> None:
        plan = AnalysisPlan(values=[ValueSpec.model_validate({"metric": "THROMBOLYSIS", "measure": {"type": "RATE"}})])
        self.assertIsNone(_plain_plan_mismatch({"metric": ["THROMBOLYSIS"]}, plan))
        self.assertIn("not a value", _plain_plan_mismatch({"metric": ["THROMBOLYSIS"], "chart_type": ["BAR"]}, plan) or "")


if __name__ == "__main__":
    unittest.main()
