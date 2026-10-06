import unittest
from datetime import date
from unittest.mock import patch

from src.domain.langchain.schema import AnalysisSemanticsSpec, ChartSpec, GroupByTime, MeasureSemanticsSpec, MetricSpec, TimeRange, TimeSemanticsSpec
from src.executors.mapping.chart_builder import _derive_title
from src.executors.planning.query_compiler import Dimension, _group_by_from_semantics, compile_chart_grouping
from src.planners.langchain.request_orchestrator import _decision_stage, _looks_like_period_reference

PERIODS = [
    TimeRange(start_date="2023-01-01", end_date="2023-03-31"),
    TimeRange(start_date="2025-07-01", end_date="2025-09-30"),
    TimeRange(start_date="2026-04-01", end_date="2026-06-30"),
]


def _chart(grain=None) -> ChartSpec:
    return ChartSpec(
        chart_type="BAR",
        metrics=[MetricSpec(metric="AGE")],
        semantics=AnalysisSemanticsSpec(intent="COMPARISON", measure=MeasureSemanticsSpec(type="MEDIAN"), time=TimeSemanticsSpec(grain=grain, periods=PERIODS)),
    )


class ExplicitTimePeriodTests(unittest.TestCase):
    def test_periods_become_a_time_dimension_with_an_inferred_grain(self) -> None:
        groups = _group_by_from_semantics(_chart())
        self.assertEqual(len(groups), 1)
        self.assertIsInstance(groups[0], GroupByTime)
        self.assertEqual(groups[0].grain, "QUARTER")
        self.assertEqual(groups[0].periods, PERIODS)

    def test_categories_are_exactly_the_requested_periods(self) -> None:
        cats = Dimension(GroupByTime(grain="QUARTER", periods=PERIODS)).categories()
        self.assertEqual(cats, [(date(2023, 1, 1), date(2023, 3, 31)), (date(2025, 7, 1), date(2025, 9, 30)), (date(2026, 4, 1), date(2026, 6, 30))])

    def test_compiled_batches_carry_one_time_period_per_requested_period(self) -> None:
        compiled = compile_chart_grouping(_chart("QUARTER"))
        periods = compiled.batches[0].batched_time_periods
        starts = [getattr(p, "startDate", None) or getattr(p, "start_date", None) for p in periods]
        self.assertEqual(starts, ["2023-01-01", "2025-07-01", "2026-04-01"])

    def test_title_lists_the_periods_instead_of_a_span(self) -> None:
        title = _derive_title(_chart("QUARTER"), [Dimension(GroupByTime(grain="QUARTER", periods=PERIODS))], sampled_period_override="2023-01-01 to 2026-06-30")
        self.assertEqual(title, "Age at stroke onset for 2023-Q1, 2025-Q3, 2026-Q2")

    def test_reversed_period_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TimeSemanticsSpec(periods=[TimeRange(start_date="2025-09-30", end_date="2025-07-01")])

    def test_single_window_is_unchanged(self) -> None:
        chart = ChartSpec(chart_type="LINE", metrics=[MetricSpec(metric="DTN")], semantics=AnalysisSemanticsSpec(intent="TREND", measure=MeasureSemanticsSpec(type="MEDIAN"), time=TimeSemanticsSpec(grain="QUARTER")))
        groups = _group_by_from_semantics(chart)
        self.assertIsNone(groups[0].periods)


class PeriodReferenceRejectionGuardTests(unittest.TestCase):
    def test_period_references_are_recognised(self) -> None:
        for value in ["Q1 2023", "2025 Q3", "q4 2026", "March 2025", "Sept 2024", "2024", "2024-06", "2024-06-30"]:
            self.assertTrue(_looks_like_period_reference(value), value)
        for value in ["2024-13-01", "yesterday", "", "the first quarter"]:
            self.assertFalse(_looks_like_period_reference(value), value)

    def test_quarter_references_are_never_rejected_as_invalid_dates(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "reject", "reason": "invalid_date_format", "missing_fields": None, "message": "The date format is invalid."},
        ):
            outcome = _decision_stage(
                question="show me age at stroke onset for Q1 2023, Q3 2025 and Q2 2026 in a bar chart",
                entities={"metric": "AGE", "chart_type": "BAR", "date": ["Q1 2023", "Q3 2025", "Q2 2026"]},
                language="en",
            )
        self.assertEqual(outcome.decision, "proceed")

    def test_a_genuinely_impossible_date_is_still_rejected(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "reject", "reason": "invalid_date_format", "missing_fields": None, "message": "The date format is invalid."},
        ):
            outcome = _decision_stage(question="dtn for 2024-13-01", entities={"metric": "DTN", "date": ["2024-13-01"]}, language="en")
        self.assertEqual(outcome.decision, "reject")


if __name__ == "__main__":
    unittest.main()
