import unittest

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.request_orchestrator import _default_trend_grain


def _plan(intent: str, time=None):
    semantics = {"intent": intent, "measure": {"type": "MEDIAN"}}
    if time is not None:
        semantics["time"] = time
    return AnalysisPlan.model_validate(
        {"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": semantics}]}
    )


def _time(plan: AnalysisPlan):
    return plan.charts[0].semantics.time


class DefaultTrendGrainTests(unittest.TestCase):
    def test_trend_without_time_is_plotted_per_quarter(self) -> None:
        # Live: "Show me DTN as a line chart" planned as TREND with no time, then drawn as a distribution.
        time = _time(_default_trend_grain(_plan("TREND")))
        self.assertEqual(time.grain, "QUARTER")
        self.assertIsNone(time.window)

    def test_trend_with_a_window_but_no_grain_keeps_the_window(self) -> None:
        window = {"start_date": "2025-01-01", "end_date": "2025-12-31"}
        time = _time(_default_trend_grain(_plan("TREND", {"window": window})))
        self.assertEqual(time.grain, "QUARTER")
        self.assertEqual(time.window.model_dump(), window)

    def test_a_named_grain_or_periods_are_kept(self) -> None:
        self.assertEqual(_time(_default_trend_grain(_plan("TREND", {"grain": "MONTH"}))).grain, "MONTH")
        periods = [{"start_date": "2023-01-01", "end_date": "2023-03-31"}, {"start_date": "2025-07-01", "end_date": "2025-09-30"}]
        time = _time(_default_trend_grain(_plan("TREND", {"periods": periods})))
        self.assertIsNone(time.grain)
        self.assertEqual(len(time.periods), 2)

    def test_other_intents_are_left_alone(self) -> None:
        self.assertIsNone(_time(_default_trend_grain(_plan("DISTRIBUTION"))))


if __name__ == "__main__":
    unittest.main()
