import unittest

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.request_orchestrator import _drop_unrequested_explicit_periods

_QUARTERS_2023 = [
    {"start_date": "2023-01-01", "end_date": "2023-03-31"},
    {"start_date": "2023-04-01", "end_date": "2023-06-30"},
    {"start_date": "2023-07-01", "end_date": "2023-09-30"},
    {"start_date": "2023-10-01", "end_date": "2023-12-31"},
]


def _plan(periods):
    return AnalysisPlan.model_validate(
        {
            "charts": [
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "AA_DTN_LE60"}],
                    "semantics": {"intent": "TREND", "measure": {"type": "RATE"}, "time": {"grain": "QUARTER", "periods": periods}},
                }
            ]
        }
    )


class DropUnrequestedExplicitPeriodsTests(unittest.TestCase):
    def test_periods_the_question_never_named_are_dropped(self) -> None:
        # Live: "percentage of AA_DTN_LE60 per quarter" came back as the four quarters of 2023.
        plan = _drop_unrequested_explicit_periods(_plan(_QUARTERS_2023), "Show me the percentage of AA_DTN_LE60 per quarter", {"metric": "AA_DTN_LE60"})

        time_spec = plan.charts[0].semantics.time
        self.assertIsNone(time_spec.periods)
        self.assertEqual(time_spec.grain, "QUARTER")

    def test_named_quarters_keep_their_periods(self) -> None:
        plan = _plan(_QUARTERS_2023)
        for question in ("age at stroke onset for Q1 2023, Q3 2025 and Q2 2026", "dtn in 2023 per quarter", "dtn for January and March 2025", "dtn for the last 4 quarters"):
            with self.subTest(question=question):
                self.assertIs(_drop_unrequested_explicit_periods(plan, question, {"metric": "DTN"}), plan)

    def test_a_date_entity_keeps_the_periods(self) -> None:
        # A follow-up turn carries the earlier date entities but not the wording.
        plan = _plan(_QUARTERS_2023)
        self.assertIs(_drop_unrequested_explicit_periods(plan, "group by sex", {"metric": "DTN", "date": ["Q1 2023", "Q3 2025"]}), plan)

    def test_plans_without_periods_are_untouched(self) -> None:
        plan = _plan(None)
        self.assertIs(_drop_unrequested_explicit_periods(plan, "dtn per quarter", {"metric": "DTN"}), plan)
        bare = AnalysisPlan.model_validate({"charts": [{"chart_type": "BAR", "metrics": [{"metric": "DTN"}]}]})
        self.assertIs(_drop_unrequested_explicit_periods(bare, "dtn per quarter", {"metric": "DTN"}), bare)


if __name__ == "__main__":
    unittest.main()
