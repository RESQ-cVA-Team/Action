import unittest

from src.domain.langchain.schema import AnalysisPlan, ChartSpec, StatisticalTestSpec
from src.planners.langchain.request_orchestrator import _split_mixed_unit_charts


class FilterListNormalizationTests(unittest.TestCase):
    """Regression tests for a real CVaLab-caught bug: the planner LLM
    sometimes emits "filters" as a bare JSON list (e.g.
    [{"type": "SexFilter", "value": "MALE"}]) instead of a single FilterNode
    object, causing an 18-way pydantic ValidationError (one failure per
    FilterNode union member) that discarded an otherwise-correct plan and
    fell through to a generic "I need a bit more detail" clarify.
    """

    def test_single_element_list_unwraps_to_the_bare_filter(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "LINE",
                "filters": [{"type": "SexFilter", "value": "MALE"}],
                "metrics": [{"metric": "DTN"}],
            }
        )
        self.assertEqual(chart.filters.type, "SexFilter")
        self.assertEqual(chart.filters.value, "MALE")

    def test_multi_element_list_wraps_in_and_filter(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "LINE",
                "filters": [
                    {"type": "SexFilter", "value": "MALE"},
                    {"type": "StrokeFilter", "value": "ISCHEMIC"},
                ],
                "metrics": [{"metric": "DTN"}],
            }
        )
        self.assertEqual(chart.filters.type, "AndFilter")
        self.assertEqual(len(chart.filters.and_), 2)

    def test_empty_list_normalizes_to_none(self) -> None:
        chart = ChartSpec.model_validate(
            {"chart_type": "LINE", "filters": [], "metrics": [{"metric": "DTN"}]}
        )
        self.assertIsNone(chart.filters)

    def test_bare_object_is_unaffected(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "LINE",
                "filters": {"type": "SexFilter", "value": "MALE"},
                "metrics": [{"metric": "DTN"}],
            }
        )
        self.assertEqual(chart.filters.type, "SexFilter")

    def test_absent_filters_stays_none(self) -> None:
        chart = ChartSpec.model_validate({"chart_type": "LINE", "metrics": [{"metric": "DTN"}]})
        self.assertIsNone(chart.filters)

    def test_statistical_test_spec_gets_the_same_normalization(self) -> None:
        spec = StatisticalTestSpec.model_validate(
            {
                "test_type": "MANN_WHITNEY_U_TEST",
                "metrics": [{"metric": "DTN"}],
                "filters": [{"type": "SexFilter", "value": "MALE"}],
            }
        )
        self.assertEqual(spec.filters.type, "SexFilter")

    def test_full_plan_with_the_exact_reported_shape_validates(self) -> None:
        """The exact plan shape from the CVaLab failure: 'make two line
        charts, one of male patients DTN, the other of female patients DTN'.
        """
        plan = AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "LINE",
                        "filters": [{"type": "SexFilter", "value": "MALE"}],
                        "semantics": {"intent": "TREND", "measure": {"type": "MEAN"}},
                        "metrics": [{"metric": "DTN"}],
                    },
                    {
                        "chart_type": "LINE",
                        "filters": [{"type": "SexFilter", "value": "FEMALE"}],
                        "semantics": {"intent": "TREND", "measure": {"type": "MEAN"}},
                        "metrics": [{"metric": "DTN"}],
                    },
                ]
            }
        )
        self.assertEqual(plan.charts[0].filters.value, "MALE")
        self.assertEqual(plan.charts[1].filters.value, "FEMALE")

    def test_same_unit_multi_metric_chart_stays_conjoined(self) -> None:
        plan = AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "LINE",
                        "semantics": {"intent": "TREND", "measure": {"type": "MEAN"}},
                        "metrics": [{"metric": "DTN"}, {"metric": "DTG"}],
                    }
                ]
            }
        )

        split_plan = _split_mixed_unit_charts(plan)

        self.assertEqual(len(split_plan.charts), 1)
        self.assertEqual([m.metric for m in split_plan.charts[0].metrics], ["DTN", "DTG"])

    def test_mixed_unit_multi_metric_chart_splits_into_separate_charts(self) -> None:
        plan = AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "LINE",
                        "semantics": {"intent": "TREND", "measure": {"type": "MEAN"}},
                        "metrics": [{"metric": "DTN"}, {"metric": "SYSTOLIC_PRESSURE"}],
                    }
                ]
            }
        )

        split_plan = _split_mixed_unit_charts(plan)

        self.assertEqual(len(split_plan.charts), 2)
        self.assertEqual([m.metric for m in split_plan.charts[0].metrics], ["DTN"])
        self.assertEqual([m.metric for m in split_plan.charts[1].metrics], ["SYSTOLIC_PRESSURE"])

    def test_numeric_and_enum_metric_chart_splits_into_separate_charts(self) -> None:
        plan = AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "LINE",
                        "semantics": {"intent": "TREND", "measure": {"type": "MEAN"}},
                        "metrics": [{"metric": "DTN"}, {"metric": "SEX"}],
                    }
                ]
            }
        )

        split_plan = _split_mixed_unit_charts(plan)

        self.assertEqual(len(split_plan.charts), 2)
        self.assertEqual([m.metric for m in split_plan.charts[0].metrics], ["DTN"])
        self.assertEqual([m.metric for m in split_plan.charts[1].metrics], ["SEX"])
