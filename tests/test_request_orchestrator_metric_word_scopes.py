import unittest

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.request_orchestrator import _drop_hospital_scopes_from_metric_words


def _plan(metric: str, scope_value: str | None, scope_type: str = "provider_name"):
    metric_spec = {"metric": metric}
    if scope_value is not None:
        metric_spec["originScope"] = {"scopeType": scope_type, "value": scope_value}
    return AnalysisPlan.model_validate({"charts": [{"chart_type": "BAR", "metrics": [metric_spec]}]})


class DropHospitalScopesFromMetricWordsTests(unittest.TestCase):
    def test_scope_made_of_the_metrics_own_words_is_dropped(self) -> None:
        # "Show me a bar chart of aa imaging at first hospital": the planner
        # read "first hospital" as a hospital name.
        plan = _drop_hospital_scopes_from_metric_words(_plan("AA_IMAGING", "first hospital"), {"metric": "AA_IMAGING"})
        self.assertIsNone(plan.charts[0].metrics[0].origin_scope)

    def test_provider_group_scope_made_of_metric_words_is_dropped(self) -> None:
        # Second live shape of the same mistake: "first hospital" emitted as a
        # provider_group_name, failing with "could not match that provider group".
        plan = _drop_hospital_scopes_from_metric_words(_plan("AA_IMAGING", "first hospital", "provider_group_name"), {"metric": "AA_IMAGING"})
        self.assertIsNone(plan.charts[0].metrics[0].origin_scope)

    def test_scope_is_kept_when_nlu_saw_a_hospital_name(self) -> None:
        plan = _drop_hospital_scopes_from_metric_words(
            _plan("AA_IMAGING", "First Hospital"),
            {"metric": "AA_IMAGING", "hospital_name": "First Hospital"},
        )
        self.assertEqual(plan.charts[0].metrics[0].origin_scope.value, "First Hospital")

    def test_real_hospital_name_is_kept(self) -> None:
        plan = _drop_hospital_scopes_from_metric_words(_plan("DTN", "Army Alhama de Murcia Hospital"), {"metric": "DTN"})
        self.assertEqual(plan.charts[0].metrics[0].origin_scope.value, "Army Alhama de Murcia Hospital")

    def test_other_scope_types_and_scopeless_plans_are_untouched(self) -> None:
        plan = AnalysisPlan.model_validate(
            {"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN", "originScope": {"scopeType": "mine"}}, {"metric": "DTN"}]}]}
        )
        self.assertIs(_drop_hospital_scopes_from_metric_words(plan, {"metric": "DTN"}), plan)


if __name__ == "__main__":
    unittest.main()
