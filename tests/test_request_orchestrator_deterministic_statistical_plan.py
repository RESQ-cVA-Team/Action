import unittest
from unittest.mock import patch

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome, orchestrate_visualization_request


class RequestOrchestratorDeterministicStatPlanTests(unittest.TestCase):
    def test_builds_deterministic_statistical_plan_for_semantic_scopes(self) -> None:
        entities = {
            "metric": ["DTN"],
            "provider_name": ["Aalborg Hospital", "Copenhagen Hospital"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between Aalborg Hospital and Copenhagen Hospital",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.metrics[0].origin_scope.scope_type, "provider_name")
        self.assertEqual(test.metrics[0].origin_scope.value, "Aalborg Hospital")
        self.assertEqual(test.metrics[1].origin_scope.scope_type, "provider_name")
        self.assertEqual(test.metrics[1].origin_scope.value, "Copenhagen Hospital")

    def test_does_not_override_metric_entity_from_question_text(self) -> None:
        entities = {
            "chart_type": ["LINE"],
            "metric": ["ICH_TREATMENT_TYPE"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="all_required_fields_present"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=AnalysisPlan(charts=[]),
            ) as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Show me a line graph of decompressive craniectomy performed",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(llm_plan.call_args)
        assert llm_plan.call_args is not None
        self.assertEqual(llm_plan.call_args.kwargs["entities"]["metric"], ["ICH_TREATMENT_TYPE"])
        self.assertEqual(entities["metric"], ["ICH_TREATMENT_TYPE"])

    def test_builds_deterministic_statistical_plan_for_provider_group_comparison(self) -> None:
        entities = {
            "metric": ["DTN"],
            "group_id": ["provider group 2825", "provider group 3001"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN, cohort A is provider group 2825 from 2023-01-01 to 2023-12-31, cohort B is provider group 3001 from 2023-01-01 to 2023-12-31",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.test_type, "MANN_WHITNEY_U_TEST")
        self.assertEqual(test.metrics[0].data_origin.provider_group_id, [2825])
        self.assertEqual(test.metrics[1].data_origin.provider_group_id, [3001])

        and_filters = test.filters.and_
        self.assertEqual(len(and_filters), 2)
        self.assertEqual(and_filters[0].operator, "GE")
        self.assertEqual(and_filters[0].value, "2023-01-01")
        self.assertEqual(and_filters[1].operator, "LE")
        self.assertEqual(and_filters[1].value, "2023-12-31")

    def test_builds_deterministic_statistical_plan_for_provider_id_comparison(self) -> None:
        entities = {
            "metric": ["DTN"],
            "provider_id": ["provider 289", "provider 252"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider 289 and provider 252",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.test_type, "MANN_WHITNEY_U_TEST")
        self.assertEqual(test.metrics[0].data_origin.provider_id, [289])
        self.assertEqual(test.metrics[1].data_origin.provider_id, [252])

    def test_prefers_provider_group_ids_when_both_id_types_present(self) -> None:
        entities = {
            "metric": ["DTN"],
            "group_id": ["provider group 2825", "provider group 3001"],
            "provider_id": ["provider 289", "provider 252"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider group 2825 and provider group 3001",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.metrics[0].data_origin.provider_group_id, [2825])
        self.assertEqual(test.metrics[1].data_origin.provider_group_id, [3001])
        self.assertIsNone(test.metrics[0].data_origin.provider_id)
        self.assertIsNone(test.metrics[1].data_origin.provider_id)

    def test_prefers_provider_ids_when_question_mentions_provider_not_group(self) -> None:
        entities = {
            "metric": ["DTN"],
            "group_id": ["provider group 2825", "provider group 3001"],
            "provider_id": ["provider 289", "provider 252"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider 289 and provider 252",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.metrics[0].data_origin.provider_id, [289])
        self.assertEqual(test.metrics[1].data_origin.provider_id, [252])
        self.assertIsNone(test.metrics[0].data_origin.provider_group_id)
        self.assertIsNone(test.metrics[1].data_origin.provider_group_id)

    def test_builds_deterministic_plan_from_scope_entities(self) -> None:
        entities = {
            "metric": ["DTN"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
            "hospital_scope_reference": ["my hospital"],
            "hospital_name": ["Army Alhama de Murcia Hospital"],
        }

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(
                    decision="proceed",
                    reason="all_required_fields_present",
                ),
            ),
            patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Can you compare my dtn against army alhama de murcia hospital using a mann whitney u test",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.metrics[0].origin_scope.scope_type, "mine")
        self.assertEqual(test.metrics[1].origin_scope.scope_type, "provider_name")
        self.assertEqual(test.metrics[1].origin_scope.value, "Army Alhama de Murcia Hospital")


if __name__ == "__main__":
    unittest.main()
