import unittest
from unittest.mock import patch

from src.domain.langchain.schema import (
    AnalysisPlan,
    AnalysisSemanticsSpec,
    AndFilter,
    ChartSpec,
    DataOriginSpec,
    DateFilter,
    MeasureSemanticsSpec,
    MetricSpec,
    OriginScopeSpec,
    SplitSpec,
    StatisticalTestSpec,
)
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome, _detect_unsupported_risk_factor_filter, orchestrate_visualization_request


class RequestOrchestratorStatisticalValidationTests(unittest.TestCase):
    def test_drops_redundant_canonical_stroke_type_self_split_before_returning_plan(self) -> None:
        plan = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[MetricSpec(metric="STROKE_TYPE")],
                    semantics=AnalysisSemanticsSpec(
                        intent="DISTRIBUTION",
                        measure=MeasureSemanticsSpec(type="DISTRIBUTION"),
                        splits=[SplitSpec(kind="CANONICAL", field="STROKE_TYPE")],
                    ),
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Show me a line graph of STROKE_TYPE",
                entities={"chart_type": "LINE", "metric": "STROKE_TYPE"},
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        chart = outcome.plan.charts[0]
        assert chart.semantics is not None
        self.assertIsNone(chart.semantics.splits)

    def test_allows_valid_vte_intervention_metrics_as_standalone_metrics(self) -> None:
        for metric in [
            "VTE_INTERVENTION_AIS",
            "VTE_INTERVENTION_ICH",
            "VTE_INTERVENTION_TYPE_AIS",
            "VTE_INTERVENTION_TYPE_ICH",
        ]:
            self.assertIsNone(
                _detect_unsupported_risk_factor_filter(
                    f"Show me a line graph of {metric}, DTO/Orchestration Error",
                    {"metric": [metric]},
                )
            )

    def test_clarifies_when_statistical_entities_do_not_define_two_cohorts(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between Aalborg University - Hospital and Kronborg Castle Hospital",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "date": ["2024-01-01", "2026-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two explicit cohorts", outcome.message or "")

    def test_proceeds_to_planning_when_entities_define_two_cohorts(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerGroupId=[279]),
                            origin_scope=OriginScopeSpec(scopeType="provider_group_name", value="Aalborg University - Hospital", label="Aalborg University - Hospital"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerGroupId=[280]),
                            origin_scope=OriginScopeSpec(scopeType="provider_group_name", value="Kronborg Castle Hospital", label="Kronborg Castle Hospital"),
                        ),
                    ],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider group 279 and provider group 280 from 2024-01-01 to 2026-12-31",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "group_id": ["provider group 279", "provider group 280"],
                    "date": ["2024-01-01", "2026-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

    def test_clarifies_when_mann_whitney_has_fewer_than_two_metrics(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[MetricSpec(metric="DTN")],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test on DTN",
                entities={"statistical_test_type": "MANN_WHITNEY_U_TEST"},
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two explicit cohorts", outcome.message or "")

    def test_clarifies_when_mann_whitney_metrics_are_not_distinct_cohorts(self) -> None:
        shared_origin = DataOriginSpec(providerId=[1])
        shared_scope = OriginScopeSpec(scopeType="mine", label="same")
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin, origin_scope=shared_scope),
                        MetricSpec(metric="DTN", data_origin=shared_origin, origin_scope=shared_scope),
                    ],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "group_id": ["provider group 111", "provider group 112"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two distinct cohorts", outcome.message or "")

    def test_proceeds_when_distinct_cohorts_are_present(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerId=[1]),
                            origin_scope=OriginScopeSpec(scopeType="mine", label="male cohort"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerId=[1]),
                            origin_scope=OriginScopeSpec(scopeType="country_code", countryCode="CZ", label="female cohort"),
                        ),
                    ],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "group_id": ["provider group 1", "provider group 2"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

    def test_clarifies_when_question_mentions_provider_group_but_entities_only_have_provider_ids(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider group 289 and provider group 252",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "provider_id": ["provider 289", "provider 252"],
                    "date": ["2023-01-01", "2023-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_group_cohorts")
        self.assertIn("provider groups", outcome.message or "")

    def test_clarifies_when_question_mentions_provider_but_entities_only_have_provider_group_ids(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider 289 and provider 252",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "group_id": ["provider group 289", "provider group 252"],
                    "date": ["2023-01-01", "2023-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_cohorts")
        self.assertIn("providers", outcome.message or "")

    def test_proceeds_for_quarter_vs_quarter_temporal_mann_whitney(self) -> None:
        shared_origin = DataOriginSpec(providerId=[279])
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2025-10-01"),
                            DateFilter(operator="LE", value="2025-12-31"),
                        ]
                    ),
                ),
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2026-01-01"),
                            DateFilter(operator="LE", value="2026-03-31"),
                        ]
                    ),
                ),
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Compare my DTN for Q4 2025 and Q1 2026 using a Mann-Whitney U test",
                entities={
                    "metric": ["DTN"],
                    "date": ["Q4 2025", "Q1 2026"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

    def test_proceeds_for_year_vs_year_temporal_mann_whitney(self) -> None:
        shared_origin = DataOriginSpec(providerId=[279])
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2025-01-01"),
                            DateFilter(operator="LE", value="2025-12-31"),
                        ]
                    ),
                ),
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                        MetricSpec(metric="DTN", data_origin=shared_origin),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2026-01-01"),
                            DateFilter(operator="LE", value="2026-12-31"),
                        ]
                    ),
                ),
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Compare DTN for 2025 and 2026 using Mann-Whitney U",
                entities={
                    "metric": ["DTN"],
                    "date": ["2025", "2026"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

    def test_still_clarifies_for_single_iso_date_range_without_cohorts(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run Mann-Whitney U for DTN from 2025-01-01 to 2025-12-31",
                entities={
                    "metric": ["DTN"],
                    "date": ["2025-01-01", "2025-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")

    def test_clarifies_when_statistical_cohorts_are_only_in_free_text(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            origin_scope=OriginScopeSpec(scopeType="mine", label="mine"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            origin_scope=OriginScopeSpec(
                                scopeType="provider_name",
                                value="Army Alhama de Murcia Hospital",
                                label="Army Alhama de Murcia Hospital",
                            ),
                        ),
                    ],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Can you compare my dtn against army alhama de murcia hospital using a mann whitney u test",
                entities={
                    "metric": ["DTN"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")

    def test_proceeds_when_scope_entities_define_statistical_cohorts(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            origin_scope=OriginScopeSpec(scopeType="mine", label="mine"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            origin_scope=OriginScopeSpec(
                                scopeType="provider_name",
                                value="Army Alhama de Murcia Hospital",
                                label="Army Alhama de Murcia Hospital",
                            ),
                        ),
                    ],
                )
            ]
        )

        with (
            patch(
                "src.planners.langchain.request_orchestrator._decision_stage",
                return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
            ),
            patch(
                "src.planners.langchain.request_orchestrator.generate_analysis_plan",
                return_value=plan,
            ),
        ):
            outcome = orchestrate_visualization_request(
                question="Can you compare my dtn against army alhama de murcia hospital using a mann whitney u test",
                entities={
                    "metric": ["DTN"],
                    "hospital_scope_reference": ["my hospital"],
                    "hospital_name": ["Army Alhama de Murcia Hospital"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)


if __name__ == "__main__":
    unittest.main()
