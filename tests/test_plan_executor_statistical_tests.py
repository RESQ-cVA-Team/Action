import asyncio
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.domain.dto.analytics.statistical_test import StatisticalTestResult
from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.graphql.request import DataOrigin, GraphQLQueryRequest, TimePeriod
from src.domain.langchain.schema import (
    AnalysisPlan,
    AnalysisSemanticsSpec,
    AndFilter,
    ChartSpec,
    DataOriginSpec,
    DateFilter,
    GroupBySex,
    MeasureSemanticsSpec,
    MetricSpec,
    SplitSpec,
    StatisticalTestSpec,
)
from src.executors.orchestration import plan_executor
from src.executors.planning.query_compiler import CompiledBatch, CompiledChartGrouping


class PlanExecutorStatisticalTestTests(unittest.TestCase):
    def test_execute_plan_async_maps_invalid_split_semantics_to_structured_error(self) -> None:
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

        with patch.object(plan_executor, "resolve_plan_metric_origins", return_value=plan):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                asyncio.run(
                    plan_executor.execute_plan_async(
                        plan=plan,
                        user_sub="user-1",
                        trace_id="trace-1",
                    )
                )

        self.assertEqual(err.exception.reason, "invalid_plan_semantics")
        self.assertEqual(err.exception.code, "EXEC_PLAN_INVALID_SEMANTICS")

    def test_to_execution_error_messages_include_actionable_guidance(self) -> None:
        graphql_error = plan_executor._to_execution_error(["graphql_error"], trace_id="trace-1")
        fallback_error = plan_executor._to_execution_error(["unexpected"], trace_id="trace-1")

        self.assertEqual(graphql_error.code, "EXEC_GRAPHQL_ERROR")
        self.assertIn("Please", graphql_error.user_message)
        self.assertIn("try again", graphql_error.user_message)

        self.assertEqual(fallback_error.code, "EXEC_DATA_FETCH_FAILED")
        self.assertIn("Please", fallback_error.user_message)
        self.assertIn("try again", fallback_error.user_message)

    def test_mann_whitney_missing_distinct_cohorts_message_is_actionable(self) -> None:
        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1]))],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-01-01"),
                    DateFilter(operator="LE", value="2023-12-31"),
                ]
            ),
        )

        with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
            plan_executor._execute_mann_whitney_test(
                test=test,
                user_sub="user-1",
                trace_id="trace-1",
            )

        self.assertEqual(err.exception.reason, "missing_statistical_cohorts")
        self.assertIn("Please provide", err.exception.user_message)
        self.assertIn("cohort A", err.exception.user_message)
        self.assertIn("cohort B", err.exception.user_message)

    def test_mann_whitney_returns_clarification_when_no_cohort_rows_exist(self) -> None:
        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-01-01"),
                    DateFilter(operator="LE", value="2023-12-31"),
                ]
            ),
        )

        with (
            patch.object(
                plan_executor,
                "_translate_mann_whitney_metrics",
                return_value=["DTN"],
            ),
            patch.object(
                plan_executor,
                "_execute_mann_whitney_query",
                return_value=[],
            ),
        ):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                plan_executor._execute_mann_whitney_test(
                    test=test,
                    user_sub="user-1",
                    trace_id="trace-1",
                )

        self.assertEqual(err.exception.reason, "empty_statistical_cohort")
        self.assertEqual(err.exception.code, "EXEC_STATS_EMPTY_COHORT")
        self.assertIn("different cohorts or broaden the date range", err.exception.user_message)

    def test_mann_whitney_group_by_is_rejected_by_schema(self) -> None:
        with self.assertRaises(ValidationError) as err:
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=[MetricSpec(metric="DTN")],
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
            )

        self.assertIn("do not support group_by", str(err.exception))

    def test_execute_plan_async_fails_fast_when_statistical_cohorts_are_missing(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                    ],
                )
            ]
        )

        with patch.object(plan_executor, "resolve_plan_metric_origins", return_value=plan):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                asyncio.run(
                    plan_executor.execute_plan_async(
                        plan=plan,
                        user_sub="user-1",
                        trace_id="trace-1",
                    )
                )

        self.assertEqual(err.exception.reason, "missing_statistical_cohorts")
        self.assertEqual(err.exception.code, "EXEC_STATS_MISSING_COHORTS")

    def test_execute_plan_async_fails_for_unsupported_statistical_test_type(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
                    ],
                )
            ]
        )
        plan.statistical_tests[0].test_type = "UNSUPPORTED_TEST"

        with patch.object(plan_executor, "resolve_plan_metric_origins", return_value=plan):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                asyncio.run(
                    plan_executor.execute_plan_async(
                        plan=plan,
                        user_sub="user-1",
                        trace_id="trace-1",
                    )
                )

        self.assertEqual(err.exception.reason, "unsupported_statistical_test_type")
        self.assertEqual(err.exception.code, "EXEC_STATS_UNSUPPORTED_TEST_TYPE")

    def test_execute_plan_async_drops_empty_scope_and_renders_the_rest(self) -> None:
        chart = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2023-01-01"),
                            DateFilter(operator="LE", value="2023-12-31"),
                        ]
                    ),
                )
            ]
        )

        first = plan_executor.RequestExecutionResult(
            spec=plan_executor.RequestSpec(
                req=GraphQLQueryRequest(
                    metrics=[],
                    timePeriod=TimePeriod(startDate="2023-01-01", endDate="2023-12-31"),
                    dataOrigin=DataOrigin(providerId=[1]),
                ),
                label_parts=["A"],
                include_metric_alias=False,
                group_by_field=None,
                add_time_period_labels=False,
                scope_label="Scope A",
            ),
            series=[
                ChartSeries(
                    name="DTN",
                    data=[ChartPoint(x="2023-01", y=1.0)],
                )
            ],
        )
        second = plan_executor.RequestExecutionResult(
            spec=plan_executor.RequestSpec(
                req=GraphQLQueryRequest(
                    metrics=[],
                    timePeriod=TimePeriod(startDate="2023-01-01", endDate="2023-12-31"),
                    dataOrigin=DataOrigin(providerId=[2]),
                ),
                label_parts=["B"],
                include_metric_alias=False,
                group_by_field=None,
                add_time_period_labels=False,
                scope_label="Scope B",
            ),
            series=[],
        )

        async def _fake_execute_specs_concurrent(**kwargs):
            return [first, second]

        with (
            patch.object(plan_executor, "resolve_plan_metric_origins", return_value=chart),
            patch.object(
                plan_executor,
                "build_metric_requests",
                return_value=([], None, [None], [None]),
            ),
            patch.object(
                plan_executor,
                "compile_chart_grouping",
                return_value=CompiledChartGrouping(
                    dimensions=[],
                    batches=[
                        CompiledBatch(
                            server_groupby=None,
                            filter_dims=[],
                            combos_list=[tuple()],
                            batched_time_enabled=False,
                            batched_time_periods=[],
                        )
                    ],
                ),
            ),
            patch.object(
                plan_executor,
                "build_primary_request_specs",
                return_value=[first.spec, second.spec],
            ),
            patch.object(
                plan_executor,
                "_execute_specs_concurrent",
                side_effect=_fake_execute_specs_concurrent,
            ),
            patch.object(
                plan_executor,
                "estimate_query_count_for_plan",
                return_value=2,
            ),
        ):
            response = asyncio.run(
                plan_executor.execute_plan_async(
                    plan=chart,
                    user_sub="user-1",
                    trace_id="trace-1",
                )
            )

        self.assertEqual(len(response.charts), 1)
        self.assertEqual(len(response.charts[0].series), 1)
        self.assertEqual(response.charts[0].series[0].name, "DTN")
        self.assertTrue(
            any("Scope B" in warning for warning in response.warnings),
            response.warnings,
        )

    def test_execute_plan_async_fails_when_every_scope_returns_no_data(self) -> None:
        chart = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                        MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
                    ],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2023-01-01"),
                            DateFilter(operator="LE", value="2023-12-31"),
                        ]
                    ),
                )
            ]
        )

        first = plan_executor.RequestExecutionResult(
            spec=plan_executor.RequestSpec(
                req=GraphQLQueryRequest(
                    metrics=[],
                    timePeriod=TimePeriod(startDate="2023-01-01", endDate="2023-12-31"),
                    dataOrigin=DataOrigin(providerId=[1]),
                ),
                label_parts=["A"],
                include_metric_alias=False,
                group_by_field=None,
                add_time_period_labels=False,
                scope_label="Scope A",
            ),
            series=[],
        )
        second = plan_executor.RequestExecutionResult(
            spec=plan_executor.RequestSpec(
                req=GraphQLQueryRequest(
                    metrics=[],
                    timePeriod=TimePeriod(startDate="2023-01-01", endDate="2023-12-31"),
                    dataOrigin=DataOrigin(providerId=[2]),
                ),
                label_parts=["B"],
                include_metric_alias=False,
                group_by_field=None,
                add_time_period_labels=False,
                scope_label="Scope B",
            ),
            series=[],
        )

        async def _fake_execute_specs_concurrent(**kwargs):
            return [first, second]

        with (
            patch.object(plan_executor, "resolve_plan_metric_origins", return_value=chart),
            patch.object(
                plan_executor,
                "build_metric_requests",
                return_value=([], None, [None], [None]),
            ),
            patch.object(
                plan_executor,
                "compile_chart_grouping",
                return_value=CompiledChartGrouping(
                    dimensions=[],
                    batches=[
                        CompiledBatch(
                            server_groupby=None,
                            filter_dims=[],
                            combos_list=[tuple()],
                            batched_time_enabled=False,
                            batched_time_periods=[],
                        )
                    ],
                ),
            ),
            patch.object(
                plan_executor,
                "build_primary_request_specs",
                return_value=[first.spec, second.spec],
            ),
            patch.object(
                plan_executor,
                "_execute_specs_concurrent",
                side_effect=_fake_execute_specs_concurrent,
            ),
            patch.object(
                plan_executor,
                "estimate_query_count_for_plan",
                return_value=2,
            ),
        ):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                asyncio.run(
                    plan_executor.execute_plan_async(
                        plan=chart,
                        user_sub="user-1",
                        trace_id="trace-1",
                    )
                )

        self.assertEqual(err.exception.reason, "no_data")

    def test_execute_temporal_pair_mann_whitney_returns_query_results(self) -> None:
        test_a = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-01-01"),
                    DateFilter(operator="LE", value="2023-06-30"),
                ]
            ),
        )
        test_b = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-07-01"),
                    DateFilter(operator="LE", value="2023-12-31"),
                ]
            ),
        )

        expected = [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="success",
                details={"cohort_a_size": 10, "cohort_b_size": 12},
            )
        ]

        with (
            patch.object(
                plan_executor,
                "_translate_mann_whitney_metrics",
                return_value=["DTN"],
            ),
            patch.object(
                plan_executor,
                "_execute_mann_whitney_query",
                return_value=expected,
            ),
        ):
            results = plan_executor._execute_temporal_pair_mann_whitney(
                test_a=test_a,
                test_b=test_b,
                user_sub="user-1",
                trace_id="trace-1",
            )

        self.assertEqual(results, expected)

    def test_validate_statistical_tests_readiness_allows_temporal_pair(self) -> None:
        test_a = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2025-10-01"),
                    DateFilter(operator="LE", value="2025-12-31"),
                ]
            ),
        )
        test_b = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2026-01-01"),
                    DateFilter(operator="LE", value="2026-03-31"),
                ]
            ),
        )
        plan = AnalysisPlan(statistical_tests=[test_a, test_b])

        plan_executor._validate_statistical_tests_readiness(plan, trace_id="trace-1")

    def test_validate_statistical_tests_readiness_terminates_for_single_valid_test(self) -> None:
        # Regression test: a single valid two-cohort Mann-Whitney test must not hang.
        # The readiness loop previously never advanced its index on this path, spinning
        # forever instead of returning. Bound execution in a thread so a regression
        # fails the test instead of hanging the whole suite.
        import threading

        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[279])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1853])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2025-01-01"),
                    DateFilter(operator="LE", value="2025-12-31"),
                ]
            ),
        )
        plan = AnalysisPlan(statistical_tests=[test])

        outcome: dict = {}

        def _run() -> None:
            try:
                plan_executor._validate_statistical_tests_readiness(plan, trace_id="trace-1")
                outcome["completed"] = True
            except Exception as exc:  # pragma: no cover - would indicate a different bug
                outcome["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive(), "readiness validation hung on a single valid test")
        self.assertTrue(outcome.get("completed"), outcome.get("error"))

    def test_mann_whitney_fails_for_unauthorized_cohorts(self) -> None:
        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-01-01"),
                    DateFilter(operator="LE", value="2023-12-31"),
                ]
            ),
        )

        with (
            patch.object(
                plan_executor,
                "_translate_mann_whitney_metrics",
                return_value=["DTN"],
            ),
            patch.object(
                plan_executor,
                "_execute_mann_whitney_query",
                return_value=[
                    StatisticalTestResult(
                        test_type="MANN_WHITNEY_U_TEST",
                        status="error",
                        reason="No permission to calculate KPIs for given data origin",
                    )
                ],
            ),
        ):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                plan_executor._execute_mann_whitney_test(
                    test=test,
                    user_sub="user-1",
                    trace_id="trace-1",
                )

        self.assertEqual(err.exception.reason, "unauthorized_statistical_cohort")
        self.assertEqual(err.exception.code, "EXEC_STATS_UNAUTHORIZED_COHORT")


if __name__ == "__main__":
    unittest.main()
