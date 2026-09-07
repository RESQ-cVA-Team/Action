import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, cast
from uuid import uuid4

from src.domain.dto.analytics import StatisticalTestResult
from src.domain.dto.charts.types import ChartSeries
from src.domain.dto.execution_summary import ExecutionBatchSummary, ExecutionSummary
from src.domain.dto.response import VisualizationResponse
from src.domain.graphql.request import BooleanFilter as GQLBooleanFilter
from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.request import IntegerFilter as GQLIntegerFilter
from src.domain.graphql.request import LogicalFilter as GQLLogicalFilter
from src.domain.graphql.request import SexFilter as GQLSexFilter
from src.domain.graphql.request import StrokeFilter as GQLStrokeFilter
from src.domain.graphql.request import TimePeriod, default_time_bounds
from src.domain.langchain.schema import AnalysisPlan, StatisticalTestSpec
from src.executors.graphql.client import GraphQLProxyClient
from src.executors.mapping.chart_builder import build_chart_dto
from src.executors.mapping.filter_mapper import to_gql_filter
from src.executors.mapping.series_mapper import merge_series_by_name
from src.executors.mapping.summary_builder import (
    make_batch_summary,
    make_execution_summary,
)
from src.executors.planning.metric_request_factory import (
    build_metric_requests,
)
from src.executors.planning.origin_scope_resolver import (
    OriginScopeResolutionError,
    resolve_plan_metric_origins,
)
from src.executors.planning.query_compiler import (
    Dimension,
    compile_chart_grouping,
    estimate_query_count_for_plan,
)
from src.executors.planning.request_plan import (
    RequestSpec,
    build_primary_request_specs,
)
from src.executors.transport.request_runner import run_graphql_request
from src.shared.ssot_loader import (
    get_statistics_metric_enum_map,
)
from src.util import env as env_util
from src.util.coalesce import coalesce
from src.util.logging_utils import bind_current_context

logger = logging.getLogger(__name__)
# Privacy/safety defaults:
# - Avoid logging raw GraphQL queries by default.
_LOG_GRAPHQL_QUERY = env_util.env_flag("EXECUTOR_LOG_GRAPHQL_QUERY", default=False)
_EMIT_COMPILER_DIAGNOSTICS = env_util.env_flag("EXECUTOR_EMIT_COMPILER_DIAGNOSTICS", default=False)
_INCLUDE_GENERAL_STATS = env_util.env_flag("EXECUTOR_INCLUDE_GENERAL_STATS", default=True)
_STRICT_MODE = env_util.env_flag("ANALYTICS_STRICT_MODE", default=False) or env_util.env_flag("EXECUTOR_STRICT_MODE", default=False)


def _parse_positive_int_env(name: str, raw_value: str, minimum: int) -> int:
    token = (raw_value or "").strip()
    try:
        parsed = int(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got: {raw_value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {parsed}")
    return parsed


def _parse_timeout_env(name: str, raw_value: str, minimum: int) -> int:
    token = (raw_value or "").strip()
    try:
        parsed = int(float(token))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got: {raw_value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {parsed}")
    return parsed


_executor_default_concurrency_raw = env_util.get_env("EXECUTOR_DEFAULT_MAX_CONCURRENCY", default="4") or "4"
_executor_default_concurrency = _parse_positive_int_env("EXECUTOR_DEFAULT_MAX_CONCURRENCY", _executor_default_concurrency_raw, 1)
_EXECUTOR_DEFAULT_MAX_CONCURRENCY = _executor_default_concurrency

_executor_sync_concurrency_raw = env_util.get_env("EXECUTOR_SYNC_MAX_CONCURRENCY", default="1") or "1"
_executor_sync_concurrency = _parse_positive_int_env("EXECUTOR_SYNC_MAX_CONCURRENCY", _executor_sync_concurrency_raw, 1)
_EXECUTOR_SYNC_MAX_CONCURRENCY = _executor_sync_concurrency

proxy_url, action_server_token = env_util.require_all_env("RASA_PROXY_URL", "ACTION_SERVER_TOKEN")
graphql_target = env_util.require_any_env("RASA_PROXY_GRAPHQL_TARGET")

_graphql_timeout_raw = env_util.get_env("EXECUTOR_GRAPHQL_TIMEOUT_SECONDS", default="30") or "30"
_graphql_timeout_seconds = _parse_timeout_env("EXECUTOR_GRAPHQL_TIMEOUT_SECONDS", _graphql_timeout_raw, 5)

_origin_scope_timeout_raw = env_util.get_env("EXECUTOR_ORIGIN_SCOPE_TIMEOUT_SECONDS", default="25") or "25"
_ORIGIN_SCOPE_TIMEOUT_SECONDS = _parse_timeout_env("EXECUTOR_ORIGIN_SCOPE_TIMEOUT_SECONDS", _origin_scope_timeout_raw, 5)

client = GraphQLProxyClient(
    proxy_url=proxy_url,
    action_server_token=action_server_token,
    target=graphql_target if isinstance(graphql_target, str) and graphql_target.strip() else "graphql",
    timeout_seconds=_graphql_timeout_seconds,
    connect_timeout_seconds=5,
    max_total_timeout_seconds=_graphql_timeout_seconds + 5,
    retry_attempts=1,
    retry_backoff_seconds=0.2,
)


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: Dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _collect_date_bounds(
    filter_obj: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    if filter_obj is None:
        return None, None

    min_start: Optional[str] = None
    max_end: Optional[str] = None

    def visit(node: Any) -> None:
        nonlocal min_start, max_end
        if isinstance(node, GQLLogicalFilter):
            for child in node.children:
                visit(child)
            return
        if isinstance(node, GQLDateFilter) and node.property == "DISCHARGE_DATE":
            op = node.operator.value
            val = node.value
            if op in ("GE", "GT"):
                if min_start is None or val < min_start:
                    min_start = val
            if op in ("LE", "LT"):
                if max_end is None or val > max_end:
                    max_end = val

    visit(filter_obj)
    return min_start, max_end


# Adaptation layer for GraphQL filter objects to the statistical test payload format. This is necessary because the GraphQL API expects a specific structure for filters.
def _serialize_case_filter_input(filter_obj: Optional[Any]) -> Optional[Dict[str, Any]]:
    if filter_obj is None:
        return None

    if isinstance(filter_obj, GQLLogicalFilter):
        children: List[Dict[str, Any]] = []
        for child in filter_obj.children:
            child_serialized = _serialize_case_filter_input(child)
            if child_serialized is not None:
                children.append(child_serialized)
        return {
            "node": {
                "logicalOperator": str(filter_obj.operator),
                "children": children,
            }
        }

    if isinstance(filter_obj, GQLIntegerFilter):
        return {
            "leaf": {
                "integerCaseFilter": {
                    "property": filter_obj.property,
                    "operator": filter_obj.operator.value,
                    "value": int(filter_obj.value),
                }
            }
        }

    if isinstance(filter_obj, GQLBooleanFilter):
        return {
            "leaf": {
                "booleanCaseFilter": {
                    "property": str(filter_obj.property),
                    "value": bool(filter_obj.value),
                }
            }
        }

    if isinstance(filter_obj, GQLSexFilter):
        return {
            "leaf": {
                "enumCaseFilter": {
                    "sexType": {
                        "values": [filter_obj.sex_type.value],
                        "contains": bool(filter_obj.contains),
                    }
                }
            }
        }

    if isinstance(filter_obj, GQLStrokeFilter):
        return {
            "leaf": {
                "enumCaseFilter": {
                    "strokeType": {
                        "values": [filter_obj.stroke_type.value],
                        "contains": bool(filter_obj.contains),
                    }
                }
            }
        }

    if isinstance(filter_obj, GQLDateFilter):
        return {
            "leaf": {
                "dateCaseFilter": {
                    "property": filter_obj.property,
                    "operator": filter_obj.operator.value,
                    "value": filter_obj.value,
                }
            }
        }

    raise VisualizationExecutionError(
        user_message=("I could not run the statistical test because one of the cohort filters has an unsupported format. Please refine the request and try again."),
        reason="invalid_statistical_filter",
        code="EXEC_STATS_INVALID_FILTER",
    )


def _has_distinct_metric_cohorts(metric_a: Optional[Any], metric_b: Optional[Any]) -> bool:
    if metric_a is None or metric_b is None:
        return False

    metric_a_origin = getattr(metric_a, "data_origin", None)
    metric_b_origin = getattr(metric_b, "data_origin", None)
    metric_a_scope = getattr(metric_a, "origin_scope", None)
    metric_b_scope = getattr(metric_b, "origin_scope", None)

    if metric_a_origin is None or metric_b_origin is None:
        return False

    origin_a_payload = metric_a_origin.model_dump(by_alias=True, exclude_none=True)
    origin_b_payload = metric_b_origin.model_dump(by_alias=True, exclude_none=True)
    if origin_a_payload != origin_b_payload:
        return True

    if metric_a_scope is None or metric_b_scope is None:
        return False

    scope_a_payload = metric_a_scope.model_dump(by_alias=True, exclude_none=True)
    scope_b_payload = metric_b_scope.model_dump(by_alias=True, exclude_none=True)
    if scope_a_payload != scope_b_payload:
        return True

    label_a = getattr(metric_a_scope, "label", None)
    label_b = getattr(metric_b_scope, "label", None)
    if isinstance(label_a, str) and isinstance(label_b, str):
        return bool(label_a.strip() and label_b.strip() and label_a.strip() != label_b.strip())

    return False


def _translate_mann_whitney_metrics(metrics: List[Any], trace_id: str) -> List[str]:
    # Convert planner metric codes to backend enum values once for all MW paths.
    metric_values = [metric.metric for metric in metrics if metric.metric.strip()]
    if not metric_values:
        raise VisualizationExecutionError(
            user_message=("I can run Mann-Whitney U only when a metric is explicitly provided. Please specify the metric to compare."),
            reason="missing_statistical_metric",
            code="EXEC_STATS_MISSING_METRIC",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    stats_enum_map = get_statistics_metric_enum_map()
    translated_metrics: List[str] = []
    ineligible: List[str] = []
    for mv in metric_values:
        gql_name = stats_enum_map.get(mv)
        if gql_name is None:
            ineligible.append(mv)
        else:
            translated_metrics.append(gql_name)
    if ineligible:
        raise VisualizationExecutionError(
            user_message=(f"I cannot run Mann-Whitney U for unsupported metric(s): {', '.join(ineligible)}. Please choose a supported metric."),
            reason="ineligible_statistical_metric",
            code="EXEC_STATS_INELIGIBLE_METRIC",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    return translated_metrics


def _label_from_date_bounds(start_date: Optional[str], end_date: Optional[str], fallback: str) -> str:
    if not start_date or not end_date:
        return fallback
    quarter_lookup = {
        ("01-01", "03-31"): "Q1",
        ("04-01", "06-30"): "Q2",
        ("07-01", "09-30"): "Q3",
        ("10-01", "12-31"): "Q4",
    }
    if len(start_date) >= 10 and len(end_date) >= 10 and start_date[:4] == end_date[:4]:
        quarter = quarter_lookup.get((start_date[5:10], end_date[5:10]))
        if quarter is not None:
            return f"{quarter} {start_date[:4]}"
    return f"{start_date} to {end_date}"


def _execute_mann_whitney_query(
    *,
    metric_values: List[str],
    user_sub: str,
    trace_id: str,
    label_a: str,
    job_id: Optional[str] = None,
    label_b: str,
    data_origin_payload_a: Dict[str, Any],
    data_origin_payload_b: Dict[str, Any],
    time_period_payload_a: Dict[str, str],
    time_period_payload_b: Dict[str, str],
    cohort_filter_a: Optional[Any],
    cohort_filter_b: Optional[Any],
) -> List[StatisticalTestResult]:
    # Shared GraphQL execution path for both standard and temporal MW comparisons.
    query = """
query MannWhitney($metric: [StatisticsMetricEnum!]!, $cohortA: CohortFilterInput!, $cohortB: CohortFilterInput!) {
  getMannWhitneyUTest(metric: $metric, cohortA: $cohortA, cohortB: $cohortB) {
    metric
    uStatistic
    pValue
    significant
    cohortA { size median }
    cohortB { size median }
  }
}
    """.strip()

    variables: Dict[str, Any] = {
        "metric": metric_values,
        "cohortA": {
            "dataOrigin": data_origin_payload_a,
            "timePeriod": time_period_payload_a,
            "caseFilter": _serialize_case_filter_input(cohort_filter_a),
        },
        "cohortB": {
            "dataOrigin": data_origin_payload_b,
            "timePeriod": time_period_payload_b,
            "caseFilter": _serialize_case_filter_input(cohort_filter_b),
        },
    }

    payload = client.query_raw(
        query_str=query,
        user_sub=user_sub,
        job_id=job_id,
        variables=variables,
        trace_id=trace_id,
        raise_on_error=False,
    )
    if payload is None:
        return [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="skipped",
                reason="Mann-Whitney endpoint returned no payload",
                title="Mann-Whitney U Test: skipped",
                details={"trace_id": trace_id},
            )
        ]

    payload_errors_any = payload.get("errors")
    if isinstance(payload_errors_any, list) and payload_errors_any:
        payload_errors = cast(List[Any], payload_errors_any)
        first_error = payload_errors[0]
        if isinstance(first_error, dict):
            first_error_dict = cast(Dict[str, Any], first_error)
            error_message = str(first_error_dict.get("message") or "GraphQL returned an error")
        else:
            error_message = str(first_error)
        return [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="error",
                reason=error_message,
                title="Mann-Whitney U Test: error",
                details={"trace_id": trace_id},
            )
        ]

    data = _mapping_to_dict(payload.get("data"))
    if not data:
        return [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="skipped",
                reason="Mann-Whitney endpoint returned empty data",
                title="Mann-Whitney U Test: skipped",
                details={"trace_id": trace_id},
            )
        ]

    rows_raw: Any = data.get("getMannWhitneyUTest")
    if isinstance(rows_raw, dict):
        rows_raw = [rows_raw]
    if not isinstance(rows_raw, list):
        return [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="error",
                reason="Mann-Whitney response shape is invalid",
                title="Mann-Whitney U Test: error",
                details={"trace_id": trace_id},
            )
        ]

    out: List[StatisticalTestResult] = []
    rows = cast(List[object], rows_raw)
    for row_any in rows:
        if not isinstance(row_any, dict):
            continue
        row = cast(Dict[str, Any], row_any)

        p_value_any = row.get("pValue")
        p_value: Optional[float]
        if isinstance(p_value_any, (int, float)):
            p_value = float(p_value_any)
        else:
            p_value = None

        u_stat_any = row.get("uStatistic")
        u_stat = float(u_stat_any) if isinstance(u_stat_any, (int, float)) else None
        significant_any = row.get("significant")
        significant = bool(significant_any) if isinstance(significant_any, bool) else None

        cohort_a = cast(Dict[str, Any], row.get("cohortA") or {})
        cohort_b = cast(Dict[str, Any], row.get("cohortB") or {})

        metric_name = row.get("metric")
        metric_label = str(metric_name) if isinstance(metric_name, str) else "UNKNOWN"

        out.append(
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="success",
                p_value=p_value,
                passed=significant,
                title=f"Mann-Whitney U Test: {metric_label}",
                details={
                    "trace_id": trace_id,
                    "metric": metric_label,
                    "u_statistic": u_stat,
                    "cohort_a_label": label_a,
                    "cohort_b_label": label_b,
                    "cohort_a_size": cohort_a.get("size"),
                    "cohort_b_size": cohort_b.get("size"),
                    "cohort_a_median": cohort_a.get("median"),
                    "cohort_b_median": cohort_b.get("median"),
                },
            )
        )

    if not out:
        raise VisualizationExecutionError(
            message="I could not find any providers in one or both cohorts for that Mann-Whitney U test.",
            code="EXEC_STATS_EMPTY_COHORT",
            reason="empty_statistical_cohort",
            clarification_type="analysis_plan",
            clarification_options=[],
            user_message=("I could not find any providers in one or both cohorts for that Mann-Whitney U test. Please choose different cohorts or a broader scope."),
            trace_id=trace_id,
        )

    return out


def _can_pair_temporal_mann_whitney_tests(test_a: StatisticalTestSpec, test_b: StatisticalTestSpec) -> bool:
    if test_a.group_by is not None or test_b.group_by is not None:
        return False
    if test_a.filters is None or test_b.filters is None:
        return False

    metrics_a = [m.metric.strip().upper() for m in (test_a.metrics or []) if m.metric and m.metric.strip()]
    metrics_b = [m.metric.strip().upper() for m in (test_b.metrics or []) if m.metric and m.metric.strip()]
    if not metrics_a or not metrics_b:
        return False
    return metrics_a == metrics_b


def _execute_temporal_pair_mann_whitney(
    test_a: StatisticalTestSpec,
    test_b: StatisticalTestSpec,
    user_sub: str,
    trace_id: str,
    job_id: Optional[str] = None,
) -> List[StatisticalTestResult]:
    metrics = list(test_a.metrics or [])
    metric_values = _translate_mann_whitney_metrics(metrics, trace_id)

    metric_a = metrics[0] if len(metrics) > 0 else None
    metric_b = metrics[1] if len(metrics) > 1 else None
    shared_origin = (metric_a.data_origin if metric_a is not None else None) or (metric_b.data_origin if metric_b is not None else None)
    if shared_origin is None:
        raise VisualizationExecutionError(
            user_message=("Please specify both cohorts explicitly for the Mann-Whitney U test, including the hospitals or groups you want to compare."),
            reason="missing_statistical_cohorts",
            code="EXEC_STATS_MISSING_COHORTS",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )
    data_origin_payload = cast(Any, shared_origin).model_dump(by_alias=True, exclude_none=True)

    filter_a = to_gql_filter(test_a.filters)
    filter_b = to_gql_filter(test_b.filters)

    start_a, end_a = _collect_date_bounds(filter_a)
    start_b, end_b = _collect_date_bounds(filter_b)
    label_a = _label_from_date_bounds(start_a, end_a, "Cohort A")
    label_b = _label_from_date_bounds(start_b, end_b, "Cohort B")

    shared_start_candidates = [candidate for candidate in [start_a, start_b] if candidate]
    shared_end_candidates = [candidate for candidate in [end_a, end_b] if candidate]
    if not shared_start_candidates or not shared_end_candidates:
        raise VisualizationExecutionError(
            user_message=("Temporal Mann-Whitney U requires explicit start and end dates for both cohorts."),
            reason="missing_statistical_date_bounds",
            code="EXEC_STATS_MISSING_DATE_BOUNDS",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )
    shared_time_period = {
        "startDate": min(shared_start_candidates),
        "endDate": max(shared_end_candidates),
    }

    results = _execute_mann_whitney_query(
        metric_values=metric_values,
        user_sub=user_sub,
        job_id=job_id,
        trace_id=trace_id,
        label_a=label_a,
        label_b=label_b,
        data_origin_payload_a=data_origin_payload,
        data_origin_payload_b=data_origin_payload,
        time_period_payload_a=shared_time_period,
        time_period_payload_b=shared_time_period,
        cohort_filter_a=filter_a,
        cohort_filter_b=filter_b,
    )
    return results


def _cohort_viability_status_for_result(result: StatisticalTestResult) -> Optional[str]:
    if result.status == "success":
        details = result.details or {}
        cohort_a_size = details.get("cohort_a_size")
        cohort_b_size = details.get("cohort_b_size")

        try:
            a_size = int(cohort_a_size)
        except Exception:
            a_size = 0
        try:
            b_size = int(cohort_b_size)
        except Exception:
            b_size = 0

        if a_size <= 0 or b_size <= 0:
            return "empty"
        return "ok"

    if result.status == "error":
        reason_text = (result.reason or "").lower()
        if "no permission" in reason_text or "not authorized" in reason_text or "unauthorized" in reason_text:
            return "unauthorized"
        return "error"

    return "empty"


def _ensure_mann_whitney_cohort_viability(results: List[StatisticalTestResult], trace_id: str) -> List[StatisticalTestResult]:
    if not results:
        raise VisualizationExecutionError(
            user_message=("I could not find data in one or both cohorts for that Mann-Whitney U test. Please choose different cohorts or broaden the date range."),
            reason="empty_statistical_cohort",
            code="EXEC_STATS_EMPTY_COHORT",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    unauthorized = False
    has_viable_success = False
    for result in results:
        status = _cohort_viability_status_for_result(result)
        if status == "unauthorized":
            unauthorized = True
        if status == "ok":
            has_viable_success = True

    if unauthorized and not has_viable_success:
        raise VisualizationExecutionError(
            user_message=("I cannot access one or both selected cohorts for this Mann-Whitney U test. Please choose cohorts you have permission to query."),
            reason="unauthorized_statistical_cohort",
            code="EXEC_STATS_UNAUTHORIZED_COHORT",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    if not has_viable_success:
        raise VisualizationExecutionError(
            user_message=("I could not find data in one or both selected cohorts for this Mann-Whitney U test. Please broaden the date range or pick different cohorts."),
            reason="empty_statistical_cohort",
            code="EXEC_STATS_EMPTY_COHORT",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    return results


def _statistical_result_signature(result: StatisticalTestResult) -> tuple[Any, ...]:
    details = result.details or {}
    return (
        result.test_type,
        result.status,
        result.reason,
        result.title,
        result.p_value,
        result.passed,
        details.get("metric"),
        details.get("cohort_a_label"),
        details.get("cohort_b_label"),
        details.get("cohort_a_size"),
        details.get("cohort_b_size"),
        details.get("cohort_a_median"),
        details.get("cohort_b_median"),
        details.get("u_statistic"),
    )


def _dedupe_statistical_results(
    results: List[StatisticalTestResult],
) -> List[StatisticalTestResult]:
    deduped: List[StatisticalTestResult] = []
    seen: set[tuple[Any, ...]] = set()
    for result in results:
        signature = _statistical_result_signature(result)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(result)
    return deduped


def _execute_mann_whitney_test(test: StatisticalTestSpec, user_sub: str, trace_id: str, job_id: Optional[str] = None) -> List[StatisticalTestResult]:
    base_filter = to_gql_filter(test.filters)

    metrics = test.metrics or []
    metric_values = _translate_mann_whitney_metrics(metrics, trace_id)

    # Preferred path: explicitly scoped metric pair (hospital-vs-hospital,
    # hospital-vs-national, etc.) where first two metric entries define cohorts.
    metric_a = metrics[0] if len(metrics) > 0 else None
    metric_b = metrics[1] if len(metrics) > 1 else None
    metric_a_origin = metric_a.data_origin if metric_a is not None else None
    metric_b_origin = metric_b.data_origin if metric_b is not None else None
    metric_a_scope = metric_a.origin_scope if metric_a is not None else None
    metric_b_scope = metric_b.origin_scope if metric_b is not None else None

    data_origin_payload_a: Optional[Dict[str, Any]] = None
    data_origin_payload_b: Optional[Dict[str, Any]] = None
    label_a = "Cohort A"
    label_b = "Cohort B"
    cohort_filter_a = base_filter
    cohort_filter_b = base_filter

    if _has_distinct_metric_cohorts(metric_a, metric_b):
        origin_a = cast(Any, metric_a_origin)
        origin_b = cast(Any, metric_b_origin)
        data_origin_payload_a = origin_a.model_dump(by_alias=True, exclude_none=True)
        data_origin_payload_b = origin_b.model_dump(by_alias=True, exclude_none=True)
        if metric_a_scope is not None and metric_a_scope.label and metric_a_scope.label.strip():
            label_a = metric_a_scope.label.strip()
        if metric_b_scope is not None and metric_b_scope.label and metric_b_scope.label.strip():
            label_b = metric_b_scope.label.strip()
    else:
        raise VisualizationExecutionError(
            user_message=("Mann-Whitney U requires two explicit cohorts. Please provide both cohort A and cohort B with distinct scopes."),
            reason="missing_statistical_cohorts",
            code="EXEC_STATS_MISSING_COHORTS",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )

    if data_origin_payload_a is None or data_origin_payload_b is None:
        raise VisualizationExecutionError(
            user_message=("Please provide explicit cohort origins for both cohorts, such as provider IDs or provider-group IDs."),
            reason="missing_statistical_cohorts",
            code="EXEC_STATS_MISSING_COHORTS",
            trace_id=trace_id,
            clarification_type="analysis_plan",
            clarification_options=[],
        )
    start_bound, end_bound = _collect_date_bounds(base_filter)
    if start_bound is None or end_bound is None:
        # Mirror TimePeriod's implicit default window (see default_time_bounds) so a
        # Mann-Whitney request without explicit dates behaves the same as a chart
        # request without explicit dates, instead of rejecting outright.
        start_bound, end_bound = default_time_bounds()
    time_period_payload = {"startDate": start_bound, "endDate": end_bound}
    results = _execute_mann_whitney_query(
        metric_values=metric_values,
        user_sub=user_sub,
        job_id=job_id,
        trace_id=trace_id,
        label_a=label_a,
        label_b=label_b,
        data_origin_payload_a=data_origin_payload_a,
        data_origin_payload_b=data_origin_payload_b,
        time_period_payload_a=time_period_payload,
        time_period_payload_b=time_period_payload,
        cohort_filter_a=cohort_filter_a,
        cohort_filter_b=cohort_filter_b,
    )
    return _ensure_mann_whitney_cohort_viability(results, trace_id)


def _execute_statistical_tests(plan: AnalysisPlan, user_sub: str, trace_id: str, job_id: Optional[str] = None) -> List[StatisticalTestResult]:
    tests = plan.statistical_tests or []
    results: List[StatisticalTestResult] = []

    index = 0
    while index < len(tests):
        test = tests[index]
        test_type = (test.test_type or "").upper().strip()
        if test_type == "MANN_WHITNEY_U_TEST":
            if index + 1 < len(tests):
                next_test = tests[index + 1]
                next_type = (next_test.test_type or "").upper().strip()
                if next_type == "MANN_WHITNEY_U_TEST" and _can_pair_temporal_mann_whitney_tests(test, next_test):
                    results.extend(
                        _execute_temporal_pair_mann_whitney(
                            test_a=test,
                            test_b=next_test,
                            user_sub=user_sub,
                            job_id=job_id,
                            trace_id=trace_id,
                        )
                    )
                    index += 2
                    continue
            results.extend(_execute_mann_whitney_test(test=test, user_sub=user_sub, job_id=job_id, trace_id=trace_id))
        else:
            raise VisualizationExecutionError(
                user_message=("I cannot run the requested statistical test type yet. Please use a supported test type."),
                reason="unsupported_statistical_test_type",
                code="EXEC_STATS_UNSUPPORTED_TEST_TYPE",
                trace_id=trace_id,
                clarification_type="analysis_plan",
                clarification_options=[],
            )
        index += 1

    return results


def _validate_statistical_tests_readiness(plan: AnalysisPlan, trace_id: str) -> None:
    tests = list(plan.statistical_tests or [])
    if not tests:
        return

    index = 0
    while index < len(tests):
        test = tests[index]
        test_type = (test.test_type or "").upper().strip()
        if test_type != "MANN_WHITNEY_U_TEST":
            index += 1
            continue

        if index + 1 < len(tests):
            next_test = tests[index + 1]
            next_type = (next_test.test_type or "").upper().strip()
            if next_type == "MANN_WHITNEY_U_TEST" and _can_pair_temporal_mann_whitney_tests(test, next_test):
                index += 2
                continue

        metrics = list(test.metrics or [])
        if len(metrics) < 2:
            raise VisualizationExecutionError(
                user_message=("I can run Mann-Whitney U only when you provide two explicit cohorts to compare. Please specify both cohort A and cohort B."),
                reason="missing_statistical_cohorts",
                code="EXEC_STATS_MISSING_COHORTS",
                trace_id=trace_id,
                clarification_type="analysis_plan",
                clarification_options=[],
            )

        first = metrics[0]
        second = metrics[1]
        if not _has_distinct_metric_cohorts(first, second):
            raise VisualizationExecutionError(
                user_message=("Mann-Whitney U requires two distinct cohorts. Please provide different cohort filters or scopes for each comparison group."),
                reason="missing_statistical_cohorts",
                code="EXEC_STATS_MISSING_COHORTS",
                trace_id=trace_id,
                clarification_type="analysis_plan",
                clarification_options=[],
            )

        index += 1


def _emit_compiler_diagnostics(progress_cb: Optional[Callable[[str], None]], payload: Dict[str, Any], trace_id: str) -> None:
    log_context_fields: Dict[str, Any] = {
        "trace_id": trace_id,
        "event": "plan_executor.compiler_diagnostics",
        "operation": "_emit_compiler_diagnostics",
        "outcome": "info",
    }
    log_context_fields.update(payload)
    logger.debug(
        "[plan_executor] Compiler diagnostics emitted",
        extra={"log_context": log_context_fields},
    )
    if progress_cb is not None:
        progress_cb(f"Compiler diagnostics: {json.dumps(payload, default=str, sort_keys=True)}")


class VisualizationExecutionError(RuntimeError):
    """Structured execution error surfaced from Action executor paths.

    Field contract:
    - ``reason`` is a stable internal classifier for diagnostics and branching.
    - ``code`` is a stable machine-readable token used by downstream handlers.
    - ``user_message`` is clinician-facing guidance and must be actionable.

    Statistical execution codes used in this module:
    - ``EXEC_STATS_MISSING_METRIC``: no metric provided for Mann-Whitney U.
    - ``EXEC_STATS_INELIGIBLE_METRIC``: requested metric is unsupported for the test.
    - ``EXEC_STATS_MISSING_COHORTS``: two explicit and distinct cohorts were not provided.
    - ``EXEC_STATS_MISSING_DATE_BOUNDS``: explicit start/end dates are missing.
    - ``EXEC_STATS_EMPTY_COHORT``: one or both cohorts have no usable data.
    - ``EXEC_STATS_UNAUTHORIZED_COHORT``: cohort access is not authorized.
    - ``EXEC_STATS_QUERY_FAILED``: upstream statistical query failed unexpectedly.
    - ``EXEC_STATS_INVALID_FILTER``: cohort filter shape is unsupported.
    """

    def __init__(
        self,
        user_message: str,
        reason: str = "unknown",
        code: str = "EXEC_UNKNOWN",
        trace_id: Optional[str] = None,
        clarification_type: Optional[str] = None,
        clarification_options: Optional[List[str]] = None,
    ):
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason
        self.code = code
        self.trace_id = trace_id
        self.clarification_type = clarification_type
        self.clarification_options = list(clarification_options or [])


def _invalid_plan_semantics_error(exc: Exception, trace_id: str) -> VisualizationExecutionError:
    return VisualizationExecutionError(
        user_message=("I could not apply the requested grouping for this chart. Please try grouping by stroke type, sex, age, NIHSS, or a time period."),
        reason="invalid_plan_semantics",
        code="EXEC_PLAN_INVALID_SEMANTICS",
        trace_id=trace_id,
        clarification_type="analysis_plan",
        clarification_options=[],
    )


def _to_execution_error(failure_reasons: List[str], trace_id: Optional[str] = None) -> VisualizationExecutionError:
    reason_set = set(failure_reasons)
    service_unavailable_count = sum(1 for reason in failure_reasons if reason == "service_unavailable")
    if "no_data" in reason_set:
        return VisualizationExecutionError(
            user_message="The analytics service returned no data for this visualization request. Try a wider date range or different filters.",
            reason="no_data",
            code="EXEC_NO_DATA",
            trace_id=trace_id,
        )
    if "timeout" in reason_set:
        return VisualizationExecutionError(
            user_message="The data service is currently down. Please try again in a few minutes.",
            reason="timeout",
            code="EXEC_TIMEOUT",
            trace_id=trace_id,
        )
    if "service_unavailable" in reason_set:
        if service_unavailable_count >= 2:
            return VisualizationExecutionError(
                user_message=("The analytics platform appears to be experiencing an outage right now (upstream service unavailable). Please try again in a moment."),
                reason="service_unavailable",
                code="EXEC_SERVICE_UNAVAILABLE",
                trace_id=trace_id,
            )
        return VisualizationExecutionError(
            user_message="The analytics service is temporarily unavailable. Please try again in a moment.",
            reason="service_unavailable",
            code="EXEC_SERVICE_UNAVAILABLE",
            trace_id=trace_id,
        )
    if "graphql_error" in reason_set:
        return VisualizationExecutionError(
            user_message="The analytics service returned an error while generating this visualization. Please adjust the filters or try again.",
            reason="graphql_error",
            code="EXEC_GRAPHQL_ERROR",
            trace_id=trace_id,
        )
    if "upstream_error" in reason_set:
        return VisualizationExecutionError(
            user_message="The analytics service is currently unreachable. Please try again in a moment.",
            reason="upstream_error",
            code="EXEC_UPSTREAM_ERROR",
            trace_id=trace_id,
        )
    return VisualizationExecutionError(
        user_message="I could not fetch analytics data for this visualization. Please adjust the filters or try again.",
        reason="data_fetch_failed",
        code="EXEC_DATA_FETCH_FAILED",
        trace_id=trace_id,
    )


def _format_iso_date(value: str) -> str:
    token = (value or "").strip()
    if not token:
        raise VisualizationExecutionError(
            user_message="I could not parse a required date for this request.",
            reason="invalid_time_period",
            code="EXEC_INVALID_TIME_PERIOD",
        )
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VisualizationExecutionError(
            user_message=("I could not parse a required date for this request. Please provide dates in ISO format (YYYY-MM-DD)."),
            reason="invalid_time_period",
            code="EXEC_INVALID_TIME_PERIOD",
        ) from exc
    return parsed.date().isoformat()


def _parse_iso_date(value: str) -> datetime:
    token = (value or "").strip()
    if not token:
        raise VisualizationExecutionError(
            user_message="I could not parse a required date for this request.",
            reason="invalid_time_period",
            code="EXEC_INVALID_TIME_PERIOD",
        )
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VisualizationExecutionError(
            user_message=("I could not parse a required date for this request. Please provide dates in ISO format (YYYY-MM-DD)."),
            reason="invalid_time_period",
            code="EXEC_INVALID_TIME_PERIOD",
        ) from exc


def _sampled_period_from_specs(specs: List[RequestSpec]) -> Optional[str]:
    starts: List[tuple[datetime, str]] = []
    ends: List[tuple[datetime, str]] = []

    for spec in specs:
        time_period_any = spec.req.time_period
        periods: List[TimePeriod]
        if isinstance(time_period_any, list):
            periods = list(time_period_any)
        else:
            periods = [time_period_any]

        for period in periods:
            start_raw = cast(Optional[str], getattr(period, "start_date", None))
            end_raw = cast(Optional[str], getattr(period, "end_date", None))

            if isinstance(start_raw, str):
                parsed = _parse_iso_date(start_raw)
                shown = _format_iso_date(start_raw)
                if shown:
                    starts.append((parsed, shown))

            if isinstance(end_raw, str):
                parsed = _parse_iso_date(end_raw)
                shown = _format_iso_date(end_raw)
                if shown:
                    ends.append((parsed, shown))

    start_text = min(starts, key=lambda item: item[0])[1] if starts else None
    end_text = max(ends, key=lambda item: item[0])[1] if ends else None

    if start_text and end_text:
        if start_text == end_text:
            return start_text
        return f"{start_text} to {end_text}"
    if start_text:
        return f"{start_text} onward"
    if end_text:
        return f"up to {end_text}"
    return None


def execute_plan(plan: AnalysisPlan, user_sub: str, job_id: Optional[str] = None) -> VisualizationResponse:
    """Sync wrapper that delegates to the async implementation with concurrency=1.

    - If no event loop is running, run the coroutine directly with asyncio.run.
    - If an event loop is already running, offload to a new thread and run a fresh loop there.
    """
    trace_id = uuid4().hex
    coro = execute_plan_async(
        plan,
        user_sub,
        job_id=job_id,
        max_concurrency=_EXECUTOR_SYNC_MAX_CONCURRENCY,
        trace_id=trace_id,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(bind_current_context(asyncio.run), coro)
        return fut.result()


ProgressCallback = Callable[[str], None]
SummaryCallback = Callable[[ExecutionSummary], None]
QueryDebugCallback = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class ExecutionContext:
    user_sub: str
    job_id: Optional[str]
    semaphore: asyncio.Semaphore
    progress_cb: Optional[ProgressCallback]
    log_graphql_query: bool
    query_cb: Optional[QueryDebugCallback]


@dataclass(frozen=True)
class RequestExecutionResult:
    spec: RequestSpec
    series: List[ChartSeries]


def _emit_progress(context: ExecutionContext, completed: int, total: int, prefix: str = "Fetching data") -> None:
    if context.progress_cb is None:
        return
    if total > 0:
        context.progress_cb(f"{prefix} ({completed}/{total})")
    else:
        context.progress_cb(f"{prefix}…")


def _request_scope_label(spec: RequestSpec) -> str:
    if spec.scope_label and spec.scope_label.strip():
        return spec.scope_label.strip()
    if spec.label_parts:
        joined = " - ".join([part for part in spec.label_parts if part.strip()])
        if joined:
            return joined
    return "one requested scope"


async def _execute_request_spec(
    spec: RequestSpec,
    request_failures: List[str],
    request_warnings: List[str],
    context: ExecutionContext,
    trace_id: str,
) -> RequestExecutionResult:
    series = await run_graphql_request(
        req=spec.req,
        label_parts=spec.label_parts,
        include_metric_alias=spec.include_metric_alias,
        group_by_field=spec.group_by_field,
        add_time_period_labels=spec.add_time_period_labels,
        scope_label=spec.scope_label,
        request_failures=request_failures,
        client=client,
        user_sub=context.user_sub,
        job_id=context.job_id,
        trace_id=trace_id,
        semaphore=context.semaphore,
        log_graphql_query=context.log_graphql_query,
        request_warnings=request_warnings,
        batched_time_periods=spec.batched_time_periods,
        query_cb=context.query_cb,
        is_filter_grouped=spec.is_filter_grouped,
    )
    return RequestExecutionResult(spec=spec, series=series)


async def _execute_specs_concurrent(
    specs: List[RequestSpec],
    request_failures: List[str],
    request_warnings: List[str],
    context: ExecutionContext,
    trace_id: str,
    total_requests: int,
    progress_prefix: str = "Fetching data",
) -> List[RequestExecutionResult]:
    _emit_progress(context, completed=0, total=total_requests, prefix=progress_prefix)
    if not specs:
        return []

    tasks = [
        asyncio.create_task(
            _execute_request_spec(
                spec=spec,
                request_failures=request_failures,
                request_warnings=request_warnings,
                context=context,
                trace_id=trace_id,
            )
        )
        for spec in specs
    ]

    results: List[RequestExecutionResult] = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1
        _emit_progress(context, completed=completed, total=total_requests, prefix=progress_prefix)

    return results


async def _execute_specs_sequential(
    specs: List[RequestSpec],
    request_failures: List[str],
    request_warnings: List[str],
    context: ExecutionContext,
    trace_id: str,
    total_requests: int,
    progress_prefix: str,
) -> List[RequestExecutionResult]:
    _emit_progress(context, completed=0, total=total_requests, prefix=progress_prefix)
    results: List[RequestExecutionResult] = []
    completed = 0

    for spec in specs:
        result = await _execute_request_spec(
            spec=spec,
            request_failures=request_failures,
            request_warnings=request_warnings,
            context=context,
            trace_id=trace_id,
        )
        results.append(result)
        completed += 1
        _emit_progress(context, completed=completed, total=total_requests, prefix=progress_prefix)

    return results


async def execute_plan_async(
    plan: AnalysisPlan,
    user_sub: str,
    job_id: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
    summary_cb: Optional[SummaryCallback] = None,
    trace_id: Optional[str] = None,
    query_cb: Optional[QueryDebugCallback] = None,
) -> VisualizationResponse:
    """Async version that runs GraphQL requests concurrently.

    - Uses asyncio.to_thread to run the existing synchronous client in a thread pool.
    - Limits concurrency via a semaphore to avoid overloading the proxy/backend.
    - Produces one chart per canonical GroupBy (or one overall if none), matching sync behavior.
    """
    trace_id_resolved = (trace_id or "").strip()
    if not trace_id_resolved:
        raise ValueError("trace_id is required for execute_plan_async")

    logger.info(
        "[plan_executor] execute_plan_async start",
        extra={"trace_id": trace_id_resolved},
    )
    logger.info(
        "[plan_executor] resolving plan metric origins",
        extra={"trace_id": trace_id_resolved},
    )

    try:
        # jobId is deliberately not threaded into origin-scope resolution here --
        # that subsystem has its own separate, confirmed authorization gap
        # (see SECURITY-TODO.md cluster 1) that this identity-hardening pass
        # does not touch. Revisit once that's fixed.
        plan = await asyncio.wait_for(
            asyncio.to_thread(
                resolve_plan_metric_origins,
                plan=plan,
                user_sub=user_sub,
                trace_id=trace_id_resolved,
            ),
            timeout=_ORIGIN_SCOPE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise VisualizationExecutionError(
            user_message=("I could not resolve your hospital scope in time. Please try again or specify both cohorts explicitly."),
            reason="origin_scope_resolution_timeout",
            code="EXEC_ORIGIN_SCOPE_TIMEOUT",
            trace_id=trace_id_resolved,
            clarification_type="mine",
            clarification_options=[],
        ) from exc
    except OriginScopeResolutionError as exc:
        raise VisualizationExecutionError(
            user_message=str(exc),
            reason="origin_scope_resolution",
            code="EXEC_ORIGIN_SCOPE",
            trace_id=trace_id_resolved,
            clarification_type=exc.clarification_type,
            clarification_options=exc.clarification_options,
        ) from exc

    logger.info(
        "[plan_executor] resolved plan metric origins",
        extra={
            "trace_id": trace_id_resolved,
            "charts_count": len(plan.charts or []),
            "statistical_tests_count": len(plan.statistical_tests or []),
        },
    )

    normalization_summary = None
    _validate_statistical_tests_readiness(plan=plan, trace_id=trace_id_resolved)

    plan_charts = coalesce(plan.charts, [])
    response: VisualizationResponse = VisualizationResponse(trace_id=trace_id_resolved)
    try:
        estimated_queries = estimate_query_count_for_plan(plan)
    except ValueError as exc:
        raise _invalid_plan_semantics_error(exc, trace_id=trace_id_resolved) from exc
    actual_queries = 0
    summary_batches: List[ExecutionBatchSummary] = []

    resolved_concurrency = _EXECUTOR_DEFAULT_MAX_CONCURRENCY if max_concurrency is None else max(1, int(max_concurrency))
    sem = asyncio.Semaphore(resolved_concurrency)
    execution_context = ExecutionContext(
        user_sub=user_sub,
        job_id=job_id,
        semaphore=sem,
        progress_cb=progress_cb,
        log_graphql_query=_LOG_GRAPHQL_QUERY,
        query_cb=query_cb,
    )

    for planChart in plan_charts:
        logger.debug(
            "[plan_executor] building metric requests for chart",
            extra={
                "trace_id": trace_id_resolved,
                "chart_type": planChart.chart_type,
                "metric_count": len(planChart.metrics),
            },
        )
        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = build_metric_requests(plan_chart=planChart)

        try:
            compiled_grouping = compile_chart_grouping(planChart)
        except ValueError as exc:
            raise _invalid_plan_semantics_error(exc, trace_id=trace_id_resolved) from exc
        dims: List[Dimension] = compiled_grouping.dimensions

        for batch in compiled_grouping.batches:
            request_failures: List[str] = []
            request_warnings: List[str] = []
            filter_dims = batch.filter_dims
            batched_time_enabled = batch.batched_time_enabled
            batched_time_periods = batch.batched_time_periods
            combos_list = batch.combos_list

            gb_field = batch.server_groupby
            include_metric_alias = len(planChart.metrics) > 1

            chart_filter = to_gql_filter(coalesce(planChart.filters, None))

            primary_specs = build_primary_request_specs(
                metric_requests=metric_requests,
                metric_data_origins=metric_data_origins,
                chart_filter=chart_filter,
                filter_dims=filter_dims,
                combos_list=combos_list,
                batched_time_enabled=batched_time_enabled,
                batched_time_periods=batched_time_periods,
                include_metric_alias=include_metric_alias,
                group_by_field=gb_field,
                metric_scope_labels=metric_scope_labels,
                include_general_stats=_INCLUDE_GENERAL_STATS,
            )
            total_requests = max(1, len(primary_specs))
            actual_queries += total_requests

            summary_batches.append(
                make_batch_summary(
                    chart_title=f"{(planChart.chart_type or 'CHART').upper()} chart",
                    chart_type=planChart.chart_type,
                    server_groupby=gb_field,
                    filter_dimensions=[d.kind.__name__ for d in filter_dims],
                    batched_time_period_count=len(batched_time_periods) if batched_time_enabled else 0,
                    query_count=total_requests,
                )
            )

            if _EMIT_COMPILER_DIAGNOSTICS:
                _emit_compiler_diagnostics(
                    progress_cb,
                    {
                        "chart_title": f"{(planChart.chart_type or 'CHART').upper()} chart",
                        "chart_type": planChart.chart_type,
                        "server_groupby": gb_field,
                        "batched_time_enabled": batched_time_enabled,
                        "batched_time_period_count": len(batched_time_periods),
                        "filter_dimensions": [d.kind.__name__ for d in filter_dims],
                        "query_count_estimate": batch.request_count,
                        "query_count_planned": total_requests,
                    },
                    trace_id=trace_id_resolved,
                )

            request_results = await _execute_specs_concurrent(
                specs=primary_specs,
                request_failures=request_failures,
                request_warnings=request_warnings,
                context=execution_context,
                trace_id=trace_id_resolved,
                total_requests=total_requests,
                progress_prefix="Fetching data",
            )
            all_series = [item for result in request_results for item in result.series]

            sampled_period_override = _sampled_period_from_specs(primary_specs)

            # A scope that came back with no rows (a genuinely empty result, or a
            # GraphQL partial error such as an unsupported filter value on just
            # one scope) is dropped rather than failing the entire chart -- the
            # other requested scopes still render. Only total failure (every
            # scope empty) is fatal; see the `if not all_series` check below.
            empty_scope_labels = [_request_scope_label(result.spec) for result in request_results if not result.series]
            if empty_scope_labels:
                missing_labels = ", ".join(sorted(set(empty_scope_labels)))
                request_warnings.append(f"No data returned for: {missing_labels}. This scope was omitted from the chart.")

            for warning_msg in request_warnings:
                if warning_msg not in response.warnings:
                    logger.debug(
                        "[plan_executor] Appending request warning",
                        extra={
                            "log_context": {
                                "trace_id": trace_id_resolved,
                                "event": "plan_executor.warning.request_warning_appended",
                                "operation": "execute_plan_async",
                                "outcome": "degraded",
                                "chart_type": planChart.chart_type or "Chart",
                                "warning_text": warning_msg,
                            }
                        },
                    )
                    response.warnings.append(warning_msg)

            all_series = merge_series_by_name(all_series)

            if not all_series:
                logger.warning(
                    "[plan_executor] No series generated for chart '%s'%s. This often indicates a backend error or empty results.",
                    planChart.chart_type or "Chart",
                    "",
                    extra={
                        "log_context": {
                            "trace_id": trace_id_resolved,
                            "event": "plan_executor.chart.no_series_generated",
                            "operation": "execute_plan_async",
                            "outcome": "degraded",
                            "chart_type": planChart.chart_type or "Chart",
                            "request_failure_count": len(request_failures),
                            "warning_count": len(request_warnings),
                        }
                    },
                )
                if request_failures:
                    raise _to_execution_error(request_failures, trace_id=trace_id_resolved)
                raise _to_execution_error(["no_data"], trace_id=trace_id_resolved)
            vis_chart = build_chart_dto(
                plan_chart=planChart,
                dimensions=dims,
                series=all_series,
                derived_axes=derived_axes,
                sampled_period_override=sampled_period_override,
            )
            response.charts.append(vis_chart)

    if plan.statistical_tests:
        # Statistical tests issue backend requests outside chart batching; include
        # them in the summary count so stats-only plans are not reported as zero.
        actual_queries += len(plan.statistical_tests)
        response.stats.extend(_dedupe_statistical_results(_execute_statistical_tests(plan=plan, user_sub=user_sub, job_id=job_id, trace_id=trace_id_resolved)))

    stats_count = len(response.stats)
    stats_skipped = sum(1 for result in response.stats if result.status == "skipped")
    stats_errors = sum(1 for result in response.stats if result.status == "error")

    if summary_cb is not None:
        payload = make_execution_summary(
            trace_id=trace_id_resolved,
            chart_count=len(plan_charts),
            stats_count=stats_count,
            stats_skipped=stats_skipped,
            stats_errors=stats_errors,
            estimated_queries=estimated_queries,
            actual_queries=actual_queries,
            batches=summary_batches,
            normalization=normalization_summary,
        )
        summary_cb(payload)

    return response
