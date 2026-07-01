import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, cast
from uuid import uuid4

from src.domain.dto.analytics import StatisticalTestResult
from src.domain.dto.charts.types import ChartAxis, ChartSeries
from src.domain.dto.execution_summary import ExecutionBatchSummary, ExecutionSummary
from src.domain.dto.response import VisualizationResponse
from src.domain.graphql.request import BooleanFilter as GQLBooleanFilter
from src.domain.graphql.request import DataOrigin, TimePeriod
from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.request import IntegerFilter as GQLIntegerFilter
from src.domain.graphql.request import LogicalFilter as GQLLogicalFilter
from src.domain.graphql.request import SexFilter as GQLSexFilter
from src.domain.graphql.request import StrokeFilter as GQLStrokeFilter
from src.domain.langchain.schema import AnalysisPlan, GroupBySpec, StatisticalTestSpec
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
    build_fallback_request_specs,
    build_primary_request_specs,
    should_retry_unbatched_time,
)
from src.executors.transport.request_runner import run_graphql_request
from src.shared.ssot_loader import (
    get_metric_display_name,
    get_metric_metadata,
    get_statistics_metric_enum_map,
)
from src.util import env as env_util
from src.util.coalesce import coalesce
from src.util.logging_utils import bind_current_context

logger = logging.getLogger(__name__)
# Privacy/safety defaults:
# - Avoid logging raw GraphQL queries by default.
_LOG_GRAPHQL_QUERY = env_util.env_flag("EXECUTOR_LOG_GRAPHQL_QUERY", default=False)
_EMIT_COMPILER_DIAGNOSTICS = env_util.env_flag(
    "EXECUTOR_EMIT_COMPILER_DIAGNOSTICS", default=False
)
_ENABLE_UNBATCHED_TIME_FALLBACK = env_util.env_flag(
    "EXECUTOR_ENABLE_UNBATCHED_TIME_FALLBACK", default=True
)
_STRICT_MODE = env_util.env_flag(
    "ANALYTICS_STRICT_MODE", default=False
) or env_util.env_flag("EXECUTOR_STRICT_MODE", default=False)

_executor_default_concurrency_raw = (
    env_util.get_env("EXECUTOR_DEFAULT_MAX_CONCURRENCY", default="4") or "4"
)
try:
    _executor_default_concurrency = max(1, int(_executor_default_concurrency_raw))
except Exception:
    logger.debug(
        "[plan_executor] Invalid EXECUTOR_DEFAULT_MAX_CONCURRENCY; using fallback",
        exc_info=True,
        extra={
            "log_context": {
                "event": "plan_executor.config.default_concurrency_fallback",
                "operation": "module_init",
                "outcome": "degraded",
                "raw_value": _executor_default_concurrency_raw,
                "fallback_value": 4,
            }
        },
    )
    _executor_default_concurrency = 4
_EXECUTOR_DEFAULT_MAX_CONCURRENCY = _executor_default_concurrency

_executor_sync_concurrency_raw = (
    env_util.get_env("EXECUTOR_SYNC_MAX_CONCURRENCY", default="1") or "1"
)
try:
    _executor_sync_concurrency = max(1, int(_executor_sync_concurrency_raw))
except Exception:
    logger.debug(
        "[plan_executor] Invalid EXECUTOR_SYNC_MAX_CONCURRENCY; using fallback",
        exc_info=True,
        extra={
            "log_context": {
                "event": "plan_executor.config.sync_concurrency_fallback",
                "operation": "module_init",
                "outcome": "degraded",
                "raw_value": _executor_sync_concurrency_raw,
                "fallback_value": 1,
            }
        },
    )
    _executor_sync_concurrency = 1
_EXECUTOR_SYNC_MAX_CONCURRENCY = _executor_sync_concurrency

proxy_url, action_server_token = env_util.require_all_env(
    "RASA_PROXY_URL", "ACTION_SERVER_TOKEN"
)
graphql_target = env_util.require_any_env("RASA_PROXY_GRAPHQL_TARGET")

_graphql_timeout_raw = (
    env_util.get_env("EXECUTOR_GRAPHQL_TIMEOUT_SECONDS", default="30") or "30"
)
try:
    _graphql_timeout_seconds = max(5, int(float(_graphql_timeout_raw)))
except Exception:
    _graphql_timeout_seconds = 30

client = GraphQLProxyClient(
    proxy_url=proxy_url,
    action_server_token=action_server_token,
    target=graphql_target
    if isinstance(graphql_target, str) and graphql_target.strip()
    else "graphql",
    timeout_seconds=_graphql_timeout_seconds,
    connect_timeout_seconds=5,
    max_total_timeout_seconds=_graphql_timeout_seconds + 5,
    retry_attempts=1,
    retry_backoff_seconds=0.2,
)


METRIC_METADATA: Dict[str, Any] = get_metric_metadata()

_AXIS_LABEL_OVERRIDES: Dict[str, str] = {
    "DTN": "Door-to-Needle Time",
    "ONSET_TO_DOOR": "Onset-to-Door Time",
    "DOOR_TO_REPERFUSION": "Door-to-Reperfusion Time",
}

_AXIS_UNIT_FALLBACKS: Dict[str, str] = {
    "DTN": "minutes",
    "ONSET_TO_DOOR": "minutes",
    "DOOR_TO_REPERFUSION": "minutes",
}

_AXIS_ACRONYMS = {"NIHSS", "DTN", "IVT", "EVT", "TIA", "LVO", "ICH", "SAH", "CT", "MRI"}


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: Dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _normalize_axis_display_label(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    def _word_case(word: str) -> str:
        token = word.strip()
        if not token:
            return token
        upper = token.upper()
        if upper in _AXIS_ACRONYMS:
            return upper
        if token.isupper() and len(token) <= 4:
            return token
        if token[:1].isdigit():
            return token
        return token[:1].upper() + token[1:].lower()

    out_words: List[str] = []
    for word in text.split():
        if "-" in word:
            out_words.append("-".join(_word_case(part) for part in word.split("-")))
        else:
            out_words.append(_word_case(word))
    return " ".join(out_words)


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            logger.debug(
                "[plan_executor] Failed to parse integer CSV token; skipping token",
                exc_info=True,
                extra={
                    "log_context": {
                        "event": "plan_executor.config.int_csv_token_skipped",
                        "operation": "_parse_int_csv",
                        "outcome": "degraded",
                        "token": token,
                    }
                },
            )
            continue
    return out


_DEFAULT_PROVIDER_GROUP_IDS = _parse_int_csv(
    env_util.get_env("EXECUTOR_DEFAULT_PROVIDER_GROUP_IDS", default="1") or "1"
)
_DEFAULT_PROVIDER_IDS = _parse_int_csv(
    env_util.get_env("EXECUTOR_DEFAULT_PROVIDER_IDS", default="") or ""
)


def _build_default_data_origin() -> DataOrigin:
    provider_group_ids = list(_DEFAULT_PROVIDER_GROUP_IDS)
    provider_ids = list(_DEFAULT_PROVIDER_IDS)
    if not provider_group_ids and not provider_ids:
        provider_group_ids = [1]

    kwargs: Dict[str, Any] = {}
    if provider_group_ids:
        kwargs["providerGroupId"] = provider_group_ids
    if provider_ids:
        kwargs["providerId"] = provider_ids
    return DataOrigin(**kwargs)


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


def _default_time_period_from_filter(filter_obj: Optional[Any]) -> Dict[str, str]:
    start_bound, end_bound = _collect_date_bounds(filter_obj)
    default_tp = TimePeriod()
    return {
        "startDate": start_bound or cast(str, default_tp.start_date),
        "endDate": end_bound or cast(str, default_tp.end_date),
    }


def _merge_case_filters(
    base_filter: Optional[Any], cohort_filter: Optional[Any]
) -> Optional[Any]:
    if base_filter is None:
        return cohort_filter
    if cohort_filter is None:
        return base_filter
    return GQLLogicalFilter(operator="AND", children=[base_filter, cohort_filter])


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

    logger.warning(
        "[plan_executor] Unsupported case filter type for statistical test payload: %s",
        type(filter_obj).__name__,
        extra={
            "log_context": {
                "event": "plan_executor.statistical_test.unsupported_case_filter",
                "operation": "_serialize_case_filter_input",
                "outcome": "degraded",
                "filter_type": type(filter_obj).__name__,
            }
        },
    )
    return None


def _cohort_split_from_groupby(
    group_by: Optional[List[GroupBySpec]],
) -> Optional[tuple[Dimension, Any, Any, str, str]]:
    if not group_by:
        return None

    for spec in group_by:
        dim = Dimension(spec)
        cats = list(dim.categories())
        if not cats:
            continue

        if len(cats) < 2:
            continue

        c_a = cats[0]
        c_b = cats[1]
        f_a = dim.filter_for(c_a)
        f_b = dim.filter_for(c_b)
        if f_a is None or f_b is None:
            continue

        return dim, c_a, c_b, dim.label_for(c_a), dim.label_for(c_b)

    return None


def _has_distinct_metric_cohorts(
    metric_a: Optional[Any], metric_b: Optional[Any]
) -> bool:
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
        return bool(
            label_a.strip() and label_b.strip() and label_a.strip() != label_b.strip()
        )

    return False


def _translate_mann_whitney_metrics(
    metrics: List[Any], trace_id: str
) -> tuple[List[str], Optional[StatisticalTestResult]]:
    # Convert planner metric codes to backend enum values once for all MW paths.
    metric_values = [metric.metric for metric in metrics if metric.metric.strip()]
    if not metric_values:
        return [], StatisticalTestResult(
            test_type="MANN_WHITNEY_U_TEST",
            status="skipped",
            reason="No metric was provided for statistical testing",
            title="Mann-Whitney U Test: skipped",
            details={"trace_id": trace_id},
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
        reason = (
            f"Metric(s) not supported for statistical testing: {', '.join(ineligible)}"
        )
        logger.warning(
            "[plan_executor] Skipping MANN_WHITNEY_U_TEST: %s",
            reason,
            extra={
                "log_context": {
                    "trace_id": trace_id or "-",
                    "event": "plan_executor.statistical_test.skipped_ineligible_metrics",
                    "operation": "_translate_mann_whitney_metrics",
                    "outcome": "degraded",
                    "test_type": "MANN_WHITNEY_U_TEST",
                    "ineligible_metric_count": len(ineligible),
                }
            },
        )
        return [], StatisticalTestResult(
            test_type="MANN_WHITNEY_U_TEST",
            status="skipped",
            reason=reason,
            title="Mann-Whitney U Test: skipped",
            details={"trace_id": trace_id},
        )

    return translated_metrics, None


def _label_from_date_bounds(
    start_date: Optional[str], end_date: Optional[str], fallback: str
) -> str:
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
            error_message = str(
                first_error_dict.get("message") or "GraphQL returned an error"
            )
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
        significant = (
            bool(significant_any) if isinstance(significant_any, bool) else None
        )

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
        return [
            StatisticalTestResult(
                test_type="MANN_WHITNEY_U_TEST",
                status="skipped",
                reason="Mann-Whitney returned no comparable cohort rows",
                title="Mann-Whitney U Test: skipped",
                details={"trace_id": trace_id},
            )
        ]

    return out


def _can_pair_temporal_mann_whitney_tests(
    test_a: StatisticalTestSpec, test_b: StatisticalTestSpec
) -> bool:
    if test_a.group_by is not None or test_b.group_by is not None:
        return False
    if test_a.filters is None or test_b.filters is None:
        return False

    metrics_a = [
        m.metric.strip().upper()
        for m in (test_a.metrics or [])
        if m.metric and m.metric.strip()
    ]
    metrics_b = [
        m.metric.strip().upper()
        for m in (test_b.metrics or [])
        if m.metric and m.metric.strip()
    ]
    if not metrics_a or not metrics_b:
        return False
    return metrics_a == metrics_b


def _execute_temporal_pair_mann_whitney(
    test_a: StatisticalTestSpec,
    test_b: StatisticalTestSpec,
    user_sub: str,
    trace_id: str,
) -> List[StatisticalTestResult]:
    metrics = list(test_a.metrics or [])
    metric_values, metric_error = _translate_mann_whitney_metrics(metrics, trace_id)
    if metric_error is not None:
        return [metric_error]

    metric_a = metrics[0] if len(metrics) > 0 else None
    metric_b = metrics[1] if len(metrics) > 1 else None
    shared_origin = (
        (metric_a.data_origin if metric_a is not None else None)
        or (metric_b.data_origin if metric_b is not None else None)
        or _build_default_data_origin()
    )
    data_origin_payload = cast(Any, shared_origin).model_dump(
        by_alias=True, exclude_none=True
    )

    filter_a = to_gql_filter(test_a.filters)
    filter_b = to_gql_filter(test_b.filters)

    start_a, end_a = _collect_date_bounds(filter_a)
    start_b, end_b = _collect_date_bounds(filter_b)
    label_a = _label_from_date_bounds(start_a, end_a, "Cohort A")
    label_b = _label_from_date_bounds(start_b, end_b, "Cohort B")

    shared_start_candidates = [
        candidate for candidate in [start_a, start_b] if candidate
    ]
    shared_end_candidates = [candidate for candidate in [end_a, end_b] if candidate]
    if shared_start_candidates and shared_end_candidates:
        shared_time_period = {
            "startDate": min(shared_start_candidates),
            "endDate": max(shared_end_candidates),
        }
    else:
        shared_time_period = _default_time_period_from_filter(None)

    return _execute_mann_whitney_query(
        metric_values=metric_values,
        user_sub=user_sub,
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


def _execute_mann_whitney_test(
    test: StatisticalTestSpec, user_sub: str, trace_id: str
) -> List[StatisticalTestResult]:
    base_filter = to_gql_filter(test.filters)

    metrics = test.metrics or []
    metric_values, metric_error = _translate_mann_whitney_metrics(metrics, trace_id)
    if metric_error is not None:
        return [metric_error]

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
        if (
            metric_a_scope is not None
            and metric_a_scope.label
            and metric_a_scope.label.strip()
        ):
            label_a = metric_a_scope.label.strip()
        if (
            metric_b_scope is not None
            and metric_b_scope.label
            and metric_b_scope.label.strip()
        ):
            label_b = metric_b_scope.label.strip()
    else:
        # Backward-compatible fallback: derive cohorts from two-way group_by split.
        cohort_split = _cohort_split_from_groupby(test.group_by)
        if cohort_split is None:
            reason = "Could not determine two distinct cohorts for comparison"
            logger.warning(
                "[plan_executor] Skipping MANN_WHITNEY_U_TEST: %s",
                reason,
                extra={
                    "log_context": {
                        "trace_id": trace_id or "-",
                        "event": "plan_executor.statistical_test.skipped_missing_cohorts",
                        "operation": "_execute_mann_whitney_test",
                        "outcome": "degraded",
                        "test_type": "MANN_WHITNEY_U_TEST",
                    }
                },
            )
            return [
                StatisticalTestResult(
                    test_type="MANN_WHITNEY_U_TEST",
                    status="skipped",
                    reason=reason,
                    title="Mann-Whitney U Test: skipped",
                )
            ]

        dim, cat_a, cat_b, label_a_split, label_b_split = cohort_split
        cohort_filter_a = _merge_case_filters(base_filter, dim.filter_for(cat_a))
        cohort_filter_b = _merge_case_filters(base_filter, dim.filter_for(cat_b))
        shared_origin = (
            metric_a_origin or metric_b_origin or _build_default_data_origin()
        )
        data_origin_payload_default = cast(Any, shared_origin).model_dump(
            by_alias=True, exclude_none=True
        )
        data_origin_payload_a = data_origin_payload_default
        data_origin_payload_b = data_origin_payload_default
        label_a = label_a_split
        label_b = label_b_split

    time_period_payload = _default_time_period_from_filter(base_filter)
    origin_a = data_origin_payload_a or _build_default_data_origin().model_dump(
        by_alias=True, exclude_none=True
    )
    origin_b = data_origin_payload_b or _build_default_data_origin().model_dump(
        by_alias=True, exclude_none=True
    )
    return _execute_mann_whitney_query(
        metric_values=metric_values,
        user_sub=user_sub,
        trace_id=trace_id,
        label_a=label_a,
        label_b=label_b,
        data_origin_payload_a=origin_a,
        data_origin_payload_b=origin_b,
        time_period_payload_a=time_period_payload,
        time_period_payload_b=time_period_payload,
        cohort_filter_a=cohort_filter_a,
        cohort_filter_b=cohort_filter_b,
    )


def _execute_statistical_tests(
    plan: AnalysisPlan, user_sub: str, trace_id: str
) -> List[StatisticalTestResult]:
    tests = plan.statistical_tests or []
    results: List[StatisticalTestResult] = []

    index = 0
    while index < len(tests):
        test = tests[index]
        test_type = (test.test_type or "").upper().strip()
        try:
            if test_type == "MANN_WHITNEY_U_TEST":
                if index + 1 < len(tests):
                    next_test = tests[index + 1]
                    next_type = (next_test.test_type or "").upper().strip()
                    if (
                        next_type == "MANN_WHITNEY_U_TEST"
                        and _can_pair_temporal_mann_whitney_tests(test, next_test)
                    ):
                        results.extend(
                            _execute_temporal_pair_mann_whitney(
                                test_a=test,
                                test_b=next_test,
                                user_sub=user_sub,
                                trace_id=trace_id,
                            )
                        )
                        index += 2
                        continue
                results.extend(
                    _execute_mann_whitney_test(
                        test=test, user_sub=user_sub, trace_id=trace_id
                    )
                )
            else:
                logger.warning(
                    "[plan_executor] Statistical test type '%s' is not implemented yet",
                    test_type,
                    extra={
                        "log_context": {
                            "trace_id": trace_id,
                            "event": "plan_executor.statistical_test.not_implemented",
                            "operation": "_execute_statistical_tests",
                            "outcome": "degraded",
                            "test_type": test_type or "-",
                        }
                    },
                )
        except Exception:
            logger.exception(
                "[plan_executor] Statistical test execution failed for test type '%s'",
                test_type,
                extra={
                    "log_context": {
                        "trace_id": trace_id,
                        "event": "plan_executor.statistical_test.failed",
                        "operation": "_execute_statistical_tests",
                        "outcome": "failure",
                        "test_type": test_type or "-",
                    }
                },
            )
        index += 1

    return results


def _emit_compiler_diagnostics(
    progress_cb: Optional[Callable[[str], None]], payload: Dict[str, Any], trace_id: str
) -> None:
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
        progress_cb(
            f"Compiler diagnostics: {json.dumps(payload, default=str, sort_keys=True)}"
        )


class VisualizationExecutionError(RuntimeError):
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


def _to_execution_error(
    failure_reasons: List[str], trace_id: Optional[str] = None
) -> VisualizationExecutionError:
    reason_set = set(failure_reasons)
    service_unavailable_count = sum(
        1 for reason in failure_reasons if reason == "service_unavailable"
    )
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
                user_message=(
                    "The analytics platform appears to be experiencing an outage right now (upstream service unavailable). Please try again in a moment."
                ),
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
            user_message="The analytics service returned an error while generating the visualization.",
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
        user_message="Could not fetch analytics data for this visualization. Please try again.",
        reason="data_fetch_failed",
        code="EXEC_DATA_FETCH_FAILED",
        trace_id=trace_id,
    )


def _get_metric_meta(metric_code: str) -> Dict[str, Any]:
    code = (metric_code or "").upper()
    meta: Dict[str, Any] = METRIC_METADATA.get(code) or {}
    return meta


def _derive_distribution_defaults(metric_code: str) -> tuple[int, int, int]:
    """Return (bins, min_value, max_value) using SSOT metadata with sensible fallbacks.

    Priority:
    - Use top-level promoted keys when present: range_min/range_max and distribution_default_buckets
    - Else use nested numeric block fields if present: numeric.default_buckets (and optional range_min/range_max if provided)
    - Else fallback to known safe ranges per metric; else general default (20, 0, 200)
    """
    meta = _get_metric_meta(metric_code)

    bins_any: Any = meta.get("distribution_default_buckets")
    numeric_block: Dict[str, Any] = cast(Dict[str, Any], meta.get("numeric") or {})
    bins = bins_any or numeric_block.get("default_buckets") or 20

    rmin: Any = meta.get("range_min")
    rmax: Any = meta.get("range_max")
    if rmin is None or rmax is None:
        n: Dict[str, Any] = cast(Dict[str, Any], meta.get("numeric") or {})
        rmin = rmin if rmin is not None else n.get("range_min")
        rmax = rmax if rmax is not None else n.get("range_max")

    if rmin is None or rmax is None:
        defaults: dict[str, tuple[int, int]] = {
            "AGE": (18, 95),
            "ADMISSION_NIHSS": (0, 42),
            "DTN": (0, 120),
        }
        if metric_code.upper() in defaults:
            rmin, rmax = defaults[metric_code.upper()]
        else:
            rmin = rmin if rmin is not None else 0
            rmax = rmax if rmax is not None else 200

    try:
        bins = int(bins)
    except Exception:
        logger.debug(
            "[plan_executor] Failed to parse distribution bucket count; using fallback",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "plan_executor.distribution_defaults.bins_fallback",
                    "operation": "_derive_distribution_defaults",
                    "outcome": "degraded",
                    "metric_code": metric_code,
                    "raw_bins": bins,
                    "fallback_bins": 20,
                }
            },
        )
        bins = 20
    try:
        rmin = int(rmin)
        rmax = int(rmax)
    except Exception:
        logger.debug(
            "[plan_executor] Failed to parse distribution range; using fallback range",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "plan_executor.distribution_defaults.range_fallback",
                    "operation": "_derive_distribution_defaults",
                    "outcome": "degraded",
                    "metric_code": metric_code,
                    "raw_range_min": rmin,
                    "raw_range_max": rmax,
                    "fallback_range_min": 0,
                    "fallback_range_max": 200,
                }
            },
        )
        rmin, rmax = 0, 200
    if rmin > rmax:
        rmin, rmax = rmax, rmin
    return bins, rmin, rmax


def _axis_from_meta(
    metric_code: str, x_min: int, x_max: int
) -> tuple[ChartAxis, ChartAxis]:
    metric_key = (metric_code or "").upper()
    meta = _get_metric_meta(metric_key)
    display = _AXIS_LABEL_OVERRIDES.get(metric_key) or _normalize_axis_display_label(
        get_metric_display_name(metric_key)
    )
    unit_any: Any = meta.get("unit")
    if unit_any is None:
        unit_any = cast(Dict[str, Any], meta.get("numeric") or {}).get("unit")
    unit: Optional[str] = cast(Optional[str], unit_any)
    if unit is None:
        unit = _AXIS_UNIT_FALLBACKS.get(metric_key)
    x_label = f"{display} ({unit})" if unit else display
    x_axis = ChartAxis(label=x_label, min_value=x_min, max_value=x_max)
    y_axis = ChartAxis(label="Cases")
    return x_axis, y_axis


def _format_iso_date(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except Exception:
        logger.debug(
            "[plan_executor] Failed to format ISO date; using raw token fallback",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "plan_executor.date.format_fallback",
                    "operation": "_format_iso_date",
                    "outcome": "degraded",
                    "raw_value": token,
                }
            },
        )
        return token.split("T", 1)[0]


def _parse_iso_date(value: str) -> Optional[datetime]:
    token = (value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        logger.debug(
            "[plan_executor] Failed to parse ISO date; returning None",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "plan_executor.date.parse_fallback",
                    "operation": "_parse_iso_date",
                    "outcome": "degraded",
                    "raw_value": token,
                }
            },
        )
        return None


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
                if parsed is not None and shown:
                    starts.append((parsed, shown))

            if isinstance(end_raw, str):
                parsed = _parse_iso_date(end_raw)
                shown = _format_iso_date(end_raw)
                if parsed is not None and shown:
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


def execute_plan(plan: AnalysisPlan, user_sub: str) -> VisualizationResponse:
    """Sync wrapper that delegates to the async implementation with concurrency=1.

    - If no event loop is running, run the coroutine directly with asyncio.run.
    - If an event loop is already running, offload to a new thread and run a fresh loop there.
    """
    trace_id = uuid4().hex
    coro = execute_plan_async(
        plan,
        user_sub,
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
    semaphore: asyncio.Semaphore
    progress_cb: Optional[ProgressCallback]
    log_graphql_query: bool
    query_cb: Optional[QueryDebugCallback]


@dataclass(frozen=True)
class RequestExecutionResult:
    spec: RequestSpec
    series: List[ChartSeries]


def _emit_progress(
    context: ExecutionContext, completed: int, total: int, prefix: str = "Fetching data"
) -> None:
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
        trace_id=trace_id,
        semaphore=context.semaphore,
        log_graphql_query=context.log_graphql_query,
        request_warnings=request_warnings,
        batched_time_periods=spec.batched_time_periods,
        query_cb=context.query_cb,
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
        _emit_progress(
            context, completed=completed, total=total_requests, prefix=progress_prefix
        )

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
        _emit_progress(
            context, completed=completed, total=total_requests, prefix=progress_prefix
        )

    return results


async def execute_plan_async(
    plan: AnalysisPlan,
    user_sub: str,
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

    try:
        plan = resolve_plan_metric_origins(
            plan=plan, user_sub=user_sub, trace_id=trace_id_resolved
        )
    except OriginScopeResolutionError as exc:
        raise VisualizationExecutionError(
            user_message=str(exc),
            reason="origin_scope_resolution",
            code="EXEC_ORIGIN_SCOPE",
            trace_id=trace_id_resolved,
            clarification_type=exc.clarification_type,
            clarification_options=exc.clarification_options,
        ) from exc

    normalization_summary = None

    plan_charts = coalesce(plan.charts, [])
    response: VisualizationResponse = VisualizationResponse(trace_id=trace_id_resolved)
    estimated_queries = estimate_query_count_for_plan(plan)
    actual_queries = 0
    summary_batches: List[ExecutionBatchSummary] = []

    resolved_concurrency = (
        _EXECUTOR_DEFAULT_MAX_CONCURRENCY
        if max_concurrency is None
        else max(1, int(max_concurrency))
    )
    sem = asyncio.Semaphore(resolved_concurrency)
    execution_context = ExecutionContext(
        user_sub=user_sub,
        semaphore=sem,
        progress_cb=progress_cb,
        log_graphql_query=_LOG_GRAPHQL_QUERY,
        query_cb=query_cb,
    )

    for planChart in plan_charts:
        metric_requests, derived_axes, metric_data_origins, metric_scope_labels = (
            build_metric_requests(
                plan_chart=planChart,
                derive_defaults_fn=_derive_distribution_defaults,
                axis_from_meta_fn=_axis_from_meta,
            )
        )

        compiled_grouping = compile_chart_grouping(planChart)
        dims: List[Dimension] = compiled_grouping.dimensions

        for batch in compiled_grouping.batches:
            request_failures: List[str] = []
            request_warnings: List[str] = []
            filter_dims = batch.filter_dims
            batched_time_enabled = batch.batched_time_enabled
            batched_time_periods = batch.batched_time_periods
            combos_list = batch.combos_list

            gb_field = batch.server_groupby
            fallback_specs: List[RequestSpec] = []
            include_metric_alias = len(planChart.metrics) > 1

            chart_filter = to_gql_filter(coalesce(planChart.filters, None))

            primary_specs, combo_contexts = build_primary_request_specs(
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
            )
            total_requests = max(1, len(primary_specs))
            actual_queries += total_requests

            summary_batches.append(
                make_batch_summary(
                    chart_title=f"{(planChart.chart_type or 'CHART').upper()} chart",
                    chart_type=planChart.chart_type,
                    server_groupby=gb_field,
                    filter_dimensions=[d.kind.__name__ for d in filter_dims],
                    batched_time_period_count=len(batched_time_periods)
                    if batched_time_enabled
                    else 0,
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

            if (
                _ENABLE_UNBATCHED_TIME_FALLBACK
                and not _STRICT_MODE
                and should_retry_unbatched_time(
                    all_series=all_series,
                    request_failures=request_failures,
                    batched_time_enabled=batched_time_enabled,
                    batched_time_periods=batched_time_periods,
                )
            ):
                logger.warning(
                    "[plan_executor] Batched multi-period request timed out; retrying with per-period requests (period_count=%s, combos=%s)",
                    len(batched_time_periods),
                    len(combo_contexts),
                    extra={
                        "log_context": {
                            "trace_id": trace_id_resolved,
                            "event": "plan_executor.unbatched_time_fallback",
                            "operation": "execute_plan_async",
                            "outcome": "degraded",
                            "batched_time_period_count": len(batched_time_periods),
                            "combo_count": len(combo_contexts),
                            "request_failure_count": len(request_failures),
                        }
                    },
                )
                request_failures.clear()

                fallback_specs = build_fallback_request_specs(
                    combo_contexts=combo_contexts,
                    batched_time_periods=batched_time_periods,
                )

                retry_count = max(1, len(fallback_specs))
                actual_queries += retry_count
                request_results = await _execute_specs_sequential(
                    specs=fallback_specs,
                    request_failures=request_failures,
                    request_warnings=request_warnings,
                    context=execution_context,
                    trace_id=trace_id_resolved,
                    total_requests=retry_count,
                    progress_prefix="Retrying with per-period requests",
                )
                all_series = [
                    item for result in request_results for item in result.series
                ]

            sampled_period_override = _sampled_period_from_specs(primary_specs)
            if sampled_period_override is None and fallback_specs:
                sampled_period_override = _sampled_period_from_specs(fallback_specs)

            # Surface partial-result scenarios (some scopes returned no rows) without failing whole chart.
            empty_scope_labels = [
                _request_scope_label(result.spec)
                for result in request_results
                if not result.series
            ]
            for scope_label in sorted(set(empty_scope_labels)):
                warning_msg = f"No data was returned for {scope_label}. The chart includes the data that is available."
                if warning_msg not in response.warnings:
                    logger.debug(
                        "[plan_executor] Appending partial-result warning for empty scope",
                        extra={
                            "log_context": {
                                "trace_id": trace_id_resolved,
                                "event": "plan_executor.warning.empty_scope_appended",
                                "operation": "execute_plan_async",
                                "outcome": "degraded",
                                "scope_label": scope_label,
                                "chart_type": planChart.chart_type or "Chart",
                            }
                        },
                    )
                    response.warnings.append(warning_msg)

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
                    raise _to_execution_error(
                        request_failures, trace_id=trace_id_resolved
                    )
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
        response.stats.extend(
            _dedupe_statistical_results(
                _execute_statistical_tests(
                    plan=plan, user_sub=user_sub, trace_id=trace_id_resolved
                )
            )
        )

    if summary_cb is not None:
        payload = make_execution_summary(
            trace_id=trace_id_resolved,
            chart_count=len(plan_charts),
            estimated_queries=estimated_queries,
            actual_queries=actual_queries,
            batches=summary_batches,
            normalization=normalization_summary,
        )
        try:
            summary_cb(payload)
        except Exception:
            logger.warning(
                "Failed to emit execution summary callback",
                exc_info=True,
                extra={
                    "log_context": {
                        "trace_id": trace_id_resolved,
                        "event": "plan_executor.summary_callback.failed",
                        "operation": "execute_plan_async",
                        "outcome": "degraded",
                        "chart_count": len(plan_charts),
                        "warning_count": len(response.warnings),
                    }
                },
            )

    return response
