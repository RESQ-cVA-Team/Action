from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, cast

from src.domain.dto.charts.types import ChartSeries
from src.domain.graphql.request import GraphQLQueryRequest
from src.executors.graphql.client import GraphQLProxyClient, GraphQLProxyError
from src.executors.mapping.series_mapper import map_metrics_payload_to_series
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)


def _runner_log_context(
    *,
    event: str,
    outcome: str,
    request_label: str,
    query_hash: str,
    group_by_field: Optional[str],
    **fields: Any,
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Any] = {
        "event": event,
        "operation": "run_graphql_request",
        "outcome": outcome,
        "request_label": request_label,
        "graphql_hash": query_hash,
        "graphql_group_by": group_by_field or "-",
    }
    for key, value in fields.items():
        if value is None:
            continue
        context[key] = value
    return {"log_context": context}


async def run_graphql_request(
    req: GraphQLQueryRequest,
    label_parts: List[str],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    add_time_period_labels: bool,
    request_failures: List[str],
    client: GraphQLProxyClient,
    user_sub: str,
    trace_id: str,
    semaphore: asyncio.Semaphore,
    scope_label: Optional[str] = None,
    request_warnings: Optional[List[str]] = None,
    log_graphql_query: bool = False,
    batched_time_periods: Optional[List[Any]] = None,
) -> List[ChartSeries]:
    trace_label = trace_id
    request_label = (
        scope_label or " | ".join([part for part in label_parts if part]) or "(none)"
    )
    async with semaphore:
        query_str = req.to_graphql_string()
        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        with log_context(
            trace_id=trace_label,
            graphql_group_by=group_by_field or "-",
            graphql_hash=q_hash,
            request_label=request_label,
        ):
            if log_graphql_query:
                logger.info("[plan_executor] GraphQL query for chart:\n%s", query_str)
            else:
                logger.debug(
                    "[plan_executor] GraphQL query prepared",
                    extra={"log_context": {"query_length": len(query_str)}},
                )
            try:
                resp = await asyncio.to_thread(
                    client.query,
                    query_str=query_str,
                    user_sub=user_sub,
                    trace_id=trace_id,
                    variables=None,
                    raise_on_error=True,
                )
            except GraphQLProxyError as exc:
                if exc.kind == "timeout":
                    request_failures.append("timeout")
                elif exc.kind == "http_error" and exc.status_code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    request_failures.append("service_unavailable")
                else:
                    request_failures.append("upstream_error")
                logger.error(
                    "[plan_executor] GraphQL request failed (kind=%s, status=%s)",
                    exc.kind,
                    exc.status_code,
                    extra=_runner_log_context(
                        event="request_runner.graphql_request_failed",
                        outcome="failure",
                        request_label=request_label,
                        query_hash=q_hash,
                        group_by_field=group_by_field,
                        error_kind=exc.kind,
                        status_code=exc.status_code,
                        failure_reason=request_failures[-1],
                    ),
                )
                return []

    if resp is None:
        request_failures.append("upstream_error")
        logger.error(
            "[plan_executor] GraphQL returned empty response",
            extra=_runner_log_context(
                event="request_runner.graphql_response_empty",
                outcome="failure",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
                failure_reason="upstream_error",
            ),
        )
        return []

    metrics_payload = None
    if (x := resp) and (x := x.data) and (x := x.get_metrics) and (x := x.metrics):
        metrics_payload = x

    if getattr(resp, "errors", None):
        request_failures.append("graphql_error")
        error_count = len(resp.errors or [])
        logger.error(
            "[plan_executor] GraphQL errors returned (count=%s)",
            error_count,
            extra=_runner_log_context(
                event="request_runner.graphql_errors_returned",
                outcome="failure",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
                error_count=error_count,
            ),
        )

    if not metrics_payload:
        logger.warning(
            "[plan_executor] GraphQL response had no metrics payload (groupBy=%s, labels=%s, hash=%s, has_errors=%s)",
            group_by_field,
            request_label,
            q_hash,
            bool(getattr(resp, "errors", None)),
            extra=_runner_log_context(
                event="request_runner.metrics_payload_missing",
                outcome="degraded",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
                has_errors=bool(getattr(resp, "errors", None)),
            ),
        )
        return []

    metric_count = 0
    kpi_group_count = 0
    try:
        metric_count = len(metrics_payload)
        for metric_alias, metric in metrics_payload.items():
            kpi_groups = getattr(metric, "kpi_group", None)
            if kpi_groups is None:
                continue
            try:
                kpi_group_count += len(kpi_groups)
            except Exception:
                logger.debug(
                    "[plan_executor] Failed to count KPI groups; using fallback count",
                    extra=_runner_log_context(
                        event="request_runner.kpi_group_count_fallback",
                        outcome="degraded",
                        request_label=request_label,
                        query_hash=q_hash,
                        group_by_field=group_by_field,
                        metric_alias=str(metric_alias),
                        fallback_count=1,
                    ),
                    exc_info=True,
                )
                kpi_group_count += 1
    except Exception:
        logger.debug(
            "[plan_executor] Failed to inspect metrics payload counts; using zero-count fallback",
            extra=_runner_log_context(
                event="request_runner.metrics_payload_count_fallback",
                outcome="degraded",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
            ),
            exc_info=True,
        )
        metric_count = 0

    series = map_metrics_payload_to_series(
        metrics_payload=metrics_payload,
        label_parts=label_parts,
        include_metric_alias=include_metric_alias,
        group_by_field=group_by_field,
        add_time_period_labels=add_time_period_labels,
        scope_label=scope_label,
        batched_time_periods=batched_time_periods,
    )

    skipped_rows = 0
    metric_alias_for_rows: Optional[str] = None
    try:
        for metric_alias, metric in metrics_payload.items():
            metric_alias_for_rows = str(metric_alias)
            kpi_groups = getattr(metric, "kpi_group", None)
            if not isinstance(kpi_groups, list):
                continue
            for kpi in cast(List[object], kpi_groups):
                if getattr(kpi, "kpi1", None) is None:
                    skipped_rows += 1
    except Exception:
        logger.debug(
            "[plan_executor] Failed to inspect skipped KPI rows; using zero skipped-row fallback",
            extra=_runner_log_context(
                event="request_runner.skipped_rows_count_fallback",
                outcome="degraded",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
                metric_alias=metric_alias_for_rows,
            ),
            exc_info=True,
        )
        skipped_rows = 0

    total_rows = kpi_group_count

    def _append_warning(message: str) -> None:
        if request_warnings is None:
            return
        if message not in request_warnings:
            request_warnings.append(message)

    if not series:
        if total_rows > 0:
            if skipped_rows > 0 or getattr(resp, "errors", None):
                detail_bits: List[str] = []
                if skipped_rows > 0:
                    detail_bits.append(
                        f"{skipped_rows}/{total_rows} row(s) were omitted"
                    )
                if getattr(resp, "errors", None):
                    detail_bits.append("the backend returned validation errors")
                _append_warning(
                    f"Partial data returned for {request_label}; {' and '.join(detail_bits)}."
                )
            else:
                _append_warning(
                    f"No usable data was returned for {request_label}; 0/{total_rows} row(s) had valid data."
                )

        logger.warning(
            "[plan_executor] Metrics payload mapped to zero series (groupBy=%s, labels=%s, hash=%s, metrics=%s, kpi_groups=%s)",
            group_by_field,
            request_label,
            q_hash,
            metric_count,
            kpi_group_count,
            extra=_runner_log_context(
                event="request_runner.zero_series_generated",
                outcome="degraded",
                request_label=request_label,
                query_hash=q_hash,
                group_by_field=group_by_field,
                metric_count=metric_count,
                kpi_group_count=kpi_group_count,
                skipped_rows=skipped_rows,
                has_graphql_errors=bool(getattr(resp, "errors", None)),
            ),
        )
        return series

    has_graphql_errors = bool(getattr(resp, "errors", None))
    if skipped_rows > 0 or has_graphql_errors:
        warning_bits: List[str] = []
        if skipped_rows > 0:
            warning_bits.append(f"{skipped_rows}/{total_rows} row(s) were omitted")
        if has_graphql_errors:
            warning_bits.append("the backend returned validation errors")

        warning_text = (
            f"Partial data returned for {request_label}; {' and '.join(warning_bits)}."
        )
        _append_warning(warning_text)

    return series
