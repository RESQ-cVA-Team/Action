from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import List, Optional

from src.domain.dto.charts.types import ChartSeries
from src.domain.graphql.request import GraphQLQueryRequest
from src.executors.graphql.client import GraphQLProxyClient, GraphQLProxyError
from src.executors.mapping.series_mapper import map_metrics_payload_to_series

logger = logging.getLogger(__name__)


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
) -> List[ChartSeries]:
    trace_label = trace_id
    request_label = scope_label or " | ".join([part for part in label_parts if part]) or "(none)"
    async with semaphore:
        query_str = req.to_graphql_string()
        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        if log_graphql_query:
            logger.debug(
                "[plan_executor] GraphQL query for chart (trace_id=%s, groupBy=%s, labels=%s, hash=%s):\n%s",
                trace_label,
                group_by_field,
                request_label,
                q_hash,
                query_str,
            )
        else:
            logger.info(
                "[plan_executor] GraphQL query for chart (trace_id=%s, groupBy=%s, labels=%s, hash=%s, len=%s)",
                trace_label,
                group_by_field,
                request_label,
                q_hash,
                len(query_str),
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
            elif exc.kind == "http_error" and exc.status_code in {429, 500, 502, 503, 504}:
                request_failures.append("service_unavailable")
            else:
                request_failures.append("upstream_error")
            logger.error(
                "[plan_executor] GraphQL request failed (trace_id=%s, groupBy=%s, labels=%s, kind=%s, status=%s)",
                trace_label,
                group_by_field,
                request_label,
                exc.kind,
                exc.status_code,
            )
            return []

    if resp is None:
        request_failures.append("upstream_error")
        logger.error("[plan_executor] GraphQL returned empty response (trace_id=%s)", trace_label)
        return []

    metrics_payload = None
    if (x := resp) and (x := x.data) and (x := x.get_metrics) and (x := x.metrics):
        metrics_payload = x

    if getattr(resp, "errors", None):
        request_failures.append("graphql_error")
        error_count = len(resp.errors or [])
        logger.error("[plan_executor] GraphQL errors returned (trace_id=%s, count=%s)", trace_label, error_count)

    if not metrics_payload:
        logger.warning(
            "[plan_executor] GraphQL response had no metrics payload (trace_id=%s, groupBy=%s, labels=%s, hash=%s, has_errors=%s)",
            trace_label,
            group_by_field,
            request_label,
            q_hash,
            bool(getattr(resp, "errors", None)),
        )
        return []

    metric_count = 0
    kpi_group_count = 0
    try:
        metric_count = len(metrics_payload)
        for metric in metrics_payload.values():
            kpi_groups = getattr(metric, "kpi_group", None)
            if kpi_groups is None:
                continue
            try:
                kpi_group_count += len(kpi_groups)
            except Exception:
                kpi_group_count += 1
    except Exception:
        metric_count = 0

    series = map_metrics_payload_to_series(
        metrics_payload=metrics_payload,
        label_parts=label_parts,
        include_metric_alias=include_metric_alias,
        group_by_field=group_by_field,
        add_time_period_labels=add_time_period_labels,
        scope_label=scope_label,
    )

    skipped_rows = 0
    try:
        for metric in metrics_payload.values():
            kpi_groups = getattr(metric, "kpi_group", None)
            if not isinstance(kpi_groups, list):
                continue
            for kpi in kpi_groups:
                if getattr(kpi, "kpi1", None) is None:
                    skipped_rows += 1
    except Exception:
        skipped_rows = 0

    total_rows = kpi_group_count
    valid_rows = max(0, total_rows - skipped_rows)

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
                    detail_bits.append(f"{skipped_rows}/{total_rows} row(s) were omitted")
                if getattr(resp, "errors", None):
                    detail_bits.append("the backend returned validation errors")
                _append_warning(f"Partial data returned for {request_label}; {' and '.join(detail_bits)}.")
            else:
                _append_warning(f"No usable data was returned for {request_label}; 0/{total_rows} row(s) had valid data.")

        logger.warning(
            "[plan_executor] Metrics payload mapped to zero series (trace_id=%s, groupBy=%s, labels=%s, hash=%s, metrics=%s, kpi_groups=%s)",
            trace_label,
            group_by_field,
            request_label,
            q_hash,
            metric_count,
            kpi_group_count,
        )
        return series

    has_graphql_errors = bool(getattr(resp, "errors", None))
    if skipped_rows > 0 or has_graphql_errors:
        warning_bits: List[str] = []
        if skipped_rows > 0:
            warning_bits.append(f"{skipped_rows}/{total_rows} row(s) were omitted")
        if has_graphql_errors:
            warning_bits.append("the backend returned validation errors")

        warning_text = f"Partial data returned for {request_label}; {' and '.join(warning_bits)}."
        _append_warning(warning_text)

    return series
