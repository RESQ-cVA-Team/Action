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
    semaphore: asyncio.Semaphore,
    log_graphql_query: bool = False,
) -> List[ChartSeries]:
    async with semaphore:
        query_str = req.to_graphql_string()
        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        if log_graphql_query:
            logger.debug(
                "[plan_executor] GraphQL query for chart (groupBy=%s, labels=%s, hash=%s):\n%s",
                group_by_field,
                " | ".join(label_parts),
                q_hash,
                query_str,
            )
        else:
            logger.info(
                "[plan_executor] GraphQL query for chart (groupBy=%s, labels=%s, hash=%s, len=%s)",
                group_by_field,
                " | ".join(label_parts),
                q_hash,
                len(query_str),
            )
        try:
            resp = await asyncio.to_thread(client.query, query_str, user_sub, None, True)
        except GraphQLProxyError as exc:
            if exc.kind == "timeout":
                request_failures.append("timeout")
            elif exc.kind == "http_error" and exc.status_code in {429, 500, 502, 503, 504}:
                request_failures.append("service_unavailable")
            else:
                request_failures.append("upstream_error")
            logger.error(
                "[plan_executor] GraphQL request failed (groupBy=%s, labels=%s, kind=%s, status=%s)",
                group_by_field,
                " - ".join([part for part in label_parts if part]) or "(none)",
                exc.kind,
                exc.status_code,
            )
            return []

    if resp is None:
        request_failures.append("upstream_error")
        logger.error("[plan_executor] GraphQL returned empty response")
        return []

    metrics_payload = None
    if (x := resp) and (x := x.data) and (x := x.get_metrics) and (x := x.metrics):
        metrics_payload = x

    if getattr(resp, "errors", None):
        request_failures.append("graphql_error")
        error_count = len(resp.errors or [])
        logger.error("[plan_executor] GraphQL errors returned (count=%s)", error_count)

    if not metrics_payload:
        return []

    return map_metrics_payload_to_series(
        metrics_payload=metrics_payload,
        label_parts=label_parts,
        include_metric_alias=include_metric_alias,
        group_by_field=group_by_field,
        add_time_period_labels=add_time_period_labels,
    )
