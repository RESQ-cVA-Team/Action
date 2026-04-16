from __future__ import annotations

from typing import Any, List, Optional

from src.domain.dto.execution_summary import ExecutionBatchSummary, ExecutionSummary
from src.shared.ssot_loader import get_canonical_display_name


def make_batch_summary(
    chart_title: str,
    chart_type: str,
    server_groupby: Optional[str],
    filter_dimensions: List[str],
    batched_time_period_count: int,
    query_count: int,
) -> ExecutionBatchSummary:
    return ExecutionBatchSummary(
        chart_title=chart_title,
        chart_type=chart_type,
        server_groupby=get_canonical_display_name(server_groupby) if server_groupby else None,
        filter_dimensions=filter_dimensions,
        batched_time_period_count=batched_time_period_count,
        query_count=query_count,
    )


def make_execution_summary(
    trace_id: Optional[str],
    chart_count: int,
    requested_visual_layout: Optional[str],
    estimated_queries: int,
    actual_queries: int,
    batches: List[ExecutionBatchSummary],
    normalization: Optional[Any] = None,
) -> ExecutionSummary:
    return ExecutionSummary(
        trace_id=trace_id,
        chart_count=chart_count,
        requested_visual_layout=requested_visual_layout,
        estimated_queries=estimated_queries,
        actual_queries=actual_queries,
        batches=batches,
        normalization=normalization,
    )
