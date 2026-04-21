from typing import List, Optional

from pydantic import BaseModel, Field


class ExecutionBatchSummary(BaseModel):
    chart_title: str
    chart_type: str
    server_groupby: Optional[str] = None
    filter_dimensions: List[str] = Field(default_factory=list)
    batched_time_period_count: int = 0
    query_count: int = 0


class PlanNormalizationSummary(BaseModel):
    charts_in: int = 0
    charts_out: int = 0
    dropped_empty_charts: int = 0
    metrics_in: int = 0
    metrics_out: int = 0
    dropped_empty_metrics: int = 0
    normalized_metric_codes: int = 0
    normalized_chart_types: int = 0
    deduped_groupby_entries: int = 0
    normalized_canonical_groupby_fields: int = 0
    dropped_invalid_groupby_fields: int = 0
    fallback_chart_type_count: int = 0
    normalized_text_fields: int = 0


class ExecutionSummary(BaseModel):
    trace_id: Optional[str] = None
    chart_count: int
    estimated_queries: int
    actual_queries: int
    batches: List[ExecutionBatchSummary] = Field(default_factory=list)
    normalization: Optional[PlanNormalizationSummary] = None
