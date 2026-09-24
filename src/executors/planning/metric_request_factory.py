from __future__ import annotations

import math
from typing import List, Optional, cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DataOrigin, MetricRequest
from src.domain.graphql.ssot_enums import MetricType
from src.domain.langchain import schema as S
from src.executors.planning.ssot_metric_defaults import (
    get_distribution_defaults,
    get_histogram_axes,
    is_enum_metric,
    is_minutes_metric,
    is_score_metric,
    resolve_implicit_distribution_layout,
    resolve_minutes_distribution_layout,
    resolve_score_distribution_layout,
)


def _metric_scope_label(metric: S.MetricSpec) -> Optional[str]:
    scope = cast(Optional[S.OriginScopeSpec], getattr(metric, "origin_scope", None))
    if scope is None:
        return None

    if isinstance(scope.label, str) and scope.label.strip():
        return scope.label.strip()

    scope_type = (scope.scope_type or "").strip().lower()
    if scope_type == "mine":
        return "My Hospital"
    if scope_type == "country_average":
        if isinstance(scope.value, str) and scope.value.strip():
            return f"National Mean ({scope.value.strip()})"
        if isinstance(scope.country_code, str) and scope.country_code.strip():
            return f"National Mean ({scope.country_code.strip().upper()})"
        return "National Mean"
    if scope_type == "country_code":
        if isinstance(scope.country_code, str) and scope.country_code.strip():
            return f"Country ({scope.country_code.strip().upper()})"
        if isinstance(scope.value, str) and scope.value.strip():
            return f"Country ({scope.value.strip().upper()})"
        return "Country"
    if scope_type == "provider_name" and isinstance(scope.value, str) and scope.value.strip():
        return scope.value.strip()
    if scope_type == "provider_group_name" and isinstance(scope.value, str) and scope.value.strip():
        return scope.value.strip()

    return None


def _resolve_numeric_request_options(
    plan_chart: S.ChartSpec,
    metric_code: str,
) -> tuple[int, int, int]:
    default_bins, default_min, default_max = get_distribution_defaults(metric_code)

    resolved_bins = default_bins
    resolved_lower = default_min
    resolved_upper = default_max

    numeric_resolution = cast(Optional[S.NumericResolutionSpec], getattr(plan_chart, "numeric_resolution", None))
    if numeric_resolution is None:
        if is_score_metric(metric_code):
            implicit_layout = resolve_score_distribution_layout(
                lower_bound=resolved_lower,
                upper_bound=resolved_upper,
            )
        elif is_minutes_metric(metric_code):
            implicit_layout = resolve_minutes_distribution_layout(
                lower_bound=resolved_lower,
                upper_bound=resolved_upper,
            )
        else:
            implicit_layout = resolve_implicit_distribution_layout(
                lower_bound=resolved_lower,
                upper_bound=resolved_upper,
            )
        return implicit_layout.bin_count, int(implicit_layout.lower_bound), int(implicit_layout.upper_bound)

    value_domain = numeric_resolution.value_domain
    if value_domain is not None:
        if value_domain.lower_bound is not None:
            resolved_lower = int(value_domain.lower_bound)
        if value_domain.upper_bound is not None:
            resolved_upper = int(value_domain.upper_bound)

    bucketing = numeric_resolution.bucketing
    explicit_bucket_override = False
    if bucketing is not None:
        if bucketing.bucket_count is not None:
            resolved_bins = int(bucketing.bucket_count)
            explicit_bucket_override = True
        elif bucketing.bucket_size is not None:
            span = max(1, int(resolved_upper) - int(resolved_lower))
            resolved_bins = max(1, int(math.ceil(span / int(bucketing.bucket_size))))
            explicit_bucket_override = True

    if not explicit_bucket_override:
        if is_score_metric(metric_code):
            implicit_layout = resolve_score_distribution_layout(
                lower_bound=int(resolved_lower),
                upper_bound=int(resolved_upper),
            )
        elif is_minutes_metric(metric_code):
            implicit_layout = resolve_minutes_distribution_layout(
                lower_bound=int(resolved_lower),
                upper_bound=int(resolved_upper),
            )
        else:
            implicit_layout = resolve_implicit_distribution_layout(
                lower_bound=int(resolved_lower),
                upper_bound=int(resolved_upper),
            )
        return implicit_layout.bin_count, int(implicit_layout.lower_bound), int(implicit_layout.upper_bound)

    return resolved_bins, int(resolved_lower), int(resolved_upper)


def build_metric_requests(
    plan_chart: S.ChartSpec,
) -> tuple[List[MetricRequest], Optional[tuple[ChartAxis, ChartAxis]], List[Optional[DataOrigin]], List[Optional[str]]]:
    metric_requests: List[MetricRequest] = []
    metric_data_origins: List[Optional[DataOrigin]] = []
    metric_scope_labels: List[Optional[str]] = []
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]] = None
    chart_type_upper = (plan_chart.chart_type or "").upper()

    for metric in plan_chart.metrics:
        metric_data_origin: Optional[DataOrigin] = None
        if metric.data_origin is not None:
            metric_data_origin = DataOrigin.model_validate(metric.data_origin.model_dump(by_alias=True, exclude_none=True))
        metric_scope_label = _metric_scope_label(metric)

        metric_request = MetricRequest(metricType=MetricType(metric.metric)).with_stats()

        if is_enum_metric(metric.metric):
            metric_request = metric_request.with_categorical()
        else:
            resolved_bins, resolved_min, resolved_max = _resolve_numeric_request_options(
                plan_chart=plan_chart,
                metric_code=metric.metric,
            )

            metric_request = metric_request.with_distribution(
                bin_count=resolved_bins,
                lower=resolved_min,
                upper=resolved_max,
            )

            if len(plan_chart.metrics) == 1 and chart_type_upper == "HISTOGRAM":
                derived_axes = get_histogram_axes(metric.metric, resolved_min, resolved_max)

        metric_requests.append(metric_request)
        metric_data_origins.append(metric_data_origin)
        metric_scope_labels.append(metric_scope_label)

    return metric_requests, derived_axes, metric_data_origins, metric_scope_labels
