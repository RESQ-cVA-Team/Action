from __future__ import annotations

from typing import Callable, List, Optional, cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DataOrigin, MetricRequest
from src.domain.graphql.ssot_enums import MetricType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import DistributionSpec


def derive_distribution_defaults(metric: S.MetricSpec) -> DistributionSpec:
    if metric.distribution is not None:
        return metric.distribution

    # Conservative defaults; caller may inject smarter SSOT-derived defaults via wrappers if needed.
    return DistributionSpec(num_buckets=20, min_value=0, max_value=200)


def _normalize_analysis_mode(analysis_mode: str | None) -> str:
    mode = (analysis_mode or "").strip().upper()
    if mode in {"TIME_SERIES", "DISTRIBUTION", "SUMMARY", "COMPARISON"}:
        return mode
    return ""


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


def build_metric_requests(
    plan_chart: S.ChartSpec,
    derive_defaults_fn: Callable[[str], tuple[int, int, int]],
    axis_from_meta_fn: Callable[[str, int, int], tuple[ChartAxis, ChartAxis]],
) -> tuple[List[MetricRequest], Optional[tuple[ChartAxis, ChartAxis]], List[Optional[DataOrigin]], List[Optional[str]]]:
    metric_requests: List[MetricRequest] = []
    metric_data_origins: List[Optional[DataOrigin]] = []
    metric_scope_labels: List[Optional[str]] = []
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]] = None
    has_grouping = bool(plan_chart.group_by)
    chart_analysis_mode = _normalize_analysis_mode(getattr(plan_chart, "analysis_mode", None))
    if not chart_analysis_mode:
        raise ValueError("Chart analysis_mode must be resolved before execution.")

    for metric in plan_chart.metrics:
        metric_data_origin: Optional[DataOrigin] = None
        if metric.data_origin is not None:
            metric_data_origin = DataOrigin.model_validate(metric.data_origin.model_dump(by_alias=True, exclude_none=True))
        metric_scope_label = _metric_scope_label(metric)

        request_distribution = chart_analysis_mode == "DISTRIBUTION" or metric.distribution is not None
        if request_distribution:
            distribution = metric.distribution
            if distribution is None:
                bins, rmin, rmax = derive_defaults_fn(metric.metric)
                distribution = DistributionSpec(num_buckets=bins, min_value=rmin, max_value=rmax)
            if len(plan_chart.metrics) == 1:
                derived_axes = axis_from_meta_fn(metric.metric, distribution.min_value, distribution.max_value)
            metric_requests.append(
                MetricRequest(metricType=MetricType(metric.metric)).with_distribution(
                    bin_count=distribution.num_buckets,
                    lower=distribution.min_value,
                    upper=distribution.max_value,
                )
            )
        else:
            metric_requests.append(MetricRequest(metricType=MetricType(metric.metric)).with_stats())
        metric_data_origins.append(metric_data_origin)
        metric_scope_labels.append(metric_scope_label)

    return metric_requests, derived_axes, metric_data_origins, metric_scope_labels
