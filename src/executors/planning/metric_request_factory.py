from __future__ import annotations

from typing import Callable, List, Optional, cast

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DataOrigin, MetricRequest
from src.domain.graphql.ssot_enums import MetricType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import DistributionSpec


def _scope_label(scope: Optional[S.OriginScopeSpec]) -> Optional[str]:
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
    if isinstance(plan_chart, S.LineChartSpec):
        for series in plan_chart.series:
            metric_requests.append(MetricRequest(metricType=MetricType(series.metric)).with_stats())

            metric_data_origin: Optional[DataOrigin] = None
            if series.data_origin is not None:
                metric_data_origin = DataOrigin.model_validate(
                    series.data_origin.model_dump(by_alias=True, exclude_none=True)
                )
            metric_data_origins.append(metric_data_origin)
            metric_scope_labels.append(_scope_label(series.origin_scope))

    elif isinstance(plan_chart, S.HistogramChartSpec):
        metric = plan_chart.x_axis.metric
        bins = int(plan_chart.x_axis.bins)
        min_value = plan_chart.x_axis.min_value
        max_value = plan_chart.x_axis.max_value
        if min_value is None or max_value is None:
            _, rmin, rmax = derive_defaults_fn(metric)
            if min_value is None:
                min_value = rmin
            if max_value is None:
                max_value = rmax

        distribution = DistributionSpec(num_buckets=bins, min_value=int(min_value), max_value=int(max_value))
        derived_axes = axis_from_meta_fn(metric, distribution.min_value, distribution.max_value)
        metric_requests.append(
            MetricRequest(metricType=MetricType(metric)).with_distribution(
                bin_count=distribution.num_buckets,
                lower=distribution.min_value,
                upper=distribution.max_value,
            )
        )

        metric_data_origin: Optional[DataOrigin] = None
        if plan_chart.data_origin is not None:
            metric_data_origin = DataOrigin.model_validate(
                plan_chart.data_origin.model_dump(by_alias=True, exclude_none=True)
            )
        metric_data_origins.append(metric_data_origin)
        metric_scope_labels.append(_scope_label(plan_chart.origin_scope))

    return metric_requests, derived_axes, metric_data_origins, metric_scope_labels
