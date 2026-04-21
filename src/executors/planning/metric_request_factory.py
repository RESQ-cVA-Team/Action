from __future__ import annotations

from typing import Callable, List, Optional

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


def build_metric_requests(
    plan_chart: S.ChartSpec,
    derive_defaults_fn: Callable[[str], tuple[int, int, int]],
    axis_from_meta_fn: Callable[[str, int, int], tuple[ChartAxis, ChartAxis]],
) -> tuple[List[MetricRequest], Optional[tuple[ChartAxis, ChartAxis]], List[Optional[DataOrigin]]]:
    metric_requests: List[MetricRequest] = []
    metric_data_origins: List[Optional[DataOrigin]] = []
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]] = None
    has_grouping = bool(plan_chart.group_by)

    for metric in plan_chart.metrics:
        metric_data_origin: Optional[DataOrigin] = None
        if metric.data_origin is not None:
            metric_data_origin = DataOrigin.model_validate(metric.data_origin.model_dump(by_alias=True, exclude_none=True))

        if has_grouping:
            metric_requests.append(MetricRequest(metricType=MetricType(metric.metric)).with_stats())
            metric_data_origins.append(metric_data_origin)
            continue

        distribution = metric.distribution
        if distribution is None:
            bins, rmin, rmax = derive_defaults_fn(metric.metric)
            distribution = DistributionSpec(num_buckets=bins, min_value=rmin, max_value=rmax)
            if len(plan_chart.metrics) == 1:
                derived_axes = axis_from_meta_fn(metric.metric, rmin, rmax)

        metric_requests.append(
            MetricRequest(metricType=MetricType(metric.metric)).with_distribution(
                distribution.num_buckets,
                distribution.min_value,
                distribution.max_value,
            )
        )
        metric_data_origins.append(metric_data_origin)

    return metric_requests, derived_axes, metric_data_origins
