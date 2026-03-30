from __future__ import annotations

from typing import Callable, List, Optional

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import MetricRequest
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
) -> tuple[List[MetricRequest], Optional[tuple[ChartAxis, ChartAxis]]]:
    metric_requests: List[MetricRequest] = []
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]] = None
    has_grouping = bool(plan_chart.group_by)

    for metric in plan_chart.metrics:
        if has_grouping:
            metric_requests.append(MetricRequest(metricType=MetricType(metric.metric)).with_stats())
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

    return metric_requests, derived_axes
