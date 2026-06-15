from __future__ import annotations

from typing import Callable, Dict, List, Optional

from src.domain.dto.charts.types import ChartAxis
from src.domain.graphql.request import DataOrigin, MetricRequest
from src.domain.graphql.ssot_enums import MetricType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import DistributionSpec
from src.shared.ssot_loader import get_metric_metadata


def _metric_is_numeric(metric_code: str) -> bool:
    code = (metric_code or "").strip().upper()
    if not code:
        return False
    meta = get_metric_metadata().get(code, {})
    data_type = meta.get("data_type")
    if isinstance(data_type, str) and data_type.strip().lower() == "numeric":
        return True
    return isinstance(meta.get("numeric"), dict)


def _series_data_origin(series: S.LineSeries) -> Optional[DataOrigin]:
    if series.data_origin is None:
        return None
    return DataOrigin.model_validate(
        series.data_origin.model_dump(by_alias=True, exclude_none=True)
    )


def _data_origin_key(data_origin: Optional[DataOrigin]) -> Optional[tuple]:
    if data_origin is None:
        return None
    return (
        tuple(sorted(data_origin.provider_group_id)) if data_origin.provider_group_id else None,
        tuple(sorted(data_origin.provider_id)) if data_origin.provider_id else None,
    )


def _build_line_series_request(
    plan_chart: S.LineChartSpec,
    series: S.LineSeries,
    derive_defaults_fn: Callable[[str], tuple[int, int, int]],
    axis_from_meta_fn: Callable[[str, int, int], tuple[ChartAxis, ChartAxis]],
    derived_axes_slot: List[Optional[tuple[ChartAxis, ChartAxis]]],
) -> MetricRequest:
    """Compile a canonical MetricRequest for a series based on its axis semantics.

    Determines request mode (distribution vs stats) from the axis type and metric
    data type. Series that share the same axis and metric compile to one request;
    per-category expansion is handled by the query compiler.
    """
    x_axis = plan_chart.x_axes[series.x_axis]
    y_axis = plan_chart.y_axes[series.y_axis]
    metric_code = (series.metric or "").strip().upper()

    if isinstance(x_axis, S.NumericMetricXAxis) and isinstance(y_axis, S.CountAxis):
        bins = x_axis.bins
        min_value = x_axis.min_value
        max_value = x_axis.max_value
        if bins is None or min_value is None or max_value is None:
            default_bins, rmin, rmax = derive_defaults_fn(x_axis.metric)
            bins = bins if bins is not None else default_bins
            min_value = min_value if min_value is not None else rmin
            max_value = max_value if max_value is not None else rmax
        if derived_axes_slot[0] is None:
            derived_axes_slot[0] = axis_from_meta_fn(x_axis.metric, int(min_value), int(max_value))
        return MetricRequest(metricType=MetricType(metric_code)).with_distribution(
            bin_count=int(bins),
            lower=int(min_value),
            upper=int(max_value),
        )

    if (
        isinstance(x_axis, S.CategoryXAxis)
        and isinstance(x_axis.group_by, (S.GroupBySex, S.GroupByStrokeType))
        and isinstance(y_axis, S.MetricValueAxis)
        and _metric_is_numeric(metric_code)
    ):
        bins, min_value, max_value = derive_defaults_fn(metric_code)
        if derived_axes_slot[0] is None:
            derived_axes_slot[0] = axis_from_meta_fn(metric_code, int(min_value), int(max_value))
        return MetricRequest(metricType=MetricType(metric_code)).with_distribution(
            bin_count=int(bins),
            lower=int(min_value),
            upper=int(max_value),
        )

    return MetricRequest(metricType=MetricType(metric_code)).with_stats()


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
        # Compile one canonical metric request per unique (x_axis_key, metric_code, data_origin).
        # Series that share the same axis and metric collapse into a single request;
        # per-category expansion is handled by the query compiler via combo enumeration.
        seen: Dict[tuple, bool] = {}
        derived_axes_slot: List[Optional[tuple[ChartAxis, ChartAxis]]] = [None]

        for series in plan_chart.series:
            metric_code = (series.metric or "").strip().upper()
            data_origin = _series_data_origin(series)
            key = (series.x_axis, metric_code, _data_origin_key(data_origin))
            if key in seen:
                continue
            seen[key] = True

            req = _build_line_series_request(
                plan_chart, series, derive_defaults_fn, axis_from_meta_fn, derived_axes_slot
            )
            metric_requests.append(req)
            metric_data_origins.append(data_origin)
            metric_scope_labels.append(_scope_label(series.origin_scope))

        derived_axes = derived_axes_slot[0]

    elif isinstance(plan_chart, S.HistogramChartSpec):
        metric = plan_chart.x_axis.metric
        bins = plan_chart.x_axis.bins
        min_value = plan_chart.x_axis.min_value
        max_value = plan_chart.x_axis.max_value
        if bins is None or min_value is None or max_value is None:
            default_bins, rmin, rmax = derive_defaults_fn(metric)
            bins = bins if bins is not None else default_bins
            min_value = min_value if min_value is not None else rmin
            max_value = max_value if max_value is not None else rmax

        distribution = DistributionSpec(
            num_buckets=int(bins), min_value=int(min_value), max_value=int(max_value)
        )
        derived_axes = axis_from_meta_fn(metric, distribution.min_value, distribution.max_value)

        metric_data_origin: Optional[DataOrigin] = None
        if plan_chart.data_origin is not None:
            metric_data_origin = DataOrigin.model_validate(
                plan_chart.data_origin.model_dump(by_alias=True, exclude_none=True)
            )

        metric_requests.append(
            MetricRequest(metricType=MetricType(metric)).with_distribution(
                bin_count=distribution.num_buckets,
                lower=distribution.min_value,
                upper=distribution.max_value,
            )
        )
        metric_data_origins.append(metric_data_origin)
        metric_scope_labels.append(_scope_label(plan_chart.origin_scope))

    return metric_requests, derived_axes, metric_data_origins, metric_scope_labels
