from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.graphql.request import TimePeriod
from src.executors.planning.ssot_metric_defaults import (
    build_distribution_bin_ranges,
    format_distribution_point_label,
    get_enum_labels,
)
from src.shared.ssot_loader import get_enum_option_label, get_metric_display_name


def _metric_code_from_alias(metric_alias: str) -> str:
    code = metric_alias
    if code.lower().startswith("metric_"):
        code = code[len("metric_") :]
    return code.upper()


def metric_label_from_alias(metric_alias: str) -> str:
    return get_metric_display_name(_metric_code_from_alias(metric_alias))


def _is_calendar_month(start_dt: datetime, end_dt: datetime) -> bool:
    if start_dt.year != end_dt.year or start_dt.month != end_dt.month or start_dt.day != 1:
        return False
    return end_dt.day == calendar.monthrange(start_dt.year, start_dt.month)[1]


def _is_calendar_quarter(start_dt: datetime, end_dt: datetime) -> bool:
    if start_dt.day != 1 or start_dt.month not in (1, 4, 7, 10):
        return False
    last_month = start_dt.month + 2
    if end_dt.year != start_dt.year or end_dt.month != last_month:
        return False
    return end_dt.day == calendar.monthrange(end_dt.year, last_month)[1]


def _is_calendar_year(start_dt: datetime, end_dt: datetime) -> bool:
    return (start_dt.month, start_dt.day) == (1, 1) and (end_dt.year, end_dt.month, end_dt.day) == (start_dt.year, 12, 31)


def _grouped_time_period_label(tp_start: str, tp_end: Optional[str]) -> str:
    """Label for one bucket of a grouped/time-series chart.

    Previously hardcoded to "%Y-%m" for every bucket regardless of its real
    span, which collapsed e.g. 30 distinct single-day buckets onto 1-2 month
    labels -- a "last 30 days, daily" line chart rendered as if it had ~2
    data points instead of 30. A bucket that's a genuine, calendar-aligned
    month/quarter/year (start/end fall exactly on that period's boundaries)
    still gets the clean "%Y-%m" / "%Y-Qn" / "%Y" label those grains are
    meant to show; anything else (a single day, a week, or any other
    non-calendar-aligned span) gets full date precision so distinct buckets
    can never share a label.
    """
    try:
        start_dt = datetime.fromisoformat(tp_start)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date in grouped time period label: {tp_start!r}") from exc

    end_dt: Optional[datetime] = None
    if tp_end:
        try:
            end_dt = datetime.fromisoformat(tp_end)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date in grouped time period label: {tp_end!r}") from exc

    if end_dt is not None:
        if _is_calendar_year(start_dt, end_dt):
            return start_dt.strftime("%Y")
        if _is_calendar_quarter(start_dt, end_dt):
            quarter = (start_dt.month - 1) // 3 + 1
            return f"{start_dt.year}-Q{quarter}"
        if _is_calendar_month(start_dt, end_dt):
            return start_dt.strftime("%Y-%m")

    return start_dt.strftime("%Y-%m-%d")


def period_to_label(tp: TimePeriod) -> str:
    start = tp.start_date
    end = tp.end_date
    if isinstance(start, str) and start:
        try:
            dt = datetime.fromisoformat(start)
            return dt.strftime("%Y-%m")
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO start date in time period: {start!r}"
            ) from exc
    if isinstance(start, str) and isinstance(end, str) and start and end:
        return f"{start} to {end}"
    if isinstance(start, str) and start:
        return start
    if isinstance(end, str) and end:
        return end
    return "period"


def merge_series_by_name(series: List[ChartSeries]) -> List[ChartSeries]:
    merged: Dict[str, ChartSeries] = {}
    ordered_names: List[str] = []

    for item in series:
        existing = merged.get(item.name)
        if existing is None:
            merged[item.name] = ChartSeries(name=item.name, data=list(item.data), color=item.color, style=item.style)
            ordered_names.append(item.name)
        else:
            existing.data.extend(item.data)

    return [merged[name] for name in ordered_names]


def _origin_label_from_kpi_group(kpi_group: Any) -> Optional[str]:
    origin = getattr(kpi_group, "data_origin", None)
    if origin is None:
        return None

    provider_id = getattr(origin, "provider_id", None)
    if isinstance(provider_id, int):
        return f"Provider {provider_id}"

    provider_group_id = getattr(origin, "provider_group_id", None)
    if isinstance(provider_group_id, int):
        return f"Group {provider_group_id}"

    custom_group_name = getattr(origin, "custom_group_name", None)
    if isinstance(custom_group_name, str) and custom_group_name.strip():
        return custom_group_name.strip()

    return None


def _distribution_points_from_backend(metric_code: str, edges: List[Any], case_counts: List[Any]) -> List[ChartPoint]:
    ranges = build_distribution_bin_ranges(edges)
    points: List[ChartPoint] = []
    for (start, end), frequency in zip(ranges, case_counts):
        points.append(
            ChartPoint(
                x=start,
                y=float(frequency),
                label=format_distribution_point_label(metric_code, start, end),
            )
        )
    return points


def map_metrics_payload_to_series(
    metrics_payload: Dict[str, Any],
    label_parts: List[str],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    add_time_period_labels: bool,
    scope_label: Optional[str] = None,
    batched_time_periods: Optional[List[Any]] = None,
    is_filter_grouped: bool = False,
) -> List[ChartSeries]:
    series: List[ChartSeries] = []

    for metric_name, metric in metrics_payload.items():
        for kpi_index, kpi in enumerate(metric.kpi_group):
            if getattr(kpi, "kpi1", None) is None:
                continue

            server_label = kpi.grouped_by.group_item_name if kpi.grouped_by else None
            origin_label = _origin_label_from_kpi_group(kpi)

            # is_filter_grouped: this request's category (an age/NIHSS bucket, a
            # boolean split) was realized as a client-side case filter rather
            # than a server groupBy -- group_by_field is correctly None for
            # these (GraphQL has no native groupBy for arbitrary buckets), but
            # each such request still represents exactly one category, so it
            # needs the same single-aggregate-point-per-series treatment as a
            # real server groupBy, not the plain-distribution-histogram path.
            is_grouped_or_time = bool(group_by_field) or add_time_period_labels or is_filter_grouped
            if is_grouped_or_time:
                x_value: str
                tp_start: Optional[str] = None
                tp_end: Optional[str] = None
                if kpi.time_period is not None:
                    tp_start = kpi.time_period.start_date
                    tp_end = kpi.time_period.end_date
                elif add_time_period_labels and batched_time_periods:
                    # kpi_index = list(metric.kpi_group).index(kpi)
                    if kpi_index < len(batched_time_periods):
                        tp = batched_time_periods[kpi_index]
                        tp_start = getattr(tp, "startDate", None) or getattr(tp, "start_date", None)
                        tp_end = getattr(tp, "endDate", None) or getattr(tp, "end_date", None)

                if add_time_period_labels and tp_start:
                    x_value = _grouped_time_period_label(str(tp_start), str(tp_end) if tp_end else None)
                elif server_label:
                    mapped = get_enum_option_label(group_by_field, server_label) if group_by_field else None
                    x_value = mapped or server_label
                else:
                    x_value = "value"

                y_value: Optional[float] = None
                if isinstance(kpi.kpi1.median, (int, float)):
                    y_value = float(kpi.kpi1.median)
                elif isinstance(kpi.kpi1.mean, (int, float)):
                    y_value = float(kpi.kpi1.mean)
                elif kpi.kpi1.case_count:
                    try:
                        y_value = float(kpi.kpi1.case_count[0])
                    except (TypeError, ValueError, IndexError) as exc:
                        raise ValueError(
                            "Grouped KPI case_count must provide a numeric first value"
                        ) from exc

                # Preserve explicit time buckets as missing points (y=None)
                # so frontend can render gaps instead of implied zeros.
                if y_value is None and not (add_time_period_labels and tp_start):
                    raise ValueError(
                        "Grouped KPI row is missing numeric y-value (median/mean/case_count[0])."
                    )

                name_parts: List[str] = []
                if include_metric_alias:
                    name_parts.append(metric_label_from_alias(metric_name))
                name_parts.extend([part for part in label_parts if part])
                if origin_label:
                    name_parts.append(origin_label)
                if scope_label and scope_label not in name_parts:
                    name_parts.append(scope_label)
                if not name_parts:
                    name_parts.append(metric_label_from_alias(metric_name))
                series_name = " — ".join(name_parts)
                series.append(
                    ChartSeries(
                        name=series_name,
                        data=[ChartPoint(x=x_value, y=y_value, label=x_value)],
                    )
                )
                continue

            metric_labels = getattr(metric, "labels", None)
            metric_code = _metric_code_from_alias(metric_name)
            case_counts = kpi.kpi1.case_count or []

            if not metric_labels and not kpi.kpi1.d1 and len(case_counts) == 1 and kpi.kpi1.cohort_size is not None:
                # Boolean-shaped Enum metric (e.g. WAKEUP_STROKE): backend returns a single
                # caseCount (the "true"/first-category count) with no labels array. Fall back
                # to the SSOT's own category labels and derive the complement count.
                ssot_labels = get_enum_labels(metric_code)
                if ssot_labels and len(ssot_labels) >= 2:
                    metric_labels = ssot_labels[:2]
                    case_counts = [case_counts[0], kpi.kpi1.cohort_size - case_counts[0]]

            if not kpi.kpi1.d1 and not metric_labels:
                raise ValueError(
                    "Distribution KPI payload is missing d1 histogram/series data."
                )

            parts: List[str] = []
            if include_metric_alias:
                parts.append(metric_label_from_alias(metric_name))
            parts.extend([part for part in label_parts if part])
            if origin_label:
                parts.append(origin_label)
            if server_label:
                mapped = get_enum_option_label(group_by_field, server_label) if group_by_field else None
                parts.append(mapped or server_label)
            if scope_label and scope_label not in parts:
                parts.append(scope_label)

            if add_time_period_labels and kpi.time_period is not None:
                start = kpi.time_period.start_date
                end = kpi.time_period.end_date
                if isinstance(start, str) and isinstance(end, str):
                    parts.append(f"{start} to {end}")
                elif isinstance(start, str):
                    parts.append(start)
                elif isinstance(end, str):
                    parts.append(end)

            series_name = " — ".join(parts) if parts else metric_label_from_alias(metric_name)

            if kpi.kpi1.d1:
                points = _distribution_points_from_backend(metric_code, kpi.kpi1.d1.edges, kpi.kpi1.d1.case_count)
            else:
                # Categorical (Enum) metric: labels and caseCount are parallel arrays,
                # one entry per category (e.g. male/female/unknown), not a numeric histogram.
                if len(case_counts) != len(metric_labels or []):
                    raise ValueError(
                        "Categorical KPI payload has mismatched labels/caseCount lengths."
                    )
                points = [
                    ChartPoint(x=label, y=float(count))
                    for label, count in zip(cast(List[str], metric_labels), case_counts)
                ]

            series.append(
                ChartSeries(
                    name=series_name,
                    data=points,
                )
            )

    return series
