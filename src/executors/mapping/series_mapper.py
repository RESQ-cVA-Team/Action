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
    get_enum_option_key,
    is_angels_awards_metric,
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
            raise ValueError(f"Invalid ISO start date in time period: {start!r}") from exc
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


def _percent_values(percents: Optional[List[Any]], counts: List[Any]) -> List[float]:
    """Backend percents when they line up with the counts, else each count as a share of their total."""
    if isinstance(percents, list) and len(percents) == len(counts) and any(p is not None for p in percents):
        return [float(p) if isinstance(p, (int, float)) else 0.0 for p in percents]
    total = sum(float(c) for c in counts if isinstance(c, (int, float)))
    if total <= 0:
        return [0.0 for _ in counts]
    return [float(c) / total * 100.0 if isinstance(c, (int, float)) else 0.0 for c in counts]


def _first_percent(kpi1: Any) -> Optional[float]:
    """Share of cases for one grouped row: the backend's first percent (the
    first category, e.g. the yes of an Angels Awards metric), else the first
    count over the cohort size."""
    percents = getattr(kpi1, "percents", None)
    if isinstance(percents, list):
        for value in percents:
            if isinstance(value, (int, float)):
                return float(value)
    counts = getattr(kpi1, "case_count", None) or []
    cohort = getattr(kpi1, "cohort_size", None)
    if counts and isinstance(counts[0], (int, float)) and isinstance(cohort, (int, float)) and cohort > 0:
        return float(counts[0]) / float(cohort) * 100.0
    return None


def _aa_category_indices(
    metric_code: str,
    labels: List[str],
    include_category_detail: bool,
) -> List[tuple[int, str, str]]:
    desired_keys = {"yes", "no"} if include_category_detail else {"yes"}
    found: List[tuple[int, str, str]] = []
    for index, label in enumerate(labels):
        option_key = get_enum_option_key(metric_code, str(label))
        if option_key in desired_keys:
            found.append((index, str(label), option_key))

    found_keys = {option_key for _, _, option_key in found}
    missing_keys = desired_keys - found_keys
    if missing_keys:
        raise ValueError(f"AA metric {metric_code} is missing recognized enum category labels: {sorted(missing_keys)}")
    return found


def _aa_category_value(
    kpi1: Any,
    case_counts: List[Any],
    index: int,
    value_mode: str,
    derived_complement: bool,
) -> Optional[float]:
    if index >= len(case_counts) or not isinstance(case_counts[index], (int, float)):
        return None
    count = float(case_counts[index])
    if value_mode != "percent":
        return count

    percents = getattr(kpi1, "percents", None)
    if not derived_complement and isinstance(percents, list) and index < len(percents):
        if isinstance(percents[index], (int, float)):
            return float(percents[index])

    cohort_size = getattr(kpi1, "cohort_size", None)
    if isinstance(cohort_size, (int, float)) and cohort_size > 0:
        return count / float(cohort_size) * 100.0

    return _percent_values(None, case_counts)[index]


def map_metrics_payload_to_series(
    metrics_payload: Dict[str, Any],
    label_parts: List[str],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    add_time_period_labels: bool,
    scope_label: Optional[str] = None,
    batched_time_periods: Optional[List[Any]] = None,
    is_filter_grouped: bool = False,
    value_mode: str = "count",
    category_detail_metric_codes: Optional[List[str]] = None,
) -> List[ChartSeries]:
    series: List[ChartSeries] = []
    category_detail_codes = {code.strip().upper() for code in category_detail_metric_codes or []}

    for metric_name, metric in metrics_payload.items():
        for kpi_index, kpi in enumerate(metric.kpi_group):
            if getattr(kpi, "kpi1", None) is None:
                continue

            metric_code = _metric_code_from_alias(metric_name)
            is_aa_metric = is_angels_awards_metric(metric_code)
            include_category_detail = metric_code in category_detail_codes
            metric_labels = getattr(metric, "labels", None)
            case_counts = kpi.kpi1.case_count or []
            derived_complement = False

            if is_aa_metric and not metric_labels:
                metric_labels = get_enum_labels(metric_code)
                if metric_labels and len(case_counts) == 1 and isinstance(kpi.kpi1.cohort_size, (int, float)):
                    case_counts = [case_counts[0], kpi.kpi1.cohort_size - case_counts[0]]
                    derived_complement = True

            aa_categories = _aa_category_indices(metric_code, cast(List[str], metric_labels), include_category_detail) if is_aa_metric and metric_labels else []

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

                if is_aa_metric:
                    if len(case_counts) != len(metric_labels or []) and not (
                        not case_counts and add_time_period_labels and tp_start
                    ):
                        raise ValueError("AA categorical KPI payload has mismatched labels/caseCount lengths.")
                    for category_index, category_label, _ in aa_categories:
                        y_value = _aa_category_value(
                            kpi.kpi1,
                            case_counts,
                            category_index,
                            value_mode,
                            derived_complement,
                        )
                        if y_value is None and not (add_time_period_labels and tp_start):
                            raise ValueError("Grouped AA KPI row is missing the requested category value.")

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
                        if include_category_detail:
                            name_parts.append(category_label)
                        series.append(
                            ChartSeries(
                                name=" — ".join(name_parts),
                                data=[ChartPoint(x=x_value, y=y_value, label=x_value)],
                            )
                        )
                    continue

                y_value = _first_percent(kpi.kpi1) if value_mode == "percent" else None
                if y_value is None:
                    statistics = (kpi.kpi1.mean, kpi.kpi1.median) if value_mode == "mean" else (kpi.kpi1.median, kpi.kpi1.mean)
                    y_value = next((float(s) for s in statistics if isinstance(s, (int, float))), None)
                if y_value is None and kpi.kpi1.case_count:
                    try:
                        y_value = float(kpi.kpi1.case_count[0])
                    except (TypeError, ValueError, IndexError) as exc:
                        raise ValueError("Grouped KPI case_count must provide a numeric first value") from exc

                # Preserve explicit time buckets as missing points (y=None)
                # so frontend can render gaps instead of implied zeros.
                if y_value is None and not (add_time_period_labels and tp_start):
                    raise ValueError("Grouped KPI row is missing numeric y-value (median/mean/case_count[0]).")

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

            if not metric_labels and not kpi.kpi1.d1 and len(case_counts) == 1 and kpi.kpi1.cohort_size is not None:
                # Boolean-shaped Enum metric (e.g. WAKEUP_STROKE): backend returns a single
                # caseCount (the "true"/first-category count) with no labels array. Fall back
                # to the SSOT's own category labels and derive the complement count.
                ssot_labels = get_enum_labels(metric_code)
                if ssot_labels and len(ssot_labels) >= 2:
                    metric_labels = ssot_labels[:2]
                    case_counts = [case_counts[0], kpi.kpi1.cohort_size - case_counts[0]]
                    derived_complement = True

            if not kpi.kpi1.d1 and not metric_labels:
                raise ValueError("Distribution KPI payload is missing d1 histogram/series data.")

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
                bin_values: List[Any] = list(kpi.kpi1.d1.case_count)
                if value_mode == "percent":
                    bin_values = _percent_values(getattr(kpi.kpi1.d1, "percents", None), bin_values)
                points = _distribution_points_from_backend(metric_code, kpi.kpi1.d1.edges, bin_values)
            else:
                # Categorical (Enum) metric: labels and caseCount are parallel arrays,
                # one entry per category (e.g. male/female/unknown), not a numeric histogram.
                if len(case_counts) != len(metric_labels or []):
                    raise ValueError("Categorical KPI payload has mismatched labels/caseCount lengths.")
                category_values: List[Any] = list(case_counts)
                if is_aa_metric:
                    points = [
                        ChartPoint(
                            x=category_label,
                            y=_aa_category_value(
                                kpi.kpi1,
                                case_counts,
                                category_index,
                                value_mode,
                                derived_complement,
                            ),
                        )
                        for category_index, category_label, _ in aa_categories
                    ]
                elif value_mode == "percent":
                    # A boolean-shaped payload carries one backend percent and a
                    # complement count derived above, so share it out of the counts.
                    backend_percents = None if derived_complement else getattr(kpi.kpi1, "percents", None)
                    category_values = _percent_values(backend_percents, category_values)
                    points = [ChartPoint(x=label, y=float(count)) for label, count in zip(cast(List[str], metric_labels), category_values)]
                else:
                    points = [ChartPoint(x=label, y=float(count)) for label, count in zip(cast(List[str], metric_labels), category_values)]

            series.append(
                ChartSeries(
                    name=series_name,
                    data=points,
                )
            )

    return series
