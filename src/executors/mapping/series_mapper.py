from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.graphql.request import TimePeriod
from src.shared.ssot_loader import get_enum_option_label, get_metric_display_name


def metric_label_from_alias(metric_alias: str) -> str:
    code = metric_alias
    if code.lower().startswith("metric_"):
        code = code[len("metric_") :]
    return get_metric_display_name(code.upper())


def period_to_label(tp: TimePeriod) -> str:
    start = tp.start_date
    end = tp.end_date
    if isinstance(start, str) and start:
        try:
            dt = datetime.fromisoformat(start)
            return dt.strftime("%Y-%m")
        except Exception:
            pass
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


def map_metrics_payload_to_series(
    metrics_payload: Dict[str, Any],
    label_parts: List[str],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    add_time_period_labels: bool,
) -> List[ChartSeries]:
    series: List[ChartSeries] = []

    for metric_name, metric in metrics_payload.items():
        for kpi in metric.kpi_group:
            server_label = kpi.grouped_by.group_item_name if kpi.grouped_by else None
            origin_label = _origin_label_from_kpi_group(kpi)

            # Some plans (e.g., GroupBySex when backend groupBy enum is unavailable)
            # are compiled into multiple filtered requests, one per category.
            # In that case `group_by_field` is None, but non-empty label_parts
            # still indicate grouped-style output should be produced from stats.
            is_grouped_or_time = bool(group_by_field) or add_time_period_labels or bool(label_parts)
            if is_grouped_or_time:
                y_value: Optional[float] = None
                if isinstance(kpi.kpi1.mean, (int, float)):
                    y_value = float(kpi.kpi1.mean)
                elif isinstance(kpi.kpi1.median, (int, float)):
                    y_value = float(kpi.kpi1.median)
                elif kpi.kpi1.case_count:
                    try:
                        y_value = float(kpi.kpi1.case_count[0])
                    except Exception:
                        y_value = None
                if y_value is None:
                    continue

                x_value: str
                if add_time_period_labels and kpi.time_period is not None:
                    start = kpi.time_period.start_date
                    end = kpi.time_period.end_date
                    if isinstance(start, str) and start:
                        try:
                            x_value = datetime.fromisoformat(start).strftime("%Y-%m")
                        except Exception:
                            x_value = start
                    elif isinstance(end, str) and end:
                        x_value = end
                    else:
                        x_value = label_parts[-1] if label_parts else "period"
                elif server_label:
                    mapped = get_enum_option_label(group_by_field, server_label) if group_by_field else None
                    x_value = mapped or server_label
                elif label_parts:
                    x_value = label_parts[-1]
                else:
                    x_value = "value"

                name_parts: List[str] = []
                if include_metric_alias:
                    name_parts.append(metric_label_from_alias(metric_name))
                name_parts.extend([part for part in label_parts if part])
                if origin_label:
                    name_parts.append(origin_label)
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

            if not kpi.kpi1.d1:
                continue

            parts: List[str] = []
            if include_metric_alias:
                parts.append(metric_label_from_alias(metric_name))
            parts.extend([part for part in label_parts if part])
            if origin_label:
                parts.append(origin_label)
            if server_label:
                mapped = get_enum_option_label(group_by_field, server_label) if group_by_field else None
                parts.append(mapped or server_label)

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
            series.append(
                ChartSeries(
                    name=series_name,
                    data=[ChartPoint(x=x, y=y) for x, y in zip(kpi.kpi1.d1.edges, kpi.kpi1.d1.case_count)],
                )
            )

    return series
