from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, cast

from src.domain.dto.charts import BarChart, ChartDTO, LineChart, union
from src.domain.dto.charts.box import BoxEntry, BoxPlot
from src.domain.dto.charts.histogram import Histogram, HistogramBin
from src.domain.dto.charts.pie import PieChart, PieSlice
from src.domain.dto.charts.radar import RadarChart
from src.domain.dto.charts.scatter import ScatterPlot
from src.domain.dto.charts.types import ChartAxis, ChartMetadata, ChartSeries, ChartType
from src.domain.dto.charts.waterfall import WaterfallChart, WaterfallStep
from src.domain.langchain import schema as S
from src.domain.langchain.schema import GroupByAge, GroupByCanonicalField, GroupByNIHSS, GroupBySex, GroupByStrokeType, GroupByTime
from src.executors.planning.query_compiler import Dimension
from src.shared.ssot_loader import get_canonical_display_name, get_metric_display_name, get_metric_metadata

logger = logging.getLogger(__name__)
_METRIC_METADATA = get_metric_metadata()


def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _flatten_y_values(series: List[ChartSeries]) -> List[float]:
    values: List[float] = []
    for s in series:
        for p in s.data:
            y = _coerce_float(p.y)
            if y is not None:
                values.append(y)
    return values


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _dimension_label(dimension: Dimension) -> Optional[str]:
    if isinstance(dimension.spec, GroupByTime):
        grain = getattr(dimension.spec, "grain", None)
        return str(grain or "time").lower()
    if isinstance(dimension.spec, GroupBySex):
        return "Sex"
    if isinstance(dimension.spec, GroupByStrokeType):
        return "Stroke Type"
    if isinstance(dimension.spec, GroupByNIHSS):
        return get_canonical_display_name("ADMISSION_NIHSS")
    if isinstance(dimension.spec, GroupByAge):
        return get_canonical_display_name("AGE")
    if isinstance(dimension.spec, GroupByCanonicalField):
        field = getattr(dimension.spec, "field", "")
        return get_canonical_display_name(str(field))
    return None


def _normalize_title_token(value: str) -> str:
    return (value or "").strip().replace("_", " ").lower()


def _metric_codes(plan_chart: S.ChartSpec) -> List[str]:
    out: List[str] = []
    if isinstance(plan_chart, S.LineChartSpec):
        for series in plan_chart.series:
            code = (series.metric or "").strip().upper()
            if code and code not in out:
                out.append(code)
    elif isinstance(plan_chart, S.HistogramChartSpec):
        code = (plan_chart.x_axis.metric or "").strip().upper()
        if code and code not in out:
            out.append(code)
    return out


def _format_operator(value: str) -> str:
    token = (value or "").strip().upper()
    mapping = {
        "GE": ">=",
        "GT": ">",
        "LE": "<=",
        "LT": "<",
        "EQ": "=",
        "NE": "!=",
    }
    return mapping.get(token, token)


def _format_filter_text(filter_node: Optional[Any], include_date: bool = True) -> str:
    if filter_node is None:
        return "all patients"

    def render(node: Any) -> str:
        if isinstance(node, S.AndFilter):
            parts = [render(child) for child in (node.clauses or [])]
            parts = [p for p in parts if p]
            return " and ".join(parts)
        if isinstance(node, S.OrFilter):
            parts = [render(child) for child in (node.clauses or [])]
            parts = [p for p in parts if p]
            return " or ".join(parts)
        if isinstance(node, S.NotFilter):
            inner = render(node.clause)
            return f"not ({inner})" if inner else ""
        if isinstance(node, S.PredicateFilter):
            field = (node.field or "").strip().upper()
            operator = _format_operator(node.operator)
            val = node.value if node.value is not None else (node.values[0] if node.values else "")
            if field in {"DISCHARGE_DATE", "DATE"}:
                if not include_date:
                    return ""
                return f"discharge date {operator} {val}"
            if field in {"AGE", "ADMISSION_AGE"}:
                return f"age {operator} {val:g}" if isinstance(val, (int, float)) else f"age {operator} {val}"
            if field in {"ADMISSION_NIHSS", "NIHSS"}:
                return f"nihss {operator} {val:g}" if isinstance(val, (int, float)) else f"nihss {operator} {val}"
            if field == "SEX":
                return f"sex = {_normalize_title_token(str(val))}"
            if field == "STROKE_TYPE":
                return f"stroke type = {_normalize_title_token(str(val))}"
            # Generic canonical field fallback
            field_label = _normalize_title_token(get_canonical_display_name(field))
            if isinstance(val, bool):
                return f"{field_label} = {'yes' if val else 'no'}"
            return f"{field_label} {operator} {val}"
        return ""

    rendered = render(filter_node).strip()
    return rendered or "all patients"


def _format_iso_date(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    candidate = token.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
        return parsed.date().isoformat()
    except ValueError:
        return token.split("T", 1)[0]


def _parse_iso_date(value: str) -> Optional[datetime]:
    token = (value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sample_period(filter_node: Optional[Any]) -> Optional[str]:
    if filter_node is None:
        return None

    date_constraints: List[tuple[str, str]] = []
    ambiguous = False

    def walk(node: Any, inside_or: bool = False) -> None:
        nonlocal ambiguous
        if node is None or ambiguous:
            return
        if isinstance(node, S.PredicateFilter) and (node.field or "").strip().upper() in {"DISCHARGE_DATE", "DATE"}:
            if inside_or:
                ambiguous = True
                return
            op = (node.operator or "").strip().upper()
            value = str(node.value if node.value is not None else (node.values[0] if node.values else "")).strip()
            if op and value:
                date_constraints.append((op, value))
            return
        if isinstance(node, S.AndFilter):
            for child in (node.clauses or []):
                walk(child, inside_or)
            return
        if isinstance(node, S.OrFilter):
            for child in (node.clauses or []):
                walk(child, True)
            return
        if isinstance(node, S.NotFilter):
            ambiguous = True

    walk(filter_node)
    if ambiguous or not date_constraints:
        return None

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    for op, raw in date_constraints:
        parsed = _parse_iso_date(raw)
        shown = _format_iso_date(raw)
        if parsed is None:
            continue

        if op in {"GE", "GT"}:
            if start is None or parsed > start:
                start = parsed
                start_text = shown
        elif op in {"LE", "LT"}:
            if end is None or parsed < end:
                end = parsed
                end_text = shown
        elif op == "EQ":
            start = parsed
            end = parsed
            start_text = shown
            end_text = shown

    if start_text and end_text:
        if start_text == end_text:
            return start_text
        return f"{start_text} to {end_text}"
    if start_text:
        return f"{start_text} onward"
    if end_text:
        return f"up to {end_text}"
    return None


def _metric_unit(metric_code: str) -> Optional[str]:
    metadata = _METRIC_METADATA.get((metric_code or "").upper()) or {}
    unit_any = metadata.get("unit")
    if isinstance(unit_any, str) and unit_any.strip():
        return unit_any.strip()

    numeric_any = metadata.get("numeric")
    if isinstance(numeric_any, dict):
        numeric_dict = cast(dict[str, Any], numeric_any)
        nested_unit = numeric_dict.get("unit")
        if isinstance(nested_unit, str) and nested_unit.strip():
            return nested_unit.strip()

    return None


def _title_case_token(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    if token.isupper() and len(token) <= 5:
        return token
    return token[:1].upper() + token[1:].lower()


def _axis_label_for_dimension(dimension: Dimension) -> str:
    if isinstance(dimension.spec, GroupByTime):
        grain = str(getattr(dimension.spec, "grain", "TIME") or "TIME").strip().upper()
        label_map = {
            "DAY": "Day",
            "WEEK": "Week",
            "BIWEEK": "Biweek",
            "MONTH": "Month",
            "QUARTER": "Quarter",
            "YEAR": "Year",
        }
        return label_map.get(grain, _title_case_token(grain))
    if isinstance(dimension.spec, GroupBySex):
        return "Sex"
    if isinstance(dimension.spec, GroupByStrokeType):
        return "Stroke Type"
    if isinstance(dimension.spec, GroupByNIHSS):
        return get_canonical_display_name("ADMISSION_NIHSS")
    if isinstance(dimension.spec, GroupByAge):
        return get_canonical_display_name("AGE")
    if isinstance(dimension.spec, GroupByCanonicalField):
        field = str(getattr(dimension.spec, "field", "") or "").strip().upper()
        return get_canonical_display_name(field) if field else "Category"
    return "Category"


def _axis_type_for_dimension(dimension: Dimension) -> ChartAxis.AxisType:
    if isinstance(dimension.spec, GroupByTime):
        return ChartAxis.AxisType.TIME
    return ChartAxis.AxisType.CATEGORY


def _primary_dimension_for_axes(dimensions: List[Dimension]) -> Optional[Dimension]:
    for dimension in dimensions:
        if isinstance(dimension.spec, GroupByTime):
            return dimension
    return dimensions[0] if dimensions else None


def _metric_value_axis_label(plan_chart: S.ChartSpec) -> str:
    metric_codes = _metric_codes(plan_chart)
    if len(metric_codes) != 1:
        return "Metric Value"

    metric_code = metric_codes[0]
    display = get_metric_display_name(metric_code)
    unit = _metric_unit(metric_code)
    return f"{display} ({unit})" if unit else display


def _derive_axes_from_dimensions(
    plan_chart: S.ChartSpec,
    dimensions: List[Dimension],
    chart_type_upper: str,
) -> tuple[Optional[ChartAxis], Optional[ChartAxis]]:
    if chart_type_upper in {ChartType.PIE.value, ChartType.RADAR.value}:
        return None, None

    primary = _primary_dimension_for_axes(dimensions)
    if primary is not None:
        x_axis = ChartAxis(
            label=_axis_label_for_dimension(primary),
            type=_axis_type_for_dimension(primary),
        )
    else:
        x_axis = ChartAxis(label="Category", type=ChartAxis.AxisType.CATEGORY)

    if chart_type_upper == ChartType.HISTOGRAM.value:
        y_axis_label = "Cases"
    else:
        y_axis_label = _metric_value_axis_label(plan_chart)

    y_axis = ChartAxis(label=y_axis_label, type=ChartAxis.AxisType.LINEAR)
    return x_axis, y_axis


def _fallback_across(plan_chart: S.ChartSpec, metric_codes: List[str]) -> str:
    chart_type = (plan_chart.chart_type or "").upper()
    if chart_type == ChartType.PIE.value:
        return "category"

    if len(metric_codes) != 1:
        return "value range"

    metric_code = metric_codes[0]
    metric_unit = _metric_unit(metric_code)

    if metric_unit:
        return f"value range in {metric_unit}"

    return "value range"


def _derive_title(plan_chart: S.ChartSpec, dimensions: List[Dimension], sampled_period_override: Optional[str] = None) -> str:
    metric_codes = _metric_codes(plan_chart)
    metrics_part = ", ".join(metric_codes) if metric_codes else get_metric_display_name(plan_chart.chart_type or "CHART")

    across_dim: Optional[Dimension] = None
    for dimension in dimensions:
        if isinstance(dimension.spec, GroupByTime):
            across_dim = dimension
            break
    if across_dim is None and dimensions:
        across_dim = dimensions[0]

    by_parts: List[str] = []
    for dimension in dimensions:
        if across_dim is not None and dimension is across_dim:
            continue
        label = _dimension_label(dimension)
        if not label:
            continue
        token = _normalize_title_token(label)
        if token and token not in by_parts:
            by_parts.append(token)

    if across_dim is None:
        across_part = _fallback_across(plan_chart, metric_codes)
    else:
        across_label = _dimension_label(across_dim) or _fallback_across(plan_chart, metric_codes)
        across_part = _normalize_title_token(across_label)

    filters_node = cast(Any, getattr(plan_chart, "filters", None))
    sampled_period = sampled_period_override or _sample_period(filters_node)
    filters_part = _format_filter_text(filters_node, include_date=sampled_period is None)

    title = metrics_part
    if by_parts:
        title += f" by {' × '.join(by_parts)}"
    if sampled_period:
        title += f" sampled from {sampled_period}"
    title += f" across {across_part} ({filters_part})"
    return title


def build_chart_dto(
    plan_chart: S.ChartSpec,
    dimensions: List[Dimension],
    series: List[ChartSeries],
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]],
    sampled_period_override: Optional[str] = None,
) -> ChartDTO:
    title_text = _derive_title(plan_chart, dimensions, sampled_period_override=sampled_period_override)
    chart_type_upper = (plan_chart.chart_type or "").upper()

    x_axis: Optional[ChartAxis] = None
    y_axis: Optional[ChartAxis] = None
    if derived_axes is not None:
        x_axis, y_axis = derived_axes
    else:
        x_axis, y_axis = _derive_axes_from_dimensions(
            plan_chart=plan_chart,
            dimensions=dimensions,
            chart_type_upper=chart_type_upper,
        )

    metadata = ChartMetadata(
        title=title_text,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    if chart_type_upper == ChartType.LINE.value:
        return LineChart(metadata=metadata, series=series)
    if chart_type_upper == ChartType.BAR.value:
        return BarChart(metadata=metadata, series=series)
    if chart_type_upper == ChartType.SCATTER.value:
        return ScatterPlot(metadata=metadata, series=series)
    if chart_type_upper == ChartType.AREA.value:
        return union.AreaChart(metadata=metadata, series=series)
    if chart_type_upper == ChartType.RADAR.value:
        axis_labels: List[str] = []
        for s in series:
            for p in s.data:
                x_label = str(p.x)
                if x_label not in axis_labels:
                    axis_labels.append(x_label)
        return RadarChart(metadata=metadata, series=series, axes=axis_labels)
    if chart_type_upper == ChartType.PIE.value:
        totals: dict[str, float] = {}
        for s in series:
            for p in s.data:
                key = str(p.x)
                y = _coerce_float(p.y)
                if y is None:
                    continue
                totals[key] = totals.get(key, 0.0) + y
        slices = [PieSlice(label=label, value=value) for label, value in totals.items()]
        return PieChart(metadata=metadata, data=slices)
    if chart_type_upper == ChartType.WATERFALL.value:
        steps: List[WaterfallStep] = []
        source = series[0].data if series else []
        for p in source:
            y = _coerce_float(p.y)
            if y is None:
                continue
            steps.append(WaterfallStep(label=str(p.x), value=y, is_positive=y >= 0))
        return WaterfallChart(metadata=metadata, data=steps)
    if chart_type_upper == ChartType.HISTOGRAM.value:
        bins: List[HistogramBin] = []
        source = series[0].data if series else []
        if source:
            for idx, point in enumerate(source):
                start = _coerce_float(point.x)
                end = start
                if idx + 1 < len(source):
                    end = _coerce_float(source[idx + 1].x)
                freq = _coerce_float(point.y)
                if start is None or end is None or freq is None:
                    continue
                bins.append(HistogramBin(range_start=start, range_end=end, frequency=freq))
        return Histogram(metadata=metadata, data=bins, bin_count=max(1, len(bins)))
    if chart_type_upper == ChartType.BOX.value:
        box_entries: List[BoxEntry] = []
        for item in series:
            values = sorted(_coerce_float(point.y) for point in item.data if _coerce_float(point.y) is not None)
            if not values:
                continue
            typed_values = cast(List[float], values)
            q1 = _quantile(typed_values, 0.25)
            median = _quantile(typed_values, 0.5)
            q3 = _quantile(typed_values, 0.75)
            box_entries.append(
                BoxEntry(
                    name=item.name,
                    q1=q1,
                    median=median,
                    q3=q3,
                    min=typed_values[0],
                    max=typed_values[-1],
                )
            )

        if not box_entries:
            return BoxPlot(metadata=metadata, data=[])
        return BoxPlot(metadata=metadata, data=box_entries)

    logger.warning("Chart type %s not yet implemented; defaulting to LINE rendering", plan_chart.chart_type)
    return LineChart(metadata=metadata, series=series)
