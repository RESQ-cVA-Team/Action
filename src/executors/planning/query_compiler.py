from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from typing import Any, List, Optional, Sequence, Tuple

from src.domain.graphql.request import (
    DateFilter,
    IntegerFilter,
    LogicalFilter,
    SexFilter,
    StrokeFilter,
    TimePeriod,
)
from src.domain.graphql.ssot_enums import GroupByType, Operator, SexType, StrokeType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import (
    GroupByAge,
    GroupByCanonicalField,
    GroupByNIHSS,
    GroupBySex,
    GroupBySpec,
    GroupByStrokeType,
    GroupByTime,
)
from src.shared.ssot_loader import get_sex_label, get_stroke_label
from src.util.coalesce import coalesce

logger = logging.getLogger(__name__)


def _collect_lc_date_bounds(node: Any) -> tuple[Optional[str], Optional[str]]:
    from src.domain.langchain.schema import AndFilter as LCAndFilter
    from src.domain.langchain.schema import DateFilter as LCDateFilter

    min_start: Optional[str] = None
    max_end: Optional[str] = None
    if node is None:
        return None, None
    if isinstance(node, LCAndFilter):
        for child in node.and_ or []:
            s, e = _collect_lc_date_bounds(child)
            if s and (min_start is None or s < min_start):
                min_start = s
            if e and (max_end is None or e > max_end):
                max_end = e
    elif isinstance(node, LCDateFilter):
        op = str(node.operator).upper()
        if op in ("GE", "GT"):
            min_start = node.value
        elif op in ("LE", "LT"):
            max_end = node.value
    return min_start, max_end


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except Exception:
        try:
            return datetime.fromisoformat(text).date()
        except Exception:
            return None


def _build_month_periods_from_bounds(start_iso: Optional[str], end_iso: Optional[str]) -> List[TimePeriod]:
    from calendar import monthrange

    start = _parse_iso_date(start_iso)
    end = _parse_iso_date(end_iso)
    if start is None or end is None:
        return []
    if start > end:
        start, end = end, start

    y, m = start.year, start.month
    end_key = (end.year, end.month)
    periods: List[TimePeriod] = []
    while (y, m) <= end_key:
        month_start = date(y, m, 1)
        month_end = date(y, m, monthrange(y, m)[1])
        periods.append(
            TimePeriod(
                startDate=month_start.isoformat(),
                endDate=month_end.isoformat(),
            )
        )
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

    return periods


def _build_recent_month_periods(month_count: int = 12) -> List[TimePeriod]:
    from calendar import monthrange

    if month_count <= 0:
        return []

    today = date.today()
    y, m = today.year, today.month
    periods: List[TimePeriod] = []

    for _ in range(month_count):
        month_start = date(y, m, 1)
        month_end = date(y, m, monthrange(y, m)[1])
        periods.append(
            TimePeriod(
                startDate=month_start.isoformat(),
                endDate=month_end.isoformat(),
            )
        )
        if m == 1:
            y -= 1
            m = 12
        else:
            m -= 1

    periods.reverse()
    return periods


def _compiler_log_context(event: str, operation: str, **fields: Any) -> dict[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "event": event,
        "operation": operation,
        "outcome": "degraded",
    }
    for key, value in fields.items():
        if value is None:
            continue
        context[key] = value
    return {"log_context": context}


def _groupby_type_values() -> set[str]:
    try:
        return {str(member.value).upper() for member in GroupByType}
    except Exception:
        logger.debug(
            "Failed to enumerate server-supported group-by fields; using empty supported-field set",
            exc_info=True,
            extra=_compiler_log_context(
                event="query_compiler.groupby_type_values.fallback",
                operation="_groupby_type_values",
            ),
        )
        return set()


_SUPPORTED_GROUPBY_FIELDS = _groupby_type_values()


def is_server_groupby_supported(field: str) -> bool:
    return (field or "").upper() in _SUPPORTED_GROUPBY_FIELDS


def _resolve_server_groupby_field(spec: GroupBySpec) -> Optional[str]:
    if isinstance(spec, GroupByCanonicalField):
        field = (spec.field or "").upper()
        return field if field in _SUPPORTED_GROUPBY_FIELDS else None

    candidates: List[str] = []
    if isinstance(spec, GroupBySex):
        candidates = ["SEX", "SEX_TYPE"]
    elif isinstance(spec, GroupByStrokeType):
        candidates = ["STROKE_TYPE"]

    for candidate in candidates:
        if candidate in _SUPPORTED_GROUPBY_FIELDS:
            return candidate
    return None


class Dimension:
    """Represents one grouping dimension and how to enumerate categories/filters."""

    def __init__(self, spec: GroupBySpec):
        self.spec = spec
        self.kind = type(spec)

    def canonical_field(self) -> Optional[str]:
        return _resolve_server_groupby_field(self.spec)

    def is_canonical(self) -> bool:
        return self.canonical_field() is not None

    def categories(self) -> Sequence[Any]:
        if isinstance(self.spec, GroupBySex):
            return list(self.spec.categories or list(SexType))
        if isinstance(self.spec, GroupByStrokeType):
            return list(self.spec.categories or list(StrokeType))
        if isinstance(self.spec, GroupByTime):
            time_spec = self.spec
            from calendar import monthrange
            from datetime import date, datetime

            def _month_bucket(year: int, month: int) -> tuple[date, date]:
                start_day = 1
                end_day = monthrange(year, month)[1]
                return date(year, month, start_day), date(year, month, end_day)

            def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
                total = (year * 12 + (month - 1)) + delta
                new_year = total // 12
                new_month = (total % 12) + 1
                return new_year, new_month

            def _parse_date(value: str) -> Optional[date]:
                text = (value or "").strip()
                if not text:
                    return None
                try:
                    return date.fromisoformat(text)
                except Exception:
                    try:
                        return datetime.fromisoformat(text).date()
                    except Exception:
                        logger.debug(
                            "Failed to parse GroupByTime date; returning None",
                            exc_info=True,
                            extra=_compiler_log_context(
                                event="query_compiler.groupby_time.date_parse_fallback",
                                operation="Dimension.categories",
                                dimension_type=type(self.spec).__name__,
                                grain=str(time_spec.grain).upper(),
                                raw_value=text,
                            ),
                        )
                        return None

            window = time_spec.window
            grain = str(time_spec.grain).upper()
            if isinstance(window, S.TimeWindow) and grain == "MONTH":
                unit = str(window.unit).upper()
                month_span = 0
                if unit == "MONTH":
                    month_span = window.last_n
                elif unit == "QUARTER":
                    month_span = window.last_n * 3
                elif unit == "YEAR":
                    month_span = window.last_n * 12

                if month_span <= 0:
                    return []

                today = date.today()
                buckets: list[tuple[date, date]] = []
                for i in range(month_span):
                    y, m = _shift_month(today.year, today.month, -i)
                    buckets.append(_month_bucket(y, m))
                buckets.reverse()
                return buckets

            if isinstance(window, S.TimeRange) and grain == "MONTH":
                start = _parse_date(window.start_date)
                end = _parse_date(window.end_date)
                if start is None or end is None:
                    return []
                if start > end:
                    start, end = end, start

                y, m = start.year, start.month
                end_key = (end.year, end.month)
                buckets: list[tuple[date, date]] = []
                while (y, m) <= end_key:
                    buckets.append(_month_bucket(y, m))
                    y, m = _shift_month(y, m, 1)
                return buckets

            def _quarter_bucket(year: int, quarter: int) -> tuple[date, date]:
                start_month = (quarter - 1) * 3 + 1
                end_month = start_month + 2
                end_day = monthrange(year, end_month)[1]
                return date(year, start_month, 1), date(year, end_month, end_day)

            def _shift_quarter(year: int, quarter: int, delta: int) -> tuple[int, int]:
                total = (year * 4 + (quarter - 1)) + delta
                new_year = total // 4
                new_quarter = (total % 4) + 1
                return new_year, new_quarter

            if isinstance(window, S.TimeWindow) and grain == "QUARTER":
                unit = str(window.unit).upper()
                quarter_span = 0
                if unit == "QUARTER":
                    quarter_span = window.last_n
                elif unit == "YEAR":
                    quarter_span = window.last_n * 4
                elif unit == "MONTH":
                    quarter_span = max(1, window.last_n // 3)

                if quarter_span <= 0:
                    return []

                today = date.today()
                current_quarter = (today.month - 1) // 3 + 1
                buckets: list[tuple[date, date]] = []
                for i in range(quarter_span):
                    y, q = _shift_quarter(today.year, current_quarter, -i)
                    buckets.append(_quarter_bucket(y, q))
                buckets.reverse()
                return buckets

            if isinstance(window, S.TimeRange) and grain == "QUARTER":
                start = _parse_date(window.start_date)
                end = _parse_date(window.end_date)
                if start is None or end is None:
                    return []
                if start > end:
                    start, end = end, start

                y = start.year
                q = (start.month - 1) // 3 + 1
                end_key = (end.year, (end.month - 1) // 3 + 1)
                buckets: list[tuple[date, date]] = []
                while (y, q) <= end_key:
                    buckets.append(_quarter_bucket(y, q))
                    y, q = _shift_quarter(y, q, 1)
                return buckets

            if grain == "QUARTER" and window is None:
                # No window specified - default to last 8 quarters (2 years)
                today = date.today()
                current_quarter = (today.month - 1) // 3 + 1
                buckets: list[tuple[date, date]] = []
                for i in range(8):
                    y, q = _shift_quarter(today.year, current_quarter, -i)
                    buckets.append(_quarter_bucket(y, q))
                buckets.reverse()
                return buckets

            return []
        if isinstance(self.spec, GroupByAge):
            return list(self.spec.buckets)
        if isinstance(self.spec, GroupByNIHSS):
            return list(self.spec.buckets)
        return []

    def label_for(self, cat: Any) -> str:
        if isinstance(self.spec, GroupBySex):
            val = cat if isinstance(cat, SexType) else SexType(cat)
            raw = getattr(val, "value", str(val))
            return get_sex_label(str(raw).upper())
        if isinstance(self.spec, GroupByStrokeType):
            val = cat if isinstance(cat, StrokeType) else StrokeType(cat)
            raw = getattr(val, "value", str(val))
            return get_stroke_label(str(raw).upper())
        if isinstance(self.spec, (GroupByAge, GroupByNIHSS)):
            return f"{cat.min}-{cat.max}"
        if isinstance(self.spec, GroupByCanonicalField):
            return self.spec.field
        if isinstance(self.spec, GroupByTime):
            time_spec = self.spec
            try:
                start, end = cat
                return f"{start.isoformat()} to {end.isoformat()}"
            except Exception:
                logger.debug(
                    "Failed to format GroupByTime label; using grain fallback",
                    exc_info=True,
                    extra=_compiler_log_context(
                        event="query_compiler.groupby_time.label_fallback",
                        operation="Dimension.label_for",
                        dimension_type=type(self.spec).__name__,
                        grain=str(time_spec.grain).upper(),
                    ),
                )
                return time_spec.grain
        return str(cat)

    def filter_for(self, cat: Any) -> Optional[Any]:
        if isinstance(self.spec, GroupBySex):
            val = cat if isinstance(cat, SexType) else SexType(cat)
            return SexFilter(sexType=val)
        if isinstance(self.spec, GroupByStrokeType):
            val = cat if isinstance(cat, StrokeType) else StrokeType(cat)
            return StrokeFilter(strokeType=val)
        if isinstance(self.spec, GroupByAge):
            return LogicalFilter(
                operator="AND",
                children=[
                    IntegerFilter(property="AGE", operator=Operator("GE"), value=cat.min),
                    IntegerFilter(property="AGE", operator=Operator("LT"), value=cat.max),
                ],
            )
        if isinstance(self.spec, GroupByNIHSS):
            return LogicalFilter(
                operator="AND",
                children=[
                    IntegerFilter(
                        property="ADMISSION_NIHSS",
                        operator=Operator("GE"),
                        value=cat.min,
                    ),
                    IntegerFilter(
                        property="ADMISSION_NIHSS",
                        operator=Operator("LT"),
                        value=cat.max,
                    ),
                ],
            )
        if isinstance(self.spec, GroupByTime):
            time_spec = self.spec
            try:
                start, end = cat
            except Exception:
                logger.debug(
                    "Failed to unpack GroupByTime category; skipping filter generation",
                    exc_info=True,
                    extra=_compiler_log_context(
                        event="query_compiler.groupby_time.filter_fallback",
                        operation="Dimension.filter_for",
                        dimension_type=type(self.spec).__name__,
                        grain=str(time_spec.grain).upper(),
                    ),
                )
                return None
            return LogicalFilter(
                operator="AND",
                children=[
                    DateFilter(
                        property="DISCHARGE_DATE",
                        operator=Operator("GE"),
                        value=start.isoformat(),
                    ),
                    DateFilter(
                        property="DISCHARGE_DATE",
                        operator=Operator("LE"),
                        value=end.isoformat(),
                    ),
                ],
            )
        return None


@dataclass
class CompiledBatch:
    server_groupby: Optional[str]
    filter_dims: List[Dimension]
    combos_list: List[Tuple[Any, ...]]
    batched_time_enabled: bool
    batched_time_periods: List[TimePeriod]

    @property
    def request_count(self) -> int:
        return len(self.combos_list)


@dataclass
class CompiledChartGrouping:
    dimensions: List[Dimension]
    batches: List[CompiledBatch]

    @property
    def total_requests(self) -> int:
        return sum(b.request_count for b in self.batches)


def compile_chart_grouping(chart: S.ChartSpec) -> CompiledChartGrouping:
    collected_groups: List[GroupBySpec] = list(coalesce(chart.group_by, []))

    seen: set[GroupBySpec] = set()
    uniq_groups: List[GroupBySpec] = []
    for g in collected_groups:
        if g not in seen:
            seen.add(g)
            uniq_groups.append(g)
    dims: List[Dimension] = [Dimension(g) for g in uniq_groups]

    canonical_dims: List[Dimension] = [d for d in dims if d.is_canonical()]
    server_dim: Optional[Dimension] = canonical_dims[0] if canonical_dims else None

    filter_dims_all: List[Dimension] = [d for d in dims if d is not server_dim]

    time_dims: List[Dimension] = [d for d in filter_dims_all if isinstance(d.spec, GroupByTime)]
    batched_time_periods: List[TimePeriod] = []
    batched_time_enabled = len(time_dims) == 1
    batched_time_dim: Optional[Dimension] = time_dims[0] if batched_time_enabled else None
    if batched_time_enabled and batched_time_dim is not None:
        batched_time_spec = batched_time_dim.spec
        if not isinstance(batched_time_spec, GroupByTime):
            batched_time_enabled = False
            batched_time_dim = None
        else:
            for cat in batched_time_dim.categories():
                try:
                    start, end = cat
                    batched_time_periods.append(TimePeriod(startDate=start.isoformat(), endDate=end.isoformat()))
                except Exception:
                    logger.debug(
                        "Failed to build batched time period; skipping invalid time bucket",
                        exc_info=True,
                        extra=_compiler_log_context(
                            event="query_compiler.batched_time_period.skipped",
                            operation="compile_chart_grouping",
                            chart_type=chart.chart_type,
                            dimension_type=type(batched_time_dim.spec).__name__,
                            grain=str(batched_time_spec.grain).upper(),
                        ),
                    )
                    continue
    if batched_time_enabled and not batched_time_periods and batched_time_dim is not None:
        batched_time_spec = batched_time_dim.spec
        if isinstance(batched_time_spec, GroupByTime):
            grain = str(batched_time_spec.grain).upper()
            if grain == "MONTH":
                bound_start, bound_end = _collect_lc_date_bounds(chart.filters)
                batched_time_periods = _build_month_periods_from_bounds(bound_start, bound_end)
                if not batched_time_periods:
                    # Keep monthly chart semantics stable when planner omitted
                    # explicit time window/date filters.
                    batched_time_periods = _build_recent_month_periods(12)

    if batched_time_enabled and not batched_time_periods:
        batched_time_enabled = False
        batched_time_dim = None

    # After batched_time_periods is built, constrain to DateFilter bounds if present
    if batched_time_enabled and batched_time_periods:
        bound_start, bound_end = _collect_lc_date_bounds(chart.filters)
        if bound_start or bound_end:
            filtered_periods: List[TimePeriod] = []
            for tp in batched_time_periods:
                tp_start = tp.start_date
                tp_end = tp.end_date
                if bound_start and tp_end and tp_end < bound_start:
                    continue
                if bound_end and tp_start and tp_start > bound_end:
                    continue
                filtered_periods.append(tp)
            if filtered_periods:
                batched_time_periods = filtered_periods
            elif batched_time_dim is not None and isinstance(batched_time_dim.spec, GroupByTime):
                grain = str(batched_time_dim.spec.grain).upper()
                if grain == "MONTH":
                    rebuilt = _build_month_periods_from_bounds(bound_start, bound_end)
                    if rebuilt:
                        batched_time_periods = rebuilt

    filter_dims: List[Dimension]
    if batched_time_enabled and batched_time_dim is not None:
        filter_dims = [d for d in filter_dims_all if d is not batched_time_dim]
    else:
        filter_dims = filter_dims_all

    effective_filter_dims: List[Dimension] = []
    filter_categories: List[Sequence[Any]] = []
    for d in filter_dims:
        cats = d.categories()
        if not cats:
            continue
        sample_filter = d.filter_for(cats[0])
        if sample_filter is None:
            continue
        effective_filter_dims.append(d)
        filter_categories.append(cats)

    if not filter_categories:
        combos_list: List[Tuple[Any, ...]] = [tuple()]
    else:
        combos_list = list(product(*filter_categories))

    batches: List[CompiledBatch] = [
        CompiledBatch(
            server_groupby=server_dim.canonical_field() if server_dim is not None else None,
            filter_dims=effective_filter_dims,
            combos_list=combos_list,
            batched_time_enabled=batched_time_enabled,
            batched_time_periods=batched_time_periods,
        )
    ]

    return CompiledChartGrouping(dimensions=dims, batches=batches)


def estimate_query_count_for_chart(chart: S.ChartSpec) -> int:
    compiled = compile_chart_grouping(chart)
    return max(compiled.total_requests, 1)


def estimate_query_count_for_plan(plan: S.AnalysisPlan) -> int:
    plan_charts = coalesce(plan.charts, [])
    if not plan_charts:
        return 0
    total = 0
    for c in plan_charts:
        total += estimate_query_count_for_chart(c)
    return total
