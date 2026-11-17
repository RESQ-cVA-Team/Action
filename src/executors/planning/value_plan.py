"""A value is fetched like a chart with one explicit period: the same request
and series mapping give its number as that period's single point."""

from __future__ import annotations

from datetime import date
from typing import Tuple

from src.domain.graphql.request import default_time_bounds
from src.domain.langchain import schema as S
from src.domain.langchain.schema import GroupByTime
from src.executors.planning.query_compiler import Dimension


_MONTHS_PER_UNIT = {"YEAR": 12, "QUARTER": 3, "MONTH": 1}


def value_period(value: S.ValueSpec) -> Tuple[str, str]:
    """ISO start and end of the period a value covers. A window ends today and
    starts on a month (or day) boundary: the last year is the current month
    and the eleven before it, as a monthly chart of that window shows. No
    period is the default range."""
    period = value.period
    if isinstance(period, S.TimeRange):
        return period.start_date[:10], period.end_date[:10]
    if isinstance(period, S.TimeWindow):
        months = _MONTHS_PER_UNIT.get(period.unit)
        if months:
            grouping = GroupByTime(grain="MONTH", window=S.TimeWindow(last_n=period.last_n * months, unit="MONTH"))
        else:
            grouping = GroupByTime(grain="DAY", window=period)
        buckets = list(Dimension(grouping).categories())
        if not buckets:
            raise ValueError(f"Time window last {period.last_n} {period.unit} covers no period.")
        return buckets[0][0].isoformat(), min(buckets[-1][1], date.today()).isoformat()
    return default_time_bounds()


def value_chart(value: S.ValueSpec) -> S.ChartSpec:
    start, end = value_period(value)
    return S.ChartSpec(
        chart_type="LINE",
        metrics=[value.metric_spec()],
        filters=value.filters,
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=value.measure,
            time=S.TimeSemanticsSpec(periods=[S.TimeRange(start_date=start, end_date=end)]),
        ),
    )
