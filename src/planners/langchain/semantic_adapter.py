from __future__ import annotations

from typing import List

from src.domain.dto.execution_summary import PlanNormalizationSummary
from src.domain.dto.semantic_plan import SemanticChart, SemanticMetric, SemanticPlan, SemanticYAxis
from src.domain.langchain import schema as S
from src.shared.ssot_loader import resolve_chart_type, resolve_groupby_canonical


def _normalize_chart_type(chart_type: str | None) -> tuple[str, bool, bool]:
    raw = (chart_type or "").strip()
    resolved = resolve_chart_type(raw)
    if isinstance(resolved, str) and resolved:
        return resolved, resolved != (chart_type or ""), False

    return "LINE", True, bool(raw)


def _normalize_metric_code(metric_code: str) -> str:
    return (metric_code or "").strip().upper()


def _dedupe_group_by(group_by: List[S.GroupBySpec]) -> List[S.GroupBySpec]:
    seen: set[S.GroupBySpec] = set()
    out: List[S.GroupBySpec] = []
    for g in group_by:
        if g in seen:
            continue
        seen.add(g)
        out.append(g)
    return out


def _chart_group_by(chart: S.ChartSpec | SemanticChart) -> List[S.GroupBySpec]:
    out: List[S.GroupBySpec] = []
    if isinstance(chart.x_axis, S.TimeXAxis):
        out.append(
            S.GroupByTime(
                grain=chart.x_axis.grain,
                window=chart.x_axis.window,
                include_partial=chart.x_axis.include_partial,
            )
        )
    elif isinstance(chart.x_axis, S.CategoryXAxis):
        out.append(chart.x_axis.group_by)

    series_by = chart.series_by
    if series_by is not None:
        out.append(series_by.split_by)
    return out


def _normalize_group_by(group_by: List[S.GroupBySpec]) -> tuple[List[S.GroupBySpec], int, int]:
    normalized: List[S.GroupBySpec] = []
    normalized_canonical_fields = 0
    dropped_invalid_fields = 0

    for g in group_by:
        if isinstance(g, S.GroupByCanonicalField):
            resolved = resolve_groupby_canonical(g.field)
            if resolved is None:
                dropped_invalid_fields += 1
                continue

            if resolved != g.field:
                normalized_canonical_fields += 1
            normalized.append(S.GroupByCanonicalField(field=resolved, values=g.values))
            continue

        normalized.append(g)

    return normalized, normalized_canonical_fields, dropped_invalid_fields


def _normalize_chart_type_for_semantics(
    chart_type: str,
    x_axis: S.XAxisSpec,
    series_by: S.SeriesSpec | None,
) -> str:
    if isinstance(x_axis, S.NumericXAxis) and series_by is not None:
        return "BAR"
    return chart_type


def _has_time_groupby(group_by: List[S.GroupBySpec]) -> bool:
    return any(type(g).__name__ == "GroupByTime" for g in group_by)


def _has_non_time_groupby(group_by: List[S.GroupBySpec]) -> bool:
    return any(type(g).__name__ != "GroupByTime" for g in group_by)


def _normalize_group_by_spec(spec: S.GroupBySpec) -> tuple[S.GroupBySpec | None, int, int]:
    if isinstance(spec, S.GroupByCanonicalField):
        resolved = resolve_groupby_canonical(spec.field)
        if resolved is None:
            return None, 0, 1
        if resolved != spec.field:
            return S.GroupByCanonicalField(field=resolved, values=spec.values), 1, 0
        return spec, 0, 0
    return spec, 0, 0


def _normalize_chart_axes(chart: S.ChartSpec) -> tuple[S.XAxisSpec | None, S.SeriesSpec | None, int, int]:
    normalized_fields_count = 0
    dropped_invalid_fields = 0

    x_axis: S.XAxisSpec | None = chart.x_axis
    if isinstance(chart.x_axis, S.CategoryXAxis):
        spec, norm_count, drop_count = _normalize_group_by_spec(chart.x_axis.group_by)
        normalized_fields_count += norm_count
        dropped_invalid_fields += drop_count
        if spec is None:
            x_axis = None
        else:
            x_axis = S.CategoryXAxis(groupBy=spec, order=chart.x_axis.order)

    series_by: S.SeriesSpec | None = chart.series_by
    if chart.series_by is not None:
        spec, norm_count, drop_count = _normalize_group_by_spec(chart.series_by.split_by)
        normalized_fields_count += norm_count
        dropped_invalid_fields += drop_count
        if spec is None:
            series_by = None
        else:
            series_by = S.SeriesSpec(splitBy=spec)

    return x_axis, series_by, normalized_fields_count, dropped_invalid_fields


def validate_analysis_plan_semantics(plan: S.AnalysisPlan) -> S.AnalysisPlan:
    """Normalize and validate chart semantics from explicit axes/series structure."""

    # DISTRIBUTION + GroupByTime is not supported by the executor: it fetches one flat
    # distribution and silently drops the time grain.  Fail fast here so the planner retries
    # with corrective feedback instead of producing a silently wrong query.
    dist_time_indices: List[int] = []
    for idx, chart in enumerate(plan.charts or []):
        if isinstance(chart.x_axis, S.NumericXAxis) and _has_time_groupby(_chart_group_by(chart)):
            dist_time_indices.append(idx + 1)
    if dist_time_indices:
        joined = ", ".join(str(i) for i in dist_time_indices)
        raise ValueError(
            f"Chart(s) {joined}: DISTRIBUTION analysis mode cannot be combined with GroupByTime. "
            "To show value spread use DISTRIBUTION with no time grouping. "
            "To show a monthly/yearly trend use TIME_SERIES with GroupByTime."
        )

    # Single-metric comparison requests without a real comparison dimension are usually the
    # planner drifting into a stats query that the analytics backend cannot satisfy.
    comparison_only_time_indices: List[int] = []
    for idx, chart in enumerate(plan.charts or []):
        if not isinstance(chart.x_axis, (S.CategoryXAxis, S.ScopeXAxis)):
            continue
        metrics = [m for axis in chart.y_axes for m in axis.metrics]
        if len(metrics) != 1:
            continue
        group_by = _chart_group_by(chart)
        if not group_by or not _has_non_time_groupby(group_by):
            comparison_only_time_indices.append(idx + 1)
    if comparison_only_time_indices:
        joined = ", ".join(str(i) for i in comparison_only_time_indices)
        raise ValueError(
            f"Chart(s) {joined}: COMPARISON analysis mode needs a real comparison dimension such as sex, provider, "
            "or another categorical group. For a single metric over time, use TIME_SERIES with GroupByTime; "
            "for a single metric distribution, use DISTRIBUTION."
        )

    return normalize_analysis_plan(plan)


def to_semantic_plan(plan: S.AnalysisPlan) -> SemanticPlan:
    semantic, _ = to_semantic_plan_with_diagnostics(plan)
    return semantic


def to_semantic_plan_with_diagnostics(plan: S.AnalysisPlan) -> tuple[SemanticPlan, PlanNormalizationSummary]:
    diagnostics = PlanNormalizationSummary()

    input_charts = plan.charts or []
    diagnostics.charts_in = len(input_charts)

    charts: List[SemanticChart] = []
    for chart in input_charts:
        metrics_in_chart = [metric for axis in chart.y_axes for metric in axis.metrics]
        diagnostics.metrics_in += len(metrics_in_chart)

        normalized_y_axes: List[SemanticYAxis] = []
        for axis in chart.y_axes:
            normalized_metrics: List[SemanticMetric] = []
            for metric in axis.metrics:
                code = _normalize_metric_code(metric.metric)
                if not code:
                    diagnostics.dropped_empty_metrics += 1
                    continue

                if code != (metric.metric or ""):
                    diagnostics.normalized_metric_codes += 1

                normalized_metrics.append(
                    SemanticMetric(
                        metric=code,
                        distribution=metric.distribution,
                        data_origin=metric.data_origin,
                        origin_scope=metric.origin_scope,
                    )
                )

            if not normalized_metrics:
                continue

            normalized_y_axes.append(
                SemanticYAxis(
                    metrics=normalized_metrics,
                    statistic=axis.statistic,
                    axis_id=axis.axis_id,
                )
            )

        if not normalized_y_axes:
            diagnostics.dropped_empty_charts += 1
            continue

        x_axis, series_by, axis_norm_count, axis_drop_count = _normalize_chart_axes(chart)
        diagnostics.normalized_canonical_groupby_fields += axis_norm_count
        diagnostics.dropped_invalid_groupby_fields += axis_drop_count
        if x_axis is None:
            diagnostics.dropped_empty_charts += 1
            continue

        raw_group_by = _chart_group_by(
            SemanticChart(
                chart_type=chart.chart_type,
                x_axis=x_axis,
                y_axes=normalized_y_axes,
                series_by=series_by,
                filters=chart.filters,
                title=chart.title,
            )
        )
        normalized_group_by, group_norm_count, group_drop_count = _normalize_group_by(raw_group_by)
        diagnostics.normalized_canonical_groupby_fields += group_norm_count
        diagnostics.dropped_invalid_groupby_fields += group_drop_count

        group_by = _dedupe_group_by(normalized_group_by)
        diagnostics.deduped_groupby_entries += max(0, len(raw_group_by) - len(group_by))

        normalized_chart_type, chart_type_changed, chart_type_fallback = _normalize_chart_type(chart.chart_type)
        normalized_chart_type = _normalize_chart_type_for_semantics(normalized_chart_type, x_axis, series_by)
        if chart_type_changed:
            diagnostics.normalized_chart_types += 1
        if chart_type_fallback:
            diagnostics.fallback_chart_type_count += 1

        normalized_series_by = series_by
        if normalized_series_by is not None:
            series_spec = normalized_series_by.split_by
            for gb in group_by:
                if isinstance(gb, S.GroupByTime):
                    continue
                if type(series_spec) is type(gb):
                    normalized_series_by = S.SeriesSpec(splitBy=gb)
                    break

        normalized_x_axis = x_axis
        if isinstance(x_axis, S.CategoryXAxis):
            for gb in group_by:
                if isinstance(gb, S.GroupByTime):
                    continue
                if normalized_series_by is not None and gb == normalized_series_by.split_by:
                    continue
                normalized_x_axis = S.CategoryXAxis(groupBy=gb, order=x_axis.order)
                break

        charts.append(
            SemanticChart(
                chart_type=normalized_chart_type,
                x_axis=normalized_x_axis,
                y_axes=normalized_y_axes,
                series_by=normalized_series_by,
                filters=chart.filters,
                title=chart.title,
            )
        )

    diagnostics.metrics_out = sum(len(axis.metrics) for c in charts for axis in c.y_axes)
    diagnostics.charts_out = len(charts)

    return SemanticPlan(charts=charts, statistical_tests=plan.statistical_tests), diagnostics


def to_analysis_plan(plan: SemanticPlan) -> S.AnalysisPlan:
    charts: List[S.ChartSpec] = []
    for chart in plan.charts:
        y_axes: List[S.YAxisSpec] = []
        for axis in chart.y_axes:
            metrics: List[S.MetricSpec] = []
            for m in axis.metrics:
                metrics.append(
                    S.MetricSpec(
                        metric=m.metric,
                        distribution=m.distribution,
                        dataOrigin=m.data_origin,
                        originScope=m.origin_scope,
                    )
                )

            if not metrics:
                continue

            y_axes.append(
                S.YAxisSpec(
                    metrics=metrics,
                    statistic=axis.statistic,
                    axisId=axis.axis_id,
                )
            )

        if not y_axes:
            continue

        charts.append(
            S.ChartSpec(
                chart_type=_normalize_chart_type(chart.chart_type)[0],
                xAxis=chart.x_axis,
                yAxes=y_axes,
                seriesBy=chart.series_by,
                filters=chart.filters,
                title=chart.title,
            )
        )

    return S.AnalysisPlan(charts=charts or None, statistical_tests=plan.statistical_tests)


def normalize_analysis_plan(plan: S.AnalysisPlan) -> S.AnalysisPlan:
    normalized, _ = normalize_analysis_plan_with_diagnostics(plan)
    return normalized


def normalize_analysis_plan_with_diagnostics(plan: S.AnalysisPlan) -> tuple[S.AnalysisPlan, PlanNormalizationSummary]:
    semantic, diagnostics = to_semantic_plan_with_diagnostics(plan)
    return to_analysis_plan(semantic), diagnostics
