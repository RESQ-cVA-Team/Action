from __future__ import annotations

from typing import List

from src.domain.dto.execution_summary import PlanNormalizationSummary
from src.domain.dto.semantic_plan import SemanticChart, SemanticMetric, SemanticPlan
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


def _normalize_analysis_mode(analysis_mode: str | None) -> str:
    mode = (analysis_mode or "").strip().upper()
    if mode in {"TIME_SERIES", "DISTRIBUTION", "SUMMARY", "COMPARISON"}:
        return mode
    return ""


def _normalize_chart_type_for_semantics(chart_type: str, analysis_mode: str, group_by: List[S.GroupBySpec]) -> str:
    if analysis_mode == "DISTRIBUTION" and group_by:
        return "BAR"
    return chart_type


def _has_time_groupby(group_by: List[S.GroupBySpec]) -> bool:
    return any(type(g).__name__ == "GroupByTime" for g in group_by)


def _has_non_time_groupby(group_by: List[S.GroupBySpec]) -> bool:
    return any(type(g).__name__ != "GroupByTime" for g in group_by)


def _default_time_groupby() -> S.GroupByTime:
    return S.GroupByTime(grain="MONTH", window=S.TimeWindow(last_n=24, unit="MONTH"))


def validate_analysis_plan_semantics(plan: S.AnalysisPlan) -> S.AnalysisPlan:
    """Resolve chart semantics from structure and fail when intent is still ambiguous.

    This is the planner boundary: if a chart cannot be assigned a concrete analysis mode
    after structural inference, the planner should retry or clarify before execution.
    """

    # DISTRIBUTION + GroupByTime is not supported by the executor: it fetches one flat
    # distribution and silently drops the time grain.  Fail fast here so the planner retries
    # with corrective feedback instead of producing a silently wrong query.
    dist_time_indices: List[int] = []
    for idx, chart in enumerate(plan.charts or []):
        mode = _normalize_analysis_mode(getattr(chart, "analysis_mode", None))
        if mode == "DISTRIBUTION" and _has_time_groupby(chart.group_by or []):
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
        mode = _normalize_analysis_mode(getattr(chart, "analysis_mode", None))
        if mode != "COMPARISON":
            continue
        if len(chart.metrics or []) != 1:
            continue
        group_by = chart.group_by or []
        if not group_by or not _has_non_time_groupby(group_by):
            comparison_only_time_indices.append(idx + 1)
    if comparison_only_time_indices:
        joined = ", ".join(str(i) for i in comparison_only_time_indices)
        raise ValueError(
            f"Chart(s) {joined}: COMPARISON analysis mode needs a real comparison dimension such as sex, provider, "
            "or another categorical group. For a single metric over time, use TIME_SERIES with GroupByTime; "
            "for a single metric distribution, use DISTRIBUTION."
        )

    semantic, _ = to_semantic_plan_with_diagnostics(plan)
    for chart in semantic.charts:
        mode = _normalize_analysis_mode(chart.analysis_mode)
        if mode != "TIME_SERIES":
            continue
        group_by = list(chart.group_by or [])
        if not _has_time_groupby(group_by):
            group_by.append(_default_time_groupby())
            chart.group_by = group_by
    unresolved_indices = [idx + 1 for idx, chart in enumerate(semantic.charts) if not _normalize_analysis_mode(chart.analysis_mode)]
    if unresolved_indices:
        joined = ", ".join(str(idx) for idx in unresolved_indices)
        raise ValueError(
            "Unable to infer chart analysis intent for chart(s) "
            f"{joined}. Specify whether the request is a time series, distribution, summary, or comparison."
        )

    return to_analysis_plan(semantic)


def to_semantic_plan(plan: S.AnalysisPlan) -> SemanticPlan:
    semantic, _ = to_semantic_plan_with_diagnostics(plan)
    return semantic


def to_semantic_plan_with_diagnostics(plan: S.AnalysisPlan) -> tuple[SemanticPlan, PlanNormalizationSummary]:
    diagnostics = PlanNormalizationSummary()

    input_charts = plan.charts or []
    diagnostics.charts_in = len(input_charts)

    charts: List[SemanticChart] = []
    for chart in input_charts:
        metrics_in_chart = chart.metrics or []
        diagnostics.metrics_in += len(metrics_in_chart)

        normalized_metrics: List[SemanticMetric] = []
        for metric in metrics_in_chart:
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
            diagnostics.dropped_empty_charts += 1
            continue

        raw_group_by = chart.group_by or []
        normalized_group_by, normalized_fields_count, dropped_invalid_fields = _normalize_group_by(raw_group_by)
        diagnostics.normalized_canonical_groupby_fields += normalized_fields_count
        diagnostics.dropped_invalid_groupby_fields += dropped_invalid_fields

        group_by = _dedupe_group_by(normalized_group_by)
        diagnostics.deduped_groupby_entries += max(0, len(raw_group_by) - len(group_by))

        normalized_chart_type, chart_type_changed, chart_type_fallback = _normalize_chart_type(chart.chart_type)
        normalized_analysis_mode = _normalize_analysis_mode(getattr(chart, "analysis_mode", None))
        normalized_chart_type = _normalize_chart_type_for_semantics(normalized_chart_type, normalized_analysis_mode, group_by)
        if chart_type_changed:
            diagnostics.normalized_chart_types += 1
        if chart_type_fallback:
            diagnostics.fallback_chart_type_count += 1

        charts.append(
            SemanticChart(
                chart_type=normalized_chart_type,
                analysis_mode=normalized_analysis_mode,
                metrics=normalized_metrics,
                filters=chart.filters,
                group_by=group_by,
            )
        )

    diagnostics.metrics_out = sum(len(c.metrics) for c in charts)
    diagnostics.charts_out = len(charts)

    return SemanticPlan(charts=charts, statistical_tests=plan.statistical_tests), diagnostics


def to_analysis_plan(plan: SemanticPlan) -> S.AnalysisPlan:
    charts: List[S.ChartSpec] = []
    for chart in plan.charts:
        metrics: List[S.MetricSpec] = []
        for m in chart.metrics:
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

        group_by = [g.model_dump(by_alias=True, exclude_none=True) if hasattr(g, "model_dump") else g for g in (chart.group_by or [])]

        charts.append(
            S.ChartSpec(
                chart_type=_normalize_chart_type(chart.chart_type)[0],
                analysisMode=_normalize_analysis_mode(chart.analysis_mode),
                filters=chart.filters,
                group_by=group_by or None,
                metrics=metrics,
            )
        )

    return S.AnalysisPlan(charts=charts or None, statistical_tests=plan.statistical_tests)


def normalize_analysis_plan(plan: S.AnalysisPlan) -> S.AnalysisPlan:
    normalized, _ = normalize_analysis_plan_with_diagnostics(plan)
    return normalized


def normalize_analysis_plan_with_diagnostics(plan: S.AnalysisPlan) -> tuple[S.AnalysisPlan, PlanNormalizationSummary]:
    semantic, diagnostics = to_semantic_plan_with_diagnostics(plan)
    return to_analysis_plan(semantic), diagnostics
