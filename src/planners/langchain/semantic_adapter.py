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
        if chart_type_changed:
            diagnostics.normalized_chart_types += 1
        if chart_type_fallback:
            diagnostics.fallback_chart_type_count += 1

        charts.append(
            SemanticChart(
                chart_type=normalized_chart_type,
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

        charts.append(
            S.ChartSpec(
                chart_type=_normalize_chart_type(chart.chart_type)[0],
                filters=chart.filters,
                group_by=chart.group_by or None,
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
