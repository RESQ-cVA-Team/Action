from __future__ import annotations

from typing import Dict, List

from src.domain.langchain import schema as S
from src.shared.ssot_loader import get_metric_metadata, get_metric_text_lookup


def _metric_unit(metric_code: str) -> str | None:
    code = (metric_code or "").strip().upper()
    meta = get_metric_metadata().get(code, {})

    unit_raw = meta.get("unit")
    if isinstance(unit_raw, str) and unit_raw.strip():
        return unit_raw.strip().lower()

    numeric = meta.get("numeric")
    if isinstance(numeric, dict):
        nested = numeric.get("unit")
        if isinstance(nested, str) and nested.strip():
            return nested.strip().lower()

    text_meta = get_metric_text_lookup().get(code.lower(), {})
    text_unit = text_meta.get("unit")
    if isinstance(text_unit, str) and text_unit.strip():
        return text_unit.strip().lower()

    return None


def _line_axis_metric_units(chart: S.LineChartSpec) -> Dict[str, List[str]]:
    units_by_y: Dict[str, List[str]] = {}
    for series in chart.series:
        unit = _metric_unit(series.metric)
        if unit is None:
            continue
        units = units_by_y.setdefault(series.y_axis, [])
        if unit not in units:
            units.append(unit)
    return units_by_y


def validate_analysis_plan_semantics(plan: S.AnalysisPlan) -> S.AnalysisPlan:
    """Run deterministic semantic checks without mutating plan shape."""

    for idx, chart in enumerate(plan.charts or []):
        chart_number = idx + 1

        if isinstance(chart, S.LineChartSpec):
            for x_key, x_axis in chart.x_axes.items():
                if isinstance(x_axis, S.NumericMetricXAxis):
                    raise ValueError(
                        f"Chart(s) {chart_number}: LINE does not support numeric_metric x-axis ('{x_key}'). "
                        "Use HISTOGRAM for metric distributions."
                    )

            units_by_y = _line_axis_metric_units(chart)
            for y_key, units in units_by_y.items():
                if len(units) > 1:
                    raise ValueError(
                        f"Chart(s) {chart_number}: y-axis '{y_key}' mixes metric units {sorted(units)}. "
                        "Use separate y-axes or separate charts for different units."
                    )

            # A metric_value y-axis should always be referenced by at least one series.
            # This is mostly enforced by schema axis-reference validation, but we keep
            # this explicit check to fail with a domain-specific message.
            used_y = {s.y_axis for s in chart.series}
            for y_key, y_axis in chart.y_axes.items():
                if isinstance(y_axis, S.MetricValueAxis) and y_key not in used_y:
                    raise ValueError(
                        f"Chart(s) {chart_number}: metric_value y-axis '{y_key}' is not referenced by any series."
                    )

        if isinstance(chart, S.HistogramChartSpec):
            if not isinstance(chart.x_axis, S.NumericMetricXAxis):
                raise ValueError(
                    f"Chart(s) {chart_number}: HISTOGRAM requires a numeric_metric x-axis."
                )

    return plan
