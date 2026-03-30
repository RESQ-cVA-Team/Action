from __future__ import annotations

import logging
from typing import List, Optional

from src.domain.dto.charts import BarChart, ChartDTO, LineChart, union
from src.domain.dto.charts.types import ChartAxis, ChartMetadata, ChartSeries, ChartType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import GroupByAge, GroupByCanonicalField, GroupByNIHSS, GroupBySex, GroupByStrokeType
from src.executors.planning.query_compiler import Dimension
from src.shared.ssot_loader import get_canonical_display_name, get_metric_display_name

logger = logging.getLogger(__name__)


def _dimension_label(dimension: Dimension) -> Optional[str]:
    if isinstance(dimension.spec, GroupBySex):
        return "Sex"
    if isinstance(dimension.spec, GroupByStrokeType):
        return "Stroke Type"
    if isinstance(dimension.spec, GroupByNIHSS):
        return get_canonical_display_name("ADMISSION_NIHSS")
    if isinstance(dimension.spec, GroupByAge):
        return get_canonical_display_name("AGE")
    if isinstance(dimension.spec, GroupByCanonicalField):
        return get_canonical_display_name(dimension.spec.field)
    return None


def _derive_title(plan_chart: S.ChartSpec, dimensions: List[Dimension]) -> str:
    metric_names = [get_metric_display_name(m.metric) for m in plan_chart.metrics]

    dim_names: List[str] = []
    for dimension in dimensions:
        label = _dimension_label(dimension)
        if label:
            dim_names.append(label)

    dims_phrase = f" by {' and '.join(dim_names)}" if dim_names else ""

    if plan_chart.title:
        base = plan_chart.title
        return base if (" by " in base.lower()) or not dims_phrase else base + dims_phrase

    if len(metric_names) == 0:
        base_title = f"{plan_chart.chart_type.title()} Chart"
    elif len(metric_names) == 1:
        base_title = f"{metric_names[0]} Distribution"
    elif len(metric_names) == 2:
        base_title = f"{metric_names[0]} and {metric_names[1]}"
    else:
        base_title = ", ".join(metric_names[:-1]) + f" and {metric_names[-1]}"

    return base_title + dims_phrase


def build_chart_dto(
    plan_chart: S.ChartSpec,
    dimensions: List[Dimension],
    series: List[ChartSeries],
    derived_axes: Optional[tuple[ChartAxis, ChartAxis]],
) -> ChartDTO:
    title_text = _derive_title(plan_chart, dimensions)

    x_axis: Optional[ChartAxis] = None
    y_axis: Optional[ChartAxis] = None
    if derived_axes is not None:
        x_axis, y_axis = derived_axes

    chart_type_upper = (plan_chart.chart_type or "").upper()
    metadata = ChartMetadata(
        title=title_text,
        description=plan_chart.description,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    if chart_type_upper == ChartType.LINE.value:
        return LineChart(metadata=metadata, series=series)
    if chart_type_upper == ChartType.BAR.value:
        return BarChart(metadata=metadata, series=series)
    if chart_type_upper == ChartType.AREA.value:
        return union.AreaChart(metadata=metadata, series=series)

    logger.warning("Chart type %s not yet implemented; defaulting to LINE rendering", plan_chart.chart_type)
    return LineChart(metadata=metadata, series=series)
