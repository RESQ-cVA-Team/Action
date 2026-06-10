from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.langchain import schema as S


def _metric_list() -> List["SemanticMetric"]:
    return []


def _group_by_list() -> List[S.GroupBySpec]:
    return []


def _y_axis_list() -> List["SemanticYAxis"]:
    return []


def _chart_list() -> List["SemanticChart"]:
    return []


@dataclass
class SemanticMetric:
    metric: str
    distribution: Optional[S.DistributionSpec] = None
    data_origin: Optional[S.DataOriginSpec] = None
    origin_scope: Optional[S.OriginScopeSpec] = None


@dataclass
class SemanticYAxis:
    metrics: List[SemanticMetric] = field(default_factory=_metric_list)
    statistic: str = "MEAN"
    axis_id: Optional[str] = None


@dataclass
class SemanticChart:
    chart_type: str
    x_axis: S.XAxisSpec
    y_axes: List[SemanticYAxis] = field(default_factory=_y_axis_list)
    series_by: Optional[S.SeriesSpec] = None
    filters: Optional[S.FilterNode] = None
    title: Optional[str] = None


@dataclass
class SemanticPlan:
    charts: List[SemanticChart] = field(default_factory=_chart_list)
    statistical_tests: Optional[List[S.StatisticalTestSpec]] = None
