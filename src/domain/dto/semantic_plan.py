from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.langchain import schema as S


def _metric_list() -> List["SemanticMetric"]:
    return []


def _group_by_list() -> List[S.GroupBySpec]:
    return []


def _chart_list() -> List["SemanticChart"]:
    return []


@dataclass
class SemanticMetric:
    metric: str
    data_origin: Optional[S.DataOriginSpec] = None
    origin_scope: Optional[S.OriginScopeSpec] = None


@dataclass
class SemanticChart:
    chart_type: str
    metrics: List[SemanticMetric] = field(default_factory=_metric_list)
    filters: Optional[S.FilterNode] = None
    group_by: List[S.GroupBySpec] = field(default_factory=_group_by_list)
    numeric_resolution: Optional[S.NumericResolutionSpec] = None


@dataclass
class SemanticPlan:
    charts: List[SemanticChart] = field(default_factory=_chart_list)
    statistical_tests: Optional[List[S.StatisticalTestSpec]] = None
