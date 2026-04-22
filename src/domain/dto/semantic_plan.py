from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.langchain import schema as S


@dataclass
class SemanticMetric:
    metric: str
    distribution: Optional[S.DistributionSpec] = None
    data_origin: Optional[S.DataOriginSpec] = None
    origin_scope: Optional[S.OriginScopeSpec] = None


@dataclass
class SemanticChart:
    chart_type: str
    metrics: List[SemanticMetric] = field(default_factory=list)
    filters: Optional[S.FilterNode] = None
    group_by: List[S.GroupBySpec] = field(default_factory=list)


@dataclass
class SemanticPlan:
    charts: List[SemanticChart] = field(default_factory=list)
    statistical_tests: Optional[List[S.StatisticalTestSpec]] = None
