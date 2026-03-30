from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.langchain import schema as S


@dataclass
class SemanticMetric:
    metric: str
    title: Optional[str] = None
    description: Optional[str] = None
    distribution: Optional[S.DistributionSpec] = None


@dataclass
class SemanticChart:
    chart_type: str
    metrics: List[SemanticMetric] = field(default_factory=list)
    title: Optional[str] = None
    description: Optional[str] = None
    filters: Optional[S.FilterNode] = None
    group_by: List[S.GroupBySpec] = field(default_factory=list)


@dataclass
class SemanticPlan:
    charts: List[SemanticChart] = field(default_factory=list)
    statistical_tests: Optional[List[S.StatisticalTestSpec]] = None
    metadata: S.PlanMetadata = field(default_factory=S.PlanMetadata)
