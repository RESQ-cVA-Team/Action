from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel

from .analytics import StatisticalTestResult
from .charts.union import ChartDTO


class VisualizationResponse(BaseModel):
    """Response containing charts and statistical results (v1)."""

    type: Literal["visualization_response"] = "visualization_response"
    schema_version: int = 1
    trace_id: Optional[str] = None
    charts: List[ChartDTO] = []
    stats: List[StatisticalTestResult] = []
    timestamp: Optional[datetime] = None
