from typing import Optional

from pydantic import BaseModel

from src.domain.dto.charts.types import ChartMetric


class ValueResult(BaseModel):
    """One number answering a question, with what it was computed over."""

    metric: ChartMetric
    # MEDIAN, MEAN, COUNT or RATE (a share of cases, in percent).
    statistic: str
    # None when the period has no cases.
    value: Optional[float] = None
    # The metric's cohort in the period: the cases the number is computed over.
    patients: Optional[int] = None
    start_date: str
    end_date: str
    scope: Optional[str] = None
    filters: Optional[str] = None
