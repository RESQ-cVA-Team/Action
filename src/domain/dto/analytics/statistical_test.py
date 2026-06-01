from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class StatisticalTestResult(BaseModel):
    """Generic statistical test result.

    Covers any statistical test executed by the analytics service.
    MW-specific fields (cohort sizes, medians, u-statistic) live in 'details'.
    """

    test_type: str
    status: Literal["success", "skipped", "error"] = "success"
    reason: Optional[str] = None  # Human-readable explanation for skipped/error status
    p_value: Optional[float] = None
    passed: Optional[bool] = None
    title: Optional[str] = None
    details: Optional[Dict[str, Any]] = None  # Per-test structured payload
