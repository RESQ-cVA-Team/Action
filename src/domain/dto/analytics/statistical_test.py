from typing import Any, Dict, Optional

from pydantic import BaseModel


class StatisticalTestResult(BaseModel):
    """Generic statistical test result.

    This structure is intentionally generic so it can be used across multiple tests.
    Specific executors can extend or add fields via the 'details' mapping.
    """

    test_type: str
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    significance_level: Optional[float] = None
    passed: Optional[bool] = None
    details: Optional[Dict[str, Any]] = None  # Arbitrary per-test payload (e.g., coefficients, tables)
    title: Optional[str] = None
