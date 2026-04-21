from typing import Any, Dict, Optional

from pydantic import BaseModel


class AnalyticsResult(BaseModel):
    """Base analytics result with common metadata."""

    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
