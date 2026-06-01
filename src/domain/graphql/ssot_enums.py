"""Dynamic SSOT-based enums for GraphQL models.

Refactored to use central loader (src/shared/ssot_loader.py) for:
 - Cached YAML access
 - Single canonical source of enum creation
 - Future metadata access (e.g., labels, units) via get_metric_metadata()
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING

from src.shared.ssot_loader import create_enum, get_metric_metadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # Type stubs (match runtime signature: str subclass of Enum)

    class SexType(str, Enum): ...

    class StrokeType(str, Enum): ...

    class MetricType(str, Enum): ...

    class GroupByType(str, Enum): ...

    class BooleanPropertyType(str, Enum): ...

    class Operator(str, Enum): ...


# Create dynamic enums from SSOT via unified loader
SexType = create_enum("SexType", "SexType.yml")
StrokeType = create_enum("StrokeType", "StrokeType.yml")
MetricType = create_enum("MetricType", "MetricType.yml")
GroupByType = create_enum("GroupByType", "GroupByType.yml")
BooleanPropertyType = create_enum("BooleanPropertyType", "BooleanType.yml")
Operator = create_enum("Operator", "OperatorType.yml")

# Expose metric metadata (optional consumers can import)
MetricMetadata = get_metric_metadata()

__all__ = [
    "SexType",
    "StrokeType",
    "MetricType",
    "GroupByType",
    "BooleanPropertyType",
    "Operator",
    "MetricMetadata",
]

if __name__ == "__main__":  # Diagnostic output
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Dynamic SSOT Enum System (Unified Loader)")
    logger.info("%s", "=" * 40)
    logger.info("SexType: %s values", len(list(SexType)))  # type: ignore[arg-type]
    logger.info("StrokeType: %s values", len(list(StrokeType)))  # type: ignore[arg-type]
    logger.info("MetricType: %s values (metadata entries: %s)", len(list(MetricType)), len(MetricMetadata))  # type: ignore[arg-type]
    sample_metrics = list(MetricType)[:3]  # type: ignore[arg-type]
    logger.info("Sample metrics: %s", [m.value for m in sample_metrics])
    logger.info("GroupByType: %s values", len(list(GroupByType)))  # type: ignore[arg-type]
    logger.info("BooleanPropertyType: %s values", len(list(BooleanPropertyType)))  # type: ignore[arg-type]
    logger.info("Operator: %s values", len(list(Operator)))  # type: ignore[arg-type]
    # Show one metadata example if available
    for sm in sample_metrics:
        meta = MetricMetadata.get(sm.value)
        if meta:
            logger.info("%s meta keys: %s", sm.value, list(meta.keys()))
    logger.info("Enums & metadata now via shared loader.")
