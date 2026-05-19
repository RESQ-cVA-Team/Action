from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.analytics import StatisticalTestResult
from src.domain.dto.charts.line import LineChart
from src.domain.dto.charts.types import ChartAxis, ChartMetadata, ChartPoint, ChartSeries
from src.domain.dto.response import VisualizationResponse

from .. import command, register


@command("test_analytics")
def test_analytics(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    """Send a demo response with charts and typed statistical results."""
    # Simple mini chart
    points = [ChartPoint(x=i, y=float(i * i)) for i in range(5)]
    series = ChartSeries(name="Quadratic", data=points, color="#ef4444")
    chart = LineChart(
        metadata=ChartMetadata(
            title="Quadratic Growth",
            x_axis=ChartAxis(label="x", type=ChartAxis.AxisType.LINEAR),
            y_axis=ChartAxis(label="y", type=ChartAxis.AxisType.LINEAR),
        ),
        series=[series],
    )

    # Sample typed stats
    stats: List[StatisticalTestResult] = [
        StatisticalTestResult(
            test_type="MANN_WHITNEY_U_TEST",
            status="success",
            p_value=0.031,
            passed=True,
            title="Mann-Whitney U Test: DTN",
            details={
                "metric": "DOOR_TO_NEEDLE",
                "u_statistic": 412.0,
                "cohort_a_label": "My Hospital",
                "cohort_b_label": "National",
                "cohort_a_size": 85,
                "cohort_b_size": 4320,
                "cohort_a_median": 42.0,
                "cohort_b_median": 55.0,
            },
        )
    ]

    payload = VisualizationResponse(
        charts=[chart],
        stats=stats,
        timestamp=datetime.now(timezone.utc),
    ).model_dump(exclude_none=True)

    dispatcher.utter_message(json_message=payload)
    return []


# Provide a shorter alias as well
register("analytics", test_analytics)
