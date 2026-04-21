from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.radar import RadarChart
from src.domain.dto.charts.types import ChartMetadata, ChartPoint, ChartSeries

from .. import command


@command("test_radar")
def test_radar(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    axes = ["Speed", "Agility", "Strength", "Endurance", "IQ"]
    series_a = ChartSeries(
        name="Player",
        data=[
            ChartPoint(x="Speed", y=8),
            ChartPoint(x="Agility", y=6),
            ChartPoint(x="Strength", y=7),
            ChartPoint(x="Endurance", y=5),
            ChartPoint(x="IQ", y=9),
        ],
    )
    series_b = ChartSeries(
        name="Peer Avg",
        data=[
            ChartPoint(x="Speed", y=6),
            ChartPoint(x="Agility", y=5),
            ChartPoint(x="Strength", y=6),
            ChartPoint(x="Endurance", y=6),
            ChartPoint(x="IQ", y=7),
        ],
    )
    metadata = ChartMetadata(title="Test Radar")
    chart = RadarChart(metadata=metadata, series=[series_a, series_b], axes=axes)
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
