from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.bar import BarChart
from src.domain.dto.charts.types import ChartMetadata, ChartPoint, ChartSeries

from .. import command


@command("test_bar")
def test_bar(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    points = [ChartPoint(x=i, y=float(i)) for i in range(3)]
    series = ChartSeries(name="Bars", data=points)
    metadata = ChartMetadata(title="Test Bar")
    chart = BarChart(metadata=metadata, series=[series])
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
