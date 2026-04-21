from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.area import AreaChart
from src.domain.dto.charts.types import ChartMetadata, ChartPoint, ChartSeries

from .. import command


@command("test_area")
def test_area(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    pts = [ChartPoint(x=i, y=float(i)) for i in range(4)]
    series = ChartSeries(name="area", data=pts)
    metadata = ChartMetadata(title="Test Area")
    chart = AreaChart(metadata=metadata, series=[series])
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
