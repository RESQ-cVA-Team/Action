from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.scatter import ScatterPlot
from src.domain.dto.charts.types import ChartMetadata, ChartPoint, ChartSeries

from .. import command


@command("test_scatter")
def test_scatter(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    pts = [ChartPoint(x=1, y=2), ChartPoint(x=2, y=3)]
    series = ChartSeries(name="pts", data=pts)
    metadata = ChartMetadata(title="Test Scatter")
    chart = ScatterPlot(metadata=metadata, series=[series])
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
