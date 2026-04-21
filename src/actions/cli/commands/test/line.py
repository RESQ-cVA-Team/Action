from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.line import LineChart
from src.domain.dto.charts.types import ChartAxis, ChartMetadata, ChartPoint, ChartSeries

from .. import command


@command("test_line")
def test_line(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    today = date.today()
    points = [ChartPoint(x=(today - timedelta(days=2 - i)).isoformat(), y=float(i)) for i in range(3)]
    series = ChartSeries(name="L", data=points)
    metadata = ChartMetadata(title="Test Line", x_axis=ChartAxis(label="Date", type=ChartAxis.AxisType.TIME))
    chart = LineChart(metadata=metadata, series=[series])
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
