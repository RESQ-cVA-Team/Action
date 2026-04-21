from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.pie import PieChart, PieSlice
from src.domain.dto.charts.types import ChartMetadata

from .. import command


@command("test_pie")
def test_pie(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    slices = [PieSlice(label="A", value=40), PieSlice(label="B", value=60)]
    metadata = ChartMetadata(title="Test Pie")
    chart = PieChart(metadata=metadata, data=slices)
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
