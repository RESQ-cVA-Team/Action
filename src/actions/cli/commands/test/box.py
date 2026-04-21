from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.box import BoxEntry, BoxPlot
from src.domain.dto.charts.types import ChartMetadata

from .. import command


@command("test_box")
def test_box(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    entries = [BoxEntry(name="A", q1=1, median=2, q3=3, min=0, max=4, outliers=[-1, 5])]
    metadata = ChartMetadata(title="Test Box")
    chart = BoxPlot(metadata=metadata, data=entries)
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
