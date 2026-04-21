from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.types import ChartMetadata
from src.domain.dto.charts.waterfall import WaterfallChart, WaterfallStep

from .. import command


@command("test_waterfall")
def test_waterfall(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    steps = [WaterfallStep(label="Start", value=100), WaterfallStep(label="Gain", value=25), WaterfallStep(label="End", value=125, is_total=True)]
    metadata = ChartMetadata(title="Test Waterfall")
    chart = WaterfallChart(metadata=metadata, data=steps)
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
