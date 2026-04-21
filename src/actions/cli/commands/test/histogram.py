from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType  # type: ignore

from src.domain.dto.charts.histogram import Histogram, HistogramBin
from src.domain.dto.charts.types import ChartMetadata

from .. import command


@command("test_histogram")
def test_histogram(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    bins = [HistogramBin(range_start=0, range_end=1, frequency=5), HistogramBin(range_start=1, range_end=2, frequency=8)]
    metadata = ChartMetadata(title="Test Histogram")
    chart = Histogram(metadata=metadata, data=bins, bin_count=2)
    dispatcher.utter_message(json_message={"charts": [chart.model_dump(exclude_none=True)]})
    return []
