import json
import re
from pathlib import Path

from src.planners.langchain.examples import get_few_shot_examples
from src.planners.langchain.metric_data_types import metric_data_types
from src.shared.ssot_loader import get_metric_data_type

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "src" / "planners" / "langchain" / "fewshot_examples"
_PERCENT_WORDING = re.compile(r"\b(percent|percentage|share|proportion|rate)\b", re.IGNORECASE)


def test_entity_metrics_get_their_data_type() -> None:
    assert metric_data_types("show me door to needle time", {"metric": ["DTN"]}) == {"DTN": "Numeric"}
    assert metric_data_types("show me arrival mode", {"metric": "arrival_mode"}) == {"ARRIVAL_MODE": "Enum"}


def test_codes_in_the_question_get_their_data_type() -> None:
    previous = 'Previous chart plan:\n{"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN"}]}]}\n\nUser\'s newest instruction:\nadd ARRIVAL_MODE'
    assert metric_data_types(previous, {}) == {"ARRIVAL_MODE": "Enum", "DTN": "Numeric"}


def test_codes_that_are_not_metrics_are_left_out() -> None:
    assert metric_data_types("a LINE chart of RATE per QUARTER", {"metric": ["NOT_A_METRIC"]}) == {}
    assert get_metric_data_type("NOT_A_METRIC") is None


def test_few_shot_examples_show_the_data_types() -> None:
    sex = next(ex for ex in get_few_shot_examples() if ex["name"] == "sex_distribution_bar")
    assert sex["user"].endswith('METRIC_DATA_TYPES(JSON):\n{"SEX": "Enum"}')


def test_few_shot_measures_follow_the_data_type() -> None:
    for path in sorted(_EXAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        asks_percent = bool(_PERCENT_WORDING.search(data["user_utterance"]))
        for chart in data["assistant"].get("charts") or []:
            measure = ((chart.get("semantics") or {}).get("measure") or {}).get("type")
            for metric in chart.get("metrics") or []:
                data_type = get_metric_data_type(metric["metric"])
                if data_type == "Enum":
                    assert measure in ("RATE", "COUNT"), f"{path.name}: Enum {metric['metric']} measured as {measure}"
                if data_type == "Numeric" and not asks_percent:
                    assert measure != "RATE", f"{path.name}: Numeric {metric['metric']} measured as RATE without percent wording"
