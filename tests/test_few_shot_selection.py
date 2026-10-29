import json
from pathlib import Path

from src.planners.langchain.examples import get_few_shot_examples
from src.planners.langchain.pipeline import _select_few_shot_examples

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "src" / "planners" / "langchain" / "fewshot_examples"


def test_every_example_carries_its_entities() -> None:
    for example in get_few_shot_examples():
        data = json.loads((_EXAMPLES_DIR / f"{example['name']}.json").read_text(encoding="utf-8"))
        assert example["entities"] == data["entities_detected"], example["name"]


def test_selection_matches_examples_on_their_entities() -> None:
    selected = [ex["name"] for ex in _select_few_shot_examples("Number of patients per ARRIVAL_MODE", {"metric": ["ARRIVAL_MODE"]}, 6)]
    assert {"arrival_mode_percent_bar", "default_bare_metric_enum"} <= set(selected)
