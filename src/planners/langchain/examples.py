import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.metric_data_types import metric_data_types_json

_EXAMPLES_DIR = Path(__file__).resolve().parent / "fewshot_examples"


class FewShotExample(TypedDict):
    name: str
    user: str
    assistant: str
    entities: Dict[str, Any]


def get_few_shot_examples() -> List[FewShotExample]:
    """Loads few-shot (user, assistant) pairs from fewshot_examples/*.json.

    Each file holds {"user_utterance": <text>, "entities_detected": <object>,
    "assistant": <AnalysisPlan as a JSON object>}. The prompt-facing "user"
    string is reconstructed from the first two fields rather than stored
    pre-formatted, so an editor can work with plain text + a JSON object
    instead of one opaque blob. The plan is validated against the current
    schema on every load (not just at extraction time) so a schema change
    that invalidates an example fails loudly at Action startup instead of
    silently shipping a stale example to the LLM.
    """
    examples: List[FewShotExample] = []
    for path in sorted(_EXAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        plan = AnalysisPlan.model_validate(data["assistant"])
        user = (
            f"USER_UTTERANCE:\n{data['user_utterance']}\n\n"
            f"ENTITIES_DETECTED(JSON):\n{json.dumps(data['entities_detected'])}\n\n"
            f"METRIC_DATA_TYPES(JSON):\n{metric_data_types_json(data['user_utterance'], data['entities_detected'])}"
        )
        examples.append(
            {
                "name": path.stem,
                "user": user,
                "assistant": plan.model_dump_json(indent=2),
                "entities": data["entities_detected"],
            }
        )
    return examples
