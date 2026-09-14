import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome
from src.planners.langchain.request_orchestrator import orchestrate_visualization_request


def load_scenarios(file_name: str) -> list[dict[str, object]]:
    scenario_path = Path(__file__).resolve().parent / "fixtures" / "scenarios" / file_name
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def run_orchestrator(prompt: str, entities: dict[str, object], plan: AnalysisPlan | None = None) -> VisualizationRequestOutcome:
    decision_patch = patch(
        "src.planners.langchain.request_orchestrator._decision_stage",
        return_value=VisualizationRequestOutcome(decision="proceed", reason="all_required_fields_present"),
    )
    if plan is None:
        with decision_patch:
            return orchestrate_visualization_request(question=prompt, entities=entities, include_plan=True)

    with decision_patch, patch(
        "src.planners.langchain.request_orchestrator.generate_analysis_plan",
        return_value=plan,
    ):
        return orchestrate_visualization_request(question=prompt, entities=entities, include_plan=True)