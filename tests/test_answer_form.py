from typing import Any, List
from unittest.mock import patch

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain import pipeline
from src.planners.langchain.examples import get_few_shot_examples
from src.planners.langchain.pipeline import _select_few_shot_examples
from src.planners.langchain.request_orchestrator import _decision_stage

_CHART = {"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}}}]}
_VALUE = {"values": [{"metric": "DTN", "measure": {"type": "MEDIAN"}}]}


def _planned(replies: List[dict], answer: Any) -> tuple[AnalysisPlan, List[str]]:
    sent = iter(replies)
    feedback: List[str] = []

    def invoke(_chain: Any, inputs: Any, label: str = "") -> str:
        assert inputs["answer"] == answer
        feedback.append(inputs["reasoning"])
        return AnalysisPlan.model_validate(next(sent)).model_dump_json()

    with patch.object(pipeline, "_invoke_with_timeout", side_effect=invoke):
        plan = pipeline.generate_analysis_plan("What is my median DTN?", {"metric": "DTN", "sex": "MALE"}, "en", answer=answer)
    return plan, feedback


def test_values_come_only_when_the_answer_is_a_number() -> None:
    plan, feedback = _planned([_CHART, _VALUE], "number")
    assert plan.answer == "number" and "give values, not charts" in feedback[1]
    plan, feedback = _planned([_VALUE, _CHART], "chart")
    assert plan.answer == "chart" and "give no values" in feedback[1]
    # A test that is not supported falls back to charts.
    plan, feedback = _planned([_CHART], "test")
    assert plan.answer == "chart" and feedback == [""]


def test_the_decision_names_the_answer_form_whatever_it_decides() -> None:
    for decision, answer in [("proceed", "number"), ("clarify", "chart"), ("proceed", "test"), ("proceed", "graph")]:
        reply = {"decision": decision, "reason": "all_required_fields_present" if decision == "proceed" else "ambiguous_request", "answer_form": answer}
        with patch("src.planners.langchain.request_orchestrator._invoke_chain", return_value=reply):
            outcome = _decision_stage(question="What is my median DTN?", entities={"metric": "DTN"}, language="en")
        assert outcome.answer == (answer if answer != "graph" else None)


def test_examples_show_the_answer_form_asked() -> None:
    for example in get_few_shot_examples():
        assert f"ANSWER:\n{example['answer']}\n" in example["user"], example["name"]
    question, entities = "How many patients arrived by ambulance?", {"metric": "ARRIVAL_MODE"}
    charts = _select_few_shot_examples(question, entities, pipeline._MAX_FEW_SHOTS)
    numbers = _select_few_shot_examples(question, entities, pipeline._MAX_FEW_SHOTS, answer="number")
    assert {ex["answer"] for ex in charts} == {"chart"}
    assert numbers[:-1] == charts and numbers[-1]["answer"] == "number"
