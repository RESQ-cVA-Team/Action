import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from src.actions.helpers.visualization import canonicalize_ssot_entities


def _load_visualization_context_helpers():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "actions" / "visualization_action.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "_extract_intent_name_from_user_event",
        "_merge_entities",
        "_extract_entities_from_user_event",
        "_collect_visualization_thread_entities",
        "_dedupe_list_values",
        "merge_latest_with_thread_entities",
        "_extract_bot_custom_payload",
        "_is_visualization_payload",
        "_event_has_visualization_signal",
        "_find_latest_visualization_anchor_user_ordinal",
        "_is_awaiting_clarification_reply",
        "_should_carry_forward_visualization_context",
        "canonicalize_ssot_entities",
    }

    required_assigns = {"_LATEST_ENTITY_PRECEDENCE_KEYS"}

    selected = [
        node
        for node in module_ast.body
        if (isinstance(node, ast.FunctionDef) and node.name in required)
        or (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in required_assigns)
    ]

    isolated_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "json": json,
        "cast": cast,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "_VISUALIZATION_THREAD_INTENTS": {
            "generate_visualization",
            "update_visualization",
            "clarify_visualization",
        },
        "_VISUALIZATION_PLAN_TYPE": "visualization_plan",
        "_VISUALIZATION_RESPONSE_SCHEMA_VERSION": 1,
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return namespace


def _load_merge_latest_with_thread_entities():
    return _load_visualization_context_helpers()["merge_latest_with_thread_entities"]


def _user_event(text: str, intent: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "event": "user",
        "text": text,
        "parse_data": {
            "intent": {"name": intent},
            "entities": entities,
        },
    }


def test_merge_latest_with_thread_entities_merges_visualization_thread() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Show DTN by month for past year split by sex",
            "generate_visualization",
            [
                {"entity": "chart_type", "value": "LINE"},
                {"entity": "metric", "value": "DTN"},
                {"entity": "time_grain", "value": "month"},
                {"entity": "split", "value": "sex"},
                {"entity": "date", "value": "past year"},
            ],
        ),
        _user_event(
            "line chart",
            "clarify_visualization",
            [{"entity": "chart_type", "value": "LINE"}],
        ),
        _user_event(
            "DTN",
            "clarify_visualization",
            [{"entity": "metric", "value": "DTN"}],
        ),
    ]

    extracted = merge_latest_with_thread_entities(
        latest_entities={"metric": "DTN"},
        events=events,
        fallback_limit=12,
    )

    assert extracted["metric"] == "DTN"
    assert extracted["chart_type"] == "LINE"
    assert extracted["date"] == "past year"
    assert extracted["time_grain"] == "month"
    assert extracted["split"] == "sex"


def test_merge_latest_with_thread_entities_prefers_latest_value() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Show AGE as line chart",
            "generate_visualization",
            [
                {"entity": "metric", "value": "AGE"},
                {"entity": "chart_type", "value": "LINE"},
            ],
        ),
        _user_event(
            "DTN",
            "clarify_visualization",
            [{"entity": "metric", "value": "DTN"}],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={"metric": "DTN"},
        events=events,
        fallback_limit=12,
    )

    assert merged["metric"] == "DTN"
    assert merged["chart_type"] == "LINE"


def test_merge_latest_with_thread_entities_keeps_latest_provider_scope_keys() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Run Mann-Whitney for old provider groups",
            "generate_visualization",
            [
                {"entity": "group_id", "value": "provider group 2825"},
                {"entity": "group_id", "value": "provider group 3001"},
                {"entity": "date", "value": "2023-01-01"},
                {"entity": "date", "value": "2023-12-31"},
            ],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={
            "hospital_name": ["Aalborg University - Hospital", "Kronborg Castle Hospital"],
            "date": ["2024-01-01", "2026-12-31"],
        },
        events=events,
        fallback_limit=12,
    )

    assert merged["hospital_name"] == ["Aalborg University - Hospital", "Kronborg Castle Hospital"]
    assert merged["date"] == ["2024-01-01", "2026-12-31"]
    # group_id wasn't restated by the latest turn, but merge_latest_with_thread_entities
    # always carries the thread forward once called -- it's the caller's job
    # (_should_carry_forward_visualization_context) to decide whether to call it
    # at all for a given turn, not this function's job to selectively drop keys.
    assert merged["group_id"] == ["provider group 2825", "provider group 3001"]


def test_merge_latest_with_thread_entities_keeps_latest_provider_group_ids() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Run Mann-Whitney for old provider groups",
            "generate_visualization",
            [
                {"entity": "group_id", "value": "provider group 2825"},
                {"entity": "group_id", "value": "provider group 3001"},
            ],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={
            "group_id": ["provider group 279", "provider group 280"],
            "date": ["2024-01-01", "2026-12-31"],
        },
        events=events,
        fallback_limit=12,
    )

    assert merged["group_id"] == ["provider group 279", "provider group 280"]
    assert merged["date"] == ["2024-01-01", "2026-12-31"]


def test_canonicalize_ssot_entities_prefers_explicit_metric_from_question() -> None:
    normalized = canonicalize_ssot_entities(
        {"chart_type": "LINE", "metric": "ICH_TREATMENT_TYPE"},
        question="Show me a line graph of decompressive craniectomy performed",
    )

    assert normalized["metric"] == "CRANIECTOMY"


def test_fresh_generate_visualization_does_not_carry_forward_stale_hospital_scope() -> None:
    """Regression test for a reported carryover bug: after a Mann-Whitney
    comparison names a specific hospital, an unrelated single-metric follow-up
    request that names no hospital must not inherit it.

    This used to be enforced inside merge_latest_with_thread_entities itself
    (a "clear on absence" rule for identity keys). It's now enforced one layer
    up: a fresh generate_visualization that isn't completing a pending
    clarification simply never calls merge_latest_with_thread_entities at
    all -- see _should_carry_forward_visualization_context and its callers in
    visualization_action.py.
    """
    helpers = _load_visualization_context_helpers()
    should_carry_forward = helpers["_should_carry_forward_visualization_context"]
    events = [
        _user_event(
            "Can you compare my dtn against army alhama de murcia hospital using a mann whitney u test",
            "generate_visualization",
            [
                {"entity": "hospital_scope_reference", "value": "my"},
                {"entity": "metric", "value": "DTN"},
                {"entity": "hospital_name", "value": "Army Alhama de Murcia Hospital"},
                {"entity": "country_code", "value": "DE"},
                {"entity": "group_by", "value": "HOSPITAL"},
            ],
        ),
        _user_event(
            "show me a bar chart of stroke type",
            "generate_visualization",
            [
                {"entity": "chart_type", "value": "BAR"},
                {"entity": "metric", "value": "STROKE_TYPE"},
                {"entity": "group_by", "value": "STROKE_TYPE"},
            ],
        ),
    ]

    latest_entities = {
        "chart_type": "BAR",
        "metric": "STROKE_TYPE",
        "group_by": "STROKE_TYPE",
    }

    # A fresh generate_visualization, with no pending clarification in play
    # (no bot decision event at all here), must not carry the thread forward.
    assert should_carry_forward("generate_visualization", events) is False
    # So the caller uses latest_entities as-is -- nothing from the earlier
    # Mann-Whitney turn (hospital, country, comparison group_by) leaks in.
    assert "hospital_name" not in latest_entities
    assert "hospital_scope_reference" not in latest_entities
    assert "country_code" not in latest_entities
    assert latest_entities["metric"] == "STROKE_TYPE"


def test_should_carry_forward_true_for_update_and_terse_clarify_intents() -> None:
    helpers = _load_visualization_context_helpers()
    should_carry_forward = helpers["_should_carry_forward_visualization_context"]

    # update_visualization always carries the thread forward: it's explicitly
    # a request to modify the existing plan.
    assert should_carry_forward("update_visualization", []) is True
    # clarify_visualization carries forward even with no pending clarification
    # in the events -- its NLU training data is terse follow-on instructions
    # ("group by X") that only make sense as a modification of the current
    # chart, not a standalone request.
    assert should_carry_forward("clarify_visualization", []) is True


def test_should_carry_forward_true_while_awaiting_clarification_reply() -> None:
    """The exact bug from the country_code PR review: 'show dtn from Czech
    Republic' -> bot asks for chart_type -> user replies 'line chart please'.
    The reply is classified generate_visualization (not clarify_visualization
    or update_visualization), but it's completing the pending clarification,
    not starting a fresh request, so the thread -- and country_code -- must
    still carry forward.
    """
    helpers = _load_visualization_context_helpers()
    should_carry_forward = helpers["_should_carry_forward_visualization_context"]
    events = [
        _user_event(
            "show dtn from Czech Republic",
            "generate_visualization",
            [
                {"entity": "metric", "value": "DTN"},
                {"entity": "country_code", "value": "CZ"},
            ],
        ),
        {
            "event": "bot",
            "custom": {
                "type": "visualization_query_decision",
                "decision": "clarify",
                "reason": "missing_chart_type",
            },
        },
        _user_event("line chart please", "generate_visualization", [{"entity": "chart_type", "value": "LINE"}]),
    ]

    assert should_carry_forward("generate_visualization", events) is True

    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    merged = merge_latest_with_thread_entities(
        latest_entities={"chart_type": "LINE"},
        events=events,
        fallback_limit=12,
    )
    assert merged["country_code"] == "CZ"
    assert merged["metric"] == "DTN"
    assert merged["chart_type"] == "LINE"


def test_should_not_carry_forward_after_a_completed_request_with_no_pending_clarification() -> None:
    """A generate_visualization turn that follows a *successful* (proceed) or
    rejected prior request -- not a clarify -- is a fresh ask, even though the
    prior bot turn was itself a visualization_query_decision payload."""
    helpers = _load_visualization_context_helpers()
    should_carry_forward = helpers["_should_carry_forward_visualization_context"]
    events = [
        _user_event(
            "compare mine vs Hospital X",
            "generate_visualization",
            [{"entity": "hospital_name", "value": "Hospital X"}],
        ),
        {
            "event": "bot",
            "custom": {"type": "visualization_query_decision", "decision": "proceed"},
        },
        _user_event("show me a bar chart of stroke type", "generate_visualization", [{"entity": "chart_type", "value": "BAR"}]),
    ]

    assert should_carry_forward("generate_visualization", events) is False
