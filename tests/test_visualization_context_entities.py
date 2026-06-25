from typing import Any, Dict, List

from src.actions.actions.visualization_action import merge_latest_with_thread_entities


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
    events = [
        _user_event(
            "Show DTN by month for past year split by sex",
            "generate_visualization",
            [
                {"entity": "chart_type", "value": "LINE"},
                {"entity": "metric", "value": "DTN"},
                {"entity": "group_by", "value": "month"},
                {"entity": "group_by", "value": "sex"},
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
    assert "month" in extracted["group_by"]
    assert "sex" in extracted["group_by"]


def test_merge_latest_with_thread_entities_prefers_latest_value() -> None:
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
