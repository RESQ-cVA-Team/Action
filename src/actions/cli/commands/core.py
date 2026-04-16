from __future__ import annotations

from typing import Any, Dict, List

from rasa_sdk.events import EventType, FollowupAction, SlotSet  # type: ignore

from . import command, names


@command("help")
def _help(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    lines = [
        "CLI commands:",
        "  help                       Show this help",
        "  ping                       Simple connectivity check",
        "  gql | test_gql             Run GraphQL smoke test",
        "  hospitals | test_hospitals Test provider listing/search filters",
        "  test_charts                Send demo charts payload (all charts)",
        "  test_line                  Send a single demo line chart",
        "  test_analytics             Demo chart + typed statistical results",
        "  test_bar/test_pie/...      Per-chart test commands (bar, pie, histogram, box, scatter, radar, waterfall, area, line)",
        "  run <action>               Run an allowed action",
        "  set k=v [k2=v2...]         Set slots",
        "",
        f"Available: {', '.join(sorted(names()))}",
    ]
    dispatcher.utter_message(text="\n".join(lines))
    return []


@command("ping")
def _ping(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    dispatcher.utter_message(text="pong")
    return []


ALLOWED_ACTIONS = {
    "action_clarify_visualization_request",
    "action_guided_generate_visualization",
    "action_oneshot_generate_visualization",
    "action_dummy_countdown",
}


@command("run")
def _run(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    if not args:
        dispatcher.utter_message(text="Usage: /cli run <action>")
        return []
    action = args[0]
    if action not in ALLOWED_ACTIONS:
        dispatcher.utter_message(text=f"Action not allowed: {action}")
        return []
    return [FollowupAction(action)]


@command("set")
def _set(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    events: List[EventType] = []
    for token in args:
        if "=" in token:
            k, v = token.split("=", 1)
            events.append(SlotSet(k, v))
    for k, v in opts.items():
        events.append(SlotSet(k, v))
    if not events:
        dispatcher.utter_message(text="No slots provided to set. Usage: /cli set key=value ...")
        return []
    dispatcher.utter_message(text=f"Set {len(events)} slot(s).")
    return events


# Reference functions to appease static analyzers about unused symbols.
_ = (_help, _ping, _run, _set)
