import logging
import shlex
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import EventType  # type: ignore

from src.util.logging_utils import log_context

from .commands import get as get_command  # type: ignore
from .commands import names as list_command_names

logger = logging.getLogger(__name__)


DomainDict = Dict[str, Any]


class DispatcherLike(Protocol):
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]


def _coerce_scalar(val: str) -> Any:
    """Best-effort cast of string token to int/float/bool/str."""
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(val)
        except Exception:
            pass
    return val


def _parse_cmd(cmdline: str) -> Tuple[str | None, List[str], Dict[str, Any]]:
    """
    Parse a CLI command line into (subcommand, args, options).
    Supports:
      - flags: -k v, --key v, --flag (True)
      - key=value tokens
      - positional args collected into args list
    """
    parts = shlex.split(cmdline or "")
    if not parts:
        return None, [], {}
    sub = parts[0]
    args: List[str] = []
    opts: Dict[str, Any] = {}

    i = 1
    while i < len(parts):
        t = parts[i]
        if "=" in t and not t.startswith("--="):
            key, val = t.split("=", 1)
            opts[key.lstrip("-")] = _coerce_scalar(val)
        elif t.startswith("--") or t.startswith("-"):
            key = t.lstrip("-")
            # standalone flag
            if i + 1 >= len(parts) or parts[i + 1].startswith("-"):
                opts[key] = True
            else:
                opts[key] = _coerce_scalar(parts[i + 1])
                i += 1
        else:
            args.append(t)
        i += 1

    return sub, args, opts


class ActionCliRouter(Action):
    """
    Generic CLI router for developer shortcuts.

    Examples:
      /cli help
      /cli test_gql
      /cli gql --verbose
      /cli set session_token=abc provider=1
      /cli run action_test_graphql
    """

    def name(self) -> str:
        return "action_cli_router"

    def run(self, dispatcher: DispatcherLike, tracker: TrackerLike, domain: DomainDict) -> List[EventType]:  # type: ignore[override]
        latest_message_any = getattr(tracker, "latest_message", None)
        latest_message = cast(Dict[str, Any], latest_message_any) if isinstance(latest_message_any, dict) else {}
        md_any = latest_message.get("metadata")
        md: Dict[str, Any] = cast(Dict[str, Any], md_any) if isinstance(md_any, dict) else {}
        cmdline: str = str(md.get("cli_command_text", ""))
        sub, args, opts = _parse_cmd(cmdline)

        # If no subcommand, default to 'help'
        if not sub:
            sub = "help"

        with log_context(
            sender_id=str(getattr(tracker, "sender_id", "")),
            action=self.name(),
            cli_subcommand=sub,
            cli_arg_count=len(args),
            cli_option_count=len(opts),
        ):
            logger.info("Routing CLI command")

            # Try pluggable command handlers first
            handler = get_command(sub)
            if handler is not None:
                return handler(dispatcher, tracker, domain, args, opts)

            # No static duplicates; registry owns all commands now

            # Fallback: unknown command
            logger.warning("Unknown CLI command")
            dispatcher.utter_message(text=f"Unknown CLI command: {sub}. Try one of: {', '.join(sorted(list_command_names()))}")
            return []
