from typing import Any, Callable, Dict, List, Optional, cast


class LongActionContext:
    def __init__(self, sender_id: str, tracker_snapshot: Dict[str, Any], dispatcher: Optional[Any] = None):
        self.sender_id = sender_id
        self.tracker_snapshot = tracker_snapshot
        self.dispatcher = dispatcher
        self._pending_events: List[Dict[str, Any]] = []
        # Optional progress callback used in callback mode to stream
        # individual messages back to the frontend as they are emitted.
        self._progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._callback_mode_enabled: bool = dispatcher is None

    @property
    def text(self) -> str:
        latest_any = self.tracker_snapshot.get("latest_message")
        if isinstance(latest_any, dict):
            latest = cast(Dict[str, Any], latest_any)
            text_val = latest.get("text")
            if isinstance(text_val, str):
                return text_val
        return ""

    @property
    def metadata(self) -> Dict[str, Any]:
        latest_any = self.tracker_snapshot.get("latest_message")
        if isinstance(latest_any, dict):
            latest = cast(Dict[str, Any], latest_any)
            meta_val = latest.get("metadata")
            if isinstance(meta_val, dict):
                return cast(Dict[str, Any], meta_val)
        return {}

    @property
    def slots(self) -> Dict[str, Any]:
        """Return the tracked slots snapshot for this long action."""
        slots_any = self.tracker_snapshot.get("slots")
        if isinstance(slots_any, dict):
            return cast(Dict[str, Any], slots_any)
        return {}

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Return tracker events snapshot (bounded, if enabled)."""
        events_any = self.tracker_snapshot.get("events")
        if isinstance(events_any, list):
            out: List[Dict[str, Any]] = []
            for item in cast(List[Any], events_any):
                if isinstance(item, dict):
                    out.append(cast(Dict[str, Any], item))
            return out
        return []

    def recent_user_messages(self, limit: int = 6) -> List[str]:
        """Return recent user utterances from event history.

        This inspects tracker events for user events and returns up to `limit`
        latest non-empty texts in chronological order.
        """
        if limit <= 0:
            return []
        out: List[str] = []
        for ev in self.events:
            if ev.get("event") != "user":
                continue
            text_any = ev.get("text")
            if isinstance(text_any, str) and text_any.strip():
                out.append(text_any.strip())
        if len(out) <= limit:
            return out
        return out[-limit:]

    def user_messages_since_intent(self, intent_name: str, fallback_limit: int = 12) -> List[str]:
        """Return user messages from the latest occurrence of an intent onward.

        This scans tracker events backwards to locate the most recent user event
        whose parsed intent name matches ``intent_name``. If no anchor is found,
        it falls back to ``recent_user_messages(fallback_limit)``.
        """
        events = self.events
        if not events:
            return []

        wanted = (intent_name or "").strip()
        if not wanted:
            return self.recent_user_messages(limit=fallback_limit)

        anchor_idx = -1
        for idx in range(len(events) - 1, -1, -1):
            ev = events[idx]
            if ev.get("event") != "user":
                continue

            parse_data_any = ev.get("parse_data")
            parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}

            intent_any = parse_data.get("intent")
            intent = cast(Dict[str, Any], intent_any) if isinstance(intent_any, dict) else {}
            name_any = intent.get("name")

            if not isinstance(name_any, str) or not name_any.strip():
                fallback_intent_any = ev.get("intent")
                fallback_intent = cast(Dict[str, Any], fallback_intent_any) if isinstance(fallback_intent_any, dict) else {}
                fallback_name_any = fallback_intent.get("name")
                if isinstance(fallback_name_any, str) and fallback_name_any.strip():
                    name_any = fallback_name_any

            if isinstance(name_any, str) and name_any.strip() == wanted:
                anchor_idx = idx
                break

        if anchor_idx < 0:
            return self.recent_user_messages(limit=fallback_limit)

        out: List[str] = []
        for ev in events[anchor_idx:]:
            if ev.get("event") != "user":
                continue
            text_any = ev.get("text")
            if isinstance(text_any, str) and text_any.strip():
                out.append(text_any.strip())

        return out if out else self.recent_user_messages(limit=fallback_limit)

    def say(self, **kwargs: Any) -> None:
        """
        - ctx.say(text="hi")  normal message
        - ctx.say(json_message={...}) / ctx.say(image="...")  normal message
        - ctx.say(progress="...")  special progress event (frontend decides how to render/replace)
        """
        if self.dispatcher is not None and not self._callback_mode_enabled:
            # Synchronous mode: send directly via dispatcher.
            self.dispatcher.utter_message(**kwargs)
            return

        # Callback mode: normalise dispatcher-style kwargs into the same
        # shape Rasa's REST channel produces, i.e. content lives under
        # "custom".
        message: Dict[str, Any] = dict(kwargs)

        # Progress helper: ctx.say(progress="...") becomes
        # {"custom": {"progress": "..."}} when there is no explicit
        # custom/json_message override.
        if "progress" in message and "custom" not in message and "json_message" not in message:
            progress_val = message.pop("progress")
            message = {"custom": {"progress": progress_val}}

        # json_message helper: ctx.say(json_message={...}) should mirror
        # dispatcher.utter_message(json_message=...) which surfaces the
        # payload under "custom" in REST responses.
        if "json_message" in message and "custom" not in message:
            custom_val = message.pop("json_message")
            message = {"custom": custom_val}

        # If a progress callback is configured, invoke it so the message is
        # streamed immediately to the frontend.
        if self._progress_callback is not None:
            try:
                self._progress_callback(message)
            except Exception:
                # Swallow exceptions here so a failing callback endpoint
                # doesn't break the long-running job logic.
                pass
            return

        # Fallback path: if callback mode is enabled but no callback hook is
        # available, use dispatcher to avoid dropping messages.
        if self.dispatcher is not None:
            self.dispatcher.utter_message(**kwargs)

    def attach_progress_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to be invoked on every ctx.say() in callback mode."""
        self._progress_callback = cb

    def enable_callback_mode(self) -> None:
        """Route subsequent ctx.say calls through callback transport when available."""
        self._callback_mode_enabled = True

    def disable_callback_mode(self) -> None:
        """Route subsequent ctx.say calls through dispatcher transport."""
        self._callback_mode_enabled = False

    @property
    def callback_mode_enabled(self) -> bool:
        return self._callback_mode_enabled

    def done(self) -> None:
        if self.dispatcher is not None:
            # Nothing to clean up in synchronous mode.
            return

    @property
    def pending_events(self) -> List[Dict[str, Any]]:
        return list(self._pending_events)

    def add_event(self, event: Dict[str, Any]) -> None:
        if isinstance(event, dict):
            self._pending_events.append(event)

    def add_events(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            self.add_event(event)
