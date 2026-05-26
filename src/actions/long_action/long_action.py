from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast
from urllib.parse import urlsplit

import requests
from rasa_sdk import Action  # type: ignore

from src.util import env as env_util
from src.util.logging_utils import bind_current_context, log_context

from . import long_action_registry as registry
from .long_action_context import DispatcherLike, LongActionContext

_CALLBACK_TOKEN_ENV = "LONG_TASK_CALLBACK_TOKEN"
_CALLBACK_BASE_URL_ENV = "CALLBACK_BASE_URL"
_CALLBACK_ALLOWED_ORIGINS_ENV = "LONG_TASK_CALLBACK_ALLOWED_ORIGINS"
logger = logging.getLogger(__name__)
# Privacy/safety defaults: do not log callback payloads or URLs.
_LOG_CALLBACK_STATUS = env_util.env_flag("LONG_ACTION_LOG_CALLBACK_STATUS", default=False)
_LOG_CALLBACK_ERRORS = env_util.env_flag("LONG_ACTION_LOG_CALLBACK_ERRORS", default=False)
_DEFER_CALLBACK_HANDOFF = env_util.env_flag("LONG_ACTION_DEFER_CALLBACK_HANDOFF", default=False)


def _callback_endpoint_label(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    if parsed.netloc:
        return parsed.netloc
    return url


def _normalize_callback_origin(url: str) -> Optional[str]:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _allowed_callback_origins() -> List[str]:
    configured_origins = env_util.get_env(_CALLBACK_ALLOWED_ORIGINS_ENV, "") or ""
    base_callback_url = env_util.get_env(_CALLBACK_BASE_URL_ENV, "") or ""
    candidates = [*configured_origins.replace(";", "\n").replace(",", "\n").splitlines(), base_callback_url]

    allowed: List[str] = []
    for candidate in candidates:
        normalized = _normalize_callback_origin(candidate)
        if normalized and normalized not in allowed:
            allowed.append(normalized)

    return allowed


def _event_list() -> List[Dict[str, Any]]:
    return []


@dataclass
class PreworkResult:
    """Outcome of LongAction.prework.

    - events: immediate Rasa events to return from action run.
    - proceed: whether async/sync work phase should continue.
    """

    events: List[Dict[str, Any]] = field(default_factory=_event_list)
    proceed: bool = True


DomainDict = Dict[str, Any]
RasaEventList = List[Dict[str, Any]]


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]
    events: List[Dict[str, Any]]

    def current_state(self) -> Dict[str, Any]: ...


def _get_callback_config(tracker: TrackerLike) -> Optional[Tuple[str, str]]:
    """Return (url, token) for the long-task callback if configured.

    The callback URL is taken from the incoming message metadata as
    `metadata.callback_url`. If that is not present or empty, callback mode is
    considered unsupported for this turn. The token is read from the
    LONG_TASK_CALLBACK_TOKEN environment variable.
    """

    callback_url: Optional[str] = None

    meta_any = tracker.latest_message.get("metadata")
    if isinstance(meta_any, dict):
        meta = cast(Dict[str, Any], meta_any)
        url_val = meta.get("callback_url")
        if isinstance(url_val, str) and url_val:
            callback_url = url_val

    if not callback_url:
        return None

    token = os.getenv(_CALLBACK_TOKEN_ENV) or ""
    if not token:
        logger.warning(
            "Callback URL present but %s is not configured; falling back to synchronous execution",
            _CALLBACK_TOKEN_ENV,
            extra={
                "log_context": {
                    "callback_endpoint": _callback_endpoint_label(callback_url),
                    "callback_mode": False,
                    "misconfiguration": True,
                }
            },
        )
        return None

    callback_origin = _normalize_callback_origin(callback_url)
    if not callback_origin:
        logger.warning(
            "Callback URL present but invalid; falling back to synchronous execution",
            extra={
                "log_context": {
                    "callback_endpoint": _callback_endpoint_label(callback_url),
                    "callback_mode": False,
                    "misconfiguration": True,
                }
            },
        )
        return None

    allowed_origins = _allowed_callback_origins()
    if allowed_origins and callback_origin not in allowed_origins:
        logger.warning(
            "Callback URL origin is not allowed; falling back to synchronous execution",
            extra={
                "log_context": {
                    "callback_endpoint": callback_origin,
                    "callback_mode": False,
                    "misconfiguration": True,
                    "allowed_callback_origins": allowed_origins,
                }
            },
        )
        return None

    if not allowed_origins:
        logger.warning(
            "Callback URL present but no callback origin allowlist is configured; accepting callback URL without origin validation",
            extra={
                "log_context": {
                    "callback_endpoint": callback_origin,
                    "callback_mode": True,
                    "misconfiguration": True,
                }
            },
        )

    return callback_url, token


def _normalize_trace_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
    else:
        token = str(value).strip()
    return token or None


def _trace_id_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        trace_id = _normalize_trace_id(metadata.get(key))
        if trace_id:
            return trace_id

    headers_any = metadata.get("headers")
    headers = cast(Dict[str, Any], headers_any) if isinstance(headers_any, dict) else {}
    for key in ("x-trace-id", "x_trace_id", "trace_id", "traceId"):
        trace_id = _normalize_trace_id(headers.get(key))
        if trace_id:
            return trace_id

    return None


def _trace_id_from_message(message: Dict[str, Any]) -> Optional[str]:
    custom_any = message.get("custom")
    custom = cast(Dict[str, Any], custom_any) if isinstance(custom_any, dict) else {}
    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        trace_id = _normalize_trace_id(custom.get(key))
        if trace_id:
            return trace_id
    return None


def _resolve_progress_trace_id(ctx: LongActionContext, message: Dict[str, Any]) -> Optional[str]:
    message_trace_id = _trace_id_from_message(message)
    if message_trace_id:
        return message_trace_id

    metadata_trace_id = _trace_id_from_metadata(ctx.metadata)
    if metadata_trace_id:
        return metadata_trace_id

    for key in ("_visualization_trace_id", "_trace_id", "trace_id"):
        trace_id = _normalize_trace_id(ctx.tracker_snapshot.get(key))
        if trace_id:
            return trace_id

    return None


class LongAction(Action, ABC):
    def __init__(self):
        registry.register(self)

    async def prework(self, ctx: LongActionContext) -> PreworkResult:
        """Optional in-band phase before work() starts.

        Runs with dispatcher-backed context so messages/events are handled like
        a normal action run. Subclasses can override to perform quick routing,
        slot updates, or early exits prior to long-running callback work.
        """

        return PreworkResult()

    async def run(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> RasaEventList:
        sender_id = tracker.sender_id
        latest_message_any = getattr(tracker, "latest_message", None)
        latest_message = cast(Dict[str, Any], latest_message_any) if isinstance(latest_message_any, dict) else {}
        metadata_any = latest_message.get("metadata")
        metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
        request_trace_id = _trace_id_from_message(latest_message) or _trace_id_from_metadata(metadata)

        tracker_snapshot: Dict[str, Any] = {
            "latest_message": tracker.latest_message,
            "slots": tracker.current_state().get("slots", {}),
        }
        events_any = getattr(tracker, "events", None)
        if isinstance(events_any, list):
            tracker_snapshot["events"] = list(cast(List[Any], events_any))

        callback_cfg = _get_callback_config(tracker)

        log_fields: Dict[str, Any] = {"sender_id": sender_id, "action": self.name()}
        if request_trace_id:
            log_fields["trace_id"] = request_trace_id

        with log_context(**log_fields):
            # Prework always runs in dispatcher mode so subclasses can emit normal
            # in-band messages and return Rasa events before any long-running work.
            pre_ctx = LongActionContext(sender_id=sender_id, tracker_snapshot=tracker_snapshot, dispatcher=dispatcher)
            pre_outcome = await self.prework(pre_ctx)
            immediate_events = pre_outcome.events
            if not pre_outcome.proceed:
                return immediate_events

            # If no callback is configured, fall back to synchronous execution so
            # behavior is predictable in rasa shell and simple REST setups.
            if callback_cfg is None:
                ctx = LongActionContext(sender_id=sender_id, tracker_snapshot=tracker_snapshot, dispatcher=dispatcher)
                await self.work(ctx)
                return [*immediate_events, *ctx.pending_events]

            # Callback is configured: run the long task asynchronously and notify
            # the frontend via HTTP callback when finished. We do not schedule Rasa
            # reminders or use a poller in this mode.
            callback_url, callback_token = callback_cfg
            job_id = uuid.uuid4().hex

            if _DEFER_CALLBACK_HANDOFF:
                # Optional hybrid mode: start in normal dispatcher path and let the
                # action explicitly switch to callback transport via
                # ctx.enable_callback_mode().
                ctx = LongActionContext(sender_id=sender_id, tracker_snapshot=tracker_snapshot, dispatcher=dispatcher)
                ctx.attach_progress_callback(
                    lambda message, ctx=ctx, job_id=job_id, callback_url=callback_url, callback_token=callback_token: self._post_progress(
                        ctx,
                        job_id,
                        callback_url,
                        callback_token,
                        message,
                    )
                )
                with log_context(job_id=job_id, callback_mode=True):
                    await self.work(ctx)
                return [*immediate_events, *ctx.pending_events]

            ctx = LongActionContext(sender_id=sender_id, tracker_snapshot=tracker_snapshot)

            # In callback mode, stream every ctx.say() as a progress callback to
            # the frontend while the job is running.
            ctx.attach_progress_callback(
                lambda message, ctx=ctx, job_id=job_id, callback_url=callback_url, callback_token=callback_token: self._post_progress(
                    ctx,
                    job_id,
                    callback_url,
                    callback_token,
                    message,
                )
            )

            threading.Thread(
                target=bind_current_context(self._run_work),
                args=(ctx, job_id, callback_url, callback_token),
                daemon=True,
            ).start()

            # No additional events required; we rely on the external callback.
            return immediate_events

    def _post_progress(
        self,
        ctx: LongActionContext,
        job_id: str,
        callback_url: str,
        callback_token: str,
        message: Dict[str, Any],
    ) -> None:
        """Send a callback for a single ctx.say() message.

        The payload shape is kept intentionally simple and dispatcher-like so
        the frontend can treat it similarly to Rasa messages:

        {
            "senderId": "...",
            "messages": [ { ...ctx.say kwargs... } ]
        }
        """

        payload: Dict[str, Any] = {
            "senderId": ctx.sender_id,
            "messages": [message],
        }
        trace_id = _resolve_progress_trace_id(ctx, message)
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-long-task-callback-token": callback_token,
        }
        if isinstance(trace_id, str) and trace_id.strip():
            headers["x-trace-id"] = trace_id.strip()

        with log_context(
            trace_id=trace_id or "-",
            job_id=job_id,
            callback_mode=True,
            callback_endpoint=_callback_endpoint_label(callback_url),
        ):
            try:
                resp = requests.post(
                    callback_url,
                    headers=headers,
                    data=json.dumps(payload, default=str),
                    timeout=10,
                )
                status_code = getattr(resp, "status_code", None)
                if isinstance(status_code, int) and 200 <= status_code < 300:
                    if _LOG_CALLBACK_STATUS:
                        logger.debug(
                            "LongAction callback posted (status=%s)",
                            status_code,
                        )
                    return

                body_len = len(getattr(resp, "text", "") or "")
                log_method = logger.error if isinstance(status_code, int) and status_code >= 500 else logger.warning
                log_method(
                    "LongAction callback returned HTTP %s",
                    status_code,
                    extra={
                        "log_context": {
                            "error_category": "http_error",
                            "callback_status": status_code if isinstance(status_code, int) else "-",
                            "body_len": body_len,
                        }
                    },
                )
            except requests.Timeout as exc:
                logger.error(
                    "LongAction callback timeout: %s",
                    exc,
                    exc_info=_LOG_CALLBACK_ERRORS,
                    extra={
                        "log_context": {
                            "error_category": "timeout",
                            "error_type": type(exc).__name__,
                        }
                    },
                )
            except requests.ConnectionError as exc:
                logger.error(
                    "LongAction callback connection failure: %s",
                    exc,
                    exc_info=_LOG_CALLBACK_ERRORS,
                    extra={
                        "log_context": {
                            "error_category": "connection_error",
                            "error_type": type(exc).__name__,
                        }
                    },
                )
            except requests.RequestException as exc:
                logger.error(
                    "LongAction callback request exception: %s",
                    exc,
                    exc_info=_LOG_CALLBACK_ERRORS,
                    extra={
                        "log_context": {
                            "error_category": "request_error",
                            "error_type": type(exc).__name__,
                        }
                    },
                )

    def _run_work(self, ctx: LongActionContext, job_id: str, callback_url: str, callback_token: str) -> None:
        try:
            asyncio.run(self.work(ctx))
        except Exception:
            logger.exception("LongAction work failed")
            # Fail closed: emit an error as a normal message so the user sees
            # something, but do not propagate the exception.
            ctx.say(text="Something went wrong.")
            ctx.done()

    @abstractmethod
    async def work(self, ctx: LongActionContext) -> Any:
        """Long-running logic. Must end with ctx.done().

        Use ``ctx.say(...)`` to emit any messages or structured payloads. In
        callback mode, each ``ctx.say`` results in a callback JSON of the
        form::

            {"senderId": "...", "messages": [{...}]}

        The return value is not sent to the frontend and is only for
        internal use by subclasses if needed.
        """
        raise NotImplementedError
