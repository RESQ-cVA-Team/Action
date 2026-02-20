from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from rasa_sdk import Action, Tracker  # type: ignore
from rasa_sdk import types as rasa_types  # type: ignore
from rasa_sdk.executor import CollectingDispatcher  # type: ignore

from src.util import env as env_util

from . import long_action_registry as registry
from .long_action_context import LongActionContext

_CALLBACK_TOKEN_ENV = "LONG_TASK_CALLBACK_TOKEN"
# Privacy/safety defaults: do not log callback payloads or URLs.
_LOG_CALLBACK_STATUS = env_util.env_flag("LONG_ACTION_LOG_CALLBACK_STATUS", default=False)
_LOG_CALLBACK_ERRORS = env_util.env_flag("LONG_ACTION_LOG_CALLBACK_ERRORS", default=False)


def _get_callback_config(tracker: Tracker) -> Optional[Tuple[str, str]]:
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
        if isinstance(url_val, str) and url_val.strip():
            callback_url = url_val.strip()

    if not callback_url:
        return None

    token = os.getenv(_CALLBACK_TOKEN_ENV) or ""
    if not token:
        return None

    return callback_url, token


class LongAction(Action, ABC):
    def __init__(self):
        registry.register(self)

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: rasa_types.DomainDict,
    ) -> List[Dict[str, Any]]:
        sender_id = tracker.sender_id

        tracker_snapshot: Dict[str, Any] = {
            "latest_message": tracker.latest_message,
            "slots": tracker.current_state().get("slots", {}),
        }

        callback_cfg = _get_callback_config(tracker)

        # If no callback is configured, fall back to synchronous execution so
        # behavior is predictable in rasa shell and simple REST setups.
        if callback_cfg is None:
            ctx = LongActionContext(sender_id=sender_id, tracker_snapshot=tracker_snapshot, dispatcher=dispatcher)
            await self.work(ctx)
            return []

        # Callback is configured: run the long task asynchronously and notify
        # the frontend via HTTP callback when finished. We do not schedule Rasa
        # reminders or use a poller in this mode.
        callback_url, callback_token = callback_cfg
        job_id = uuid.uuid4().hex

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
            target=self._run_work,
            args=(ctx, job_id, callback_url, callback_token),
            daemon=True,
        ).start()

        # No additional events required; we rely on the external callback.
        return []

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

        try:
            resp = requests.post(
                callback_url,
                headers={
                    "Content-Type": "application/json",
                    "x-action-server-token": callback_token,
                },
                data=json.dumps(payload, default=str),
                timeout=10,
            )
            if _LOG_CALLBACK_STATUS:
                logger.debug("LongAction callback posted (status=%s)", getattr(resp, "status_code", None))
        except Exception:
            # Swallow errors from the callback endpoint; they should not
            # break the long-running job.
            if _LOG_CALLBACK_ERRORS:
                logger.debug("LongAction callback post failed", exc_info=True)

    def _run_work(self, ctx: LongActionContext, job_id: str, callback_url: str, callback_token: str) -> None:
        try:
            asyncio.run(self.work(ctx))
        except Exception:
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
