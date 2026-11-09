import hashlib
import logging
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

import requests

from . import env as env_util

logger = logging.getLogger(__name__)

_SENDER_THREAD_SUFFIX_RE = re.compile(r"^(.*):thread:(\d+)$")

# Keycloak realm role a token must carry to use cVA at all -- mirrors
# Webapp's REQUIRED_CVA_ROLE (src/lib/cvaAccess.ts) and Rasa's own check.
# Checked here too, not just relied on from Rasa's gate: this endpoint is
# reachable directly on Action's own port, not only via Rasa's proxy.
REQUIRED_ROLE = "cva"


def sender_sub(sender_id: str) -> str:
    """Strip the `:thread:<n>` suffix from a Rasa sender_id, leaving the Keycloak sub."""
    match = _SENDER_THREAD_SUFFIX_RE.match(sender_id)
    return match.group(1) if match else sender_id


def _has_required_role(introspection_payload: dict) -> bool:
    """Realm roles only: the top-level `roles` claim and `realm_access.roles`,
    never groups or client/resource roles."""
    roles: list = []
    for value in (introspection_payload.get("roles"), (introspection_payload.get("realm_access") or {}).get("roles")):
        if isinstance(value, list):
            roles.extend(entry.strip().lower() for entry in value if isinstance(entry, str) and entry.strip())
    return REQUIRED_ROLE in roles


_VERIFIED_MAX = 1024
_verified: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_verified_lock = Lock()


def _reuse_seconds() -> float:
    """How long a verified token is reused without asking Keycloak again
    (ACTIONS_TOKEN_CHECK_REUSE_SECONDS, never past the token's own expiry).
    0, the default, checks every call."""
    try:
        return max(0.0, float(env_util.get_env("ACTIONS_TOKEN_CHECK_REUSE_SECONDS", default="") or 0))
    except ValueError:
        return 0.0


class IntrospectionUnavailable(Exception):
    """Keycloak could not be asked whether a token is valid."""


def _post_introspection(url: str, data: Dict[str, str]) -> Dict[str, Any]:
    """One retry on a timeout, a connection error or a Keycloak server error."""
    for attempt in (1, 2):
        try:
            resp = requests.post(url, data=data, timeout=5)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.json()
            failure: Exception = requests.HTTPError(f"Keycloak answered {resp.status_code}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            failure = exc
        logger.warning("Keycloak token introspection attempt %d failed: %s", attempt, failure)
    raise IntrospectionUnavailable("Keycloak token introspection is unavailable") from failure


def introspect_token_sync(token: str) -> Optional[str]:
    """Verify a user access token via Keycloak introspection using Action's own
    confidential client; return the verified sub, or None for a token Keycloak
    does not accept. Raises IntrospectionUnavailable when Keycloak cannot be
    reached."""
    issuer = env_util.get_env("KEYCLOAK_ISSUER")
    client_id = env_util.get_env("KEYCLOAK_CLIENT_ID")
    client_secret = env_util.get_env("KEYCLOAK_CLIENT_SECRET")
    if not (issuer and client_id and client_secret):
        return None

    reuse = _reuse_seconds()
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = time.time()
    if reuse > 0:
        with _verified_lock:
            remembered = _verified.get(key)
            if remembered and remembered[1] > now:
                return remembered[0]

    try:
        payload = _post_introspection(
            f"{issuer.rstrip('/')}/protocol/openid-connect/token/introspect",
            {"token": token, "client_id": client_id, "client_secret": client_secret},
        )
    except IntrospectionUnavailable:
        raise
    except Exception:
        logger.warning("Keycloak token introspection request failed", exc_info=True)
        return None

    if not payload.get("active") or not _has_required_role(payload):
        return None
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        return None

    expires = payload.get("exp")
    until = min(now + reuse, float(expires)) if isinstance(expires, (int, float)) else now + reuse
    if until > now:
        with _verified_lock:
            _verified[key] = (sub, until)
            _verified.move_to_end(key)
            while len(_verified) > _VERIFIED_MAX:
                _verified.popitem(last=False)
    return sub
