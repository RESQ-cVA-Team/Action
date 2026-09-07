import logging
import threading
import time
from typing import Optional

import requests

from . import env as env_util

logger = logging.getLogger(__name__)

# Phase 2 of the cross-service auth redesign: Action's own service identity
# toward Webapp (currently just the static ACTION_SERVER_TOKEN/
# LONG_TASK_CALLBACK_TOKEN shared secrets) becomes a real Keycloak
# client-credentials token. Needs a dedicated Keycloak client with service
# accounts enabled -- unlike Rasa's introspection piece, this can't reuse
# Webapp's existing `cva` client (that one authenticates real users via the
# authorization-code flow, not a service via client_credentials).
#
# Deliberately optional: is_configured() lets callers fall back to the
# legacy static-token-only path when these env vars aren't set yet, so this
# ships dark until the Keycloak client actually exists.
_KEYCLOAK_ISSUER = env_util.get_env("KEYCLOAK_ISSUER")
_CLIENT_ID = env_util.get_env("ACTION_SERVICE_CLIENT_ID")
_CLIENT_SECRET = env_util.get_env("ACTION_SERVICE_CLIENT_SECRET")

# Refresh a bit before actual expiry so a request in flight doesn't race a
# token that expires mid-call.
_REFRESH_SAFETY_SECONDS = 30.0
_MIN_TTL_SECONDS = 5.0
_DEFAULT_TTL_SECONDS = 300.0

_lock = threading.Lock()
_cached_token: Optional[str] = None
_cached_expires_at: float = 0.0


def is_configured() -> bool:
    return bool(_KEYCLOAK_ISSUER and _CLIENT_ID and _CLIENT_SECRET)


def get_service_account_token() -> Optional[str]:
    """Fetch (and cache) a client-credentials access token for Action's own
    Keycloak service-account identity.

    Returns None if not configured, or if the token request fails -- callers
    should treat that as "fall back to the legacy static-token path", never
    as a reason to skip auth entirely.
    """
    if not is_configured():
        return None

    global _cached_token, _cached_expires_at
    with _lock:
        if _cached_token and time.monotonic() < _cached_expires_at:
            return _cached_token

        assert _KEYCLOAK_ISSUER and _CLIENT_ID and _CLIENT_SECRET  # narrowed by is_configured()
        try:
            resp = requests.post(
                f"{_KEYCLOAK_ISSUER.rstrip('/')}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": _CLIENT_ID,
                    "client_secret": _CLIENT_SECRET,
                },
                timeout=5,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            logger.warning("Failed to fetch Keycloak service-account token", exc_info=True)
            return None

        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            return None

        expires_in = payload.get("expires_in")
        ttl = float(expires_in) if isinstance(expires_in, (int, float)) else _DEFAULT_TTL_SECONDS

        _cached_token = token
        _cached_expires_at = time.monotonic() + max(_MIN_TTL_SECONDS, ttl - _REFRESH_SAFETY_SECONDS)
        return _cached_token
