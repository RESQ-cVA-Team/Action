import logging
import re
from typing import Optional

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


def introspect_token_sync(token: str) -> Optional[str]:
    """Verify a user access token via Keycloak introspection using Action's own
    confidential client; return the verified sub, or None."""
    issuer = env_util.get_env("KEYCLOAK_ISSUER")
    client_id = env_util.get_env("KEYCLOAK_CLIENT_ID")
    client_secret = env_util.get_env("KEYCLOAK_CLIENT_SECRET")
    if not (issuer and client_id and client_secret):
        return None
    try:
        resp = requests.post(
            f"{issuer.rstrip('/')}/protocol/openid-connect/token/introspect",
            data={"token": token, "client_id": client_id, "client_secret": client_secret},
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        logger.warning("Keycloak token introspection request failed", exc_info=True)
        return None

    if not payload.get("active") or not _has_required_role(payload):
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None
