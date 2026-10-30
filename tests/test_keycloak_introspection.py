import time
from unittest import mock

import pytest
import requests

from src.util import keycloak_introspection

SUB = "user-1"


@pytest.fixture(autouse=True)
def _fresh_checks(monkeypatch):
    monkeypatch.delenv("ACTIONS_TOKEN_CHECK_REUSE_SECONDS", raising=False)
    keycloak_introspection._verified.clear()


def _response(payload: dict, status: int = 200):
    resp = mock.Mock(status_code=status)
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = payload
    return resp


def _post(monkeypatch, payload: dict):
    monkeypatch.setattr(keycloak_introspection.requests, "post", mock.Mock(return_value=_response(payload)))


def _env(monkeypatch):
    monkeypatch.setenv("KEYCLOAK_ISSUER", "https://keycloak.example/realms/stroke")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "action")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "secret")


def test_active_token_with_role_returns_sub(monkeypatch):
    _env(monkeypatch)
    _post(monkeypatch, {"active": True, "sub": SUB, "roles": ["cva"]})
    assert keycloak_introspection.introspect_token_sync("tok") == SUB


def test_active_token_with_role_in_realm_access_returns_sub(monkeypatch):
    _env(monkeypatch)
    _post(monkeypatch, {"active": True, "sub": SUB, "realm_access": {"roles": ["default-roles-stroke", "cva"]}})
    assert keycloak_introspection.introspect_token_sync("tok") == SUB


def test_active_token_without_role_is_rejected(monkeypatch):
    _env(monkeypatch)
    _post(monkeypatch, {"active": True, "sub": SUB, "roles": ["offline_access"]})
    assert keycloak_introspection.introspect_token_sync("tok") is None


def test_client_or_resource_role_does_not_count(monkeypatch):
    _env(monkeypatch)
    _post(monkeypatch, {"active": True, "sub": SUB, "resource_access": {"account": {"roles": ["cva"]}}})
    assert keycloak_introspection.introspect_token_sync("tok") is None


def test_inactive_token_is_rejected(monkeypatch):
    _env(monkeypatch)
    _post(monkeypatch, {"active": False})
    assert keycloak_introspection.introspect_token_sync("tok") is None


def test_missing_keycloak_config_is_rejected(monkeypatch):
    monkeypatch.delenv("KEYCLOAK_ISSUER", raising=False)
    monkeypatch.delenv("KEYCLOAK_CLIENT_ID", raising=False)
    monkeypatch.delenv("KEYCLOAK_CLIENT_SECRET", raising=False)
    assert keycloak_introspection.introspect_token_sync("tok") is None


def test_request_failure_is_rejected(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(keycloak_introspection.requests, "post", mock.Mock(side_effect=OSError("down")))
    assert keycloak_introspection.introspect_token_sync("tok") is None


def test_one_retry_after_a_timeout(monkeypatch):
    _env(monkeypatch)
    post = mock.Mock(side_effect=[requests.Timeout("slow"), _response({"active": True, "sub": SUB, "roles": ["cva"]})])
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    assert keycloak_introspection.introspect_token_sync("tok") == SUB
    assert post.call_count == 2


def test_one_retry_after_a_keycloak_server_error(monkeypatch):
    _env(monkeypatch)
    post = mock.Mock(side_effect=[_response({}, status=503), _response({"active": True, "sub": SUB, "roles": ["cva"]})])
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    assert keycloak_introspection.introspect_token_sync("tok") == SUB


def test_unreachable_keycloak_is_unavailable_not_rejected(monkeypatch):
    _env(monkeypatch)
    post = mock.Mock(side_effect=requests.ConnectionError("down"))
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    with pytest.raises(keycloak_introspection.IntrospectionUnavailable):
        keycloak_introspection.introspect_token_sync("tok")
    assert post.call_count == 2


def test_every_call_is_checked_by_default(monkeypatch):
    _env(monkeypatch)
    post = mock.Mock(return_value=_response({"active": True, "sub": SUB, "roles": ["cva"]}))
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    keycloak_introspection.introspect_token_sync("tok")
    keycloak_introspection.introspect_token_sync("tok")
    assert post.call_count == 2


def test_a_verified_token_is_reused_within_the_window(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("ACTIONS_TOKEN_CHECK_REUSE_SECONDS", "60")
    post = mock.Mock(return_value=_response({"active": True, "sub": SUB, "roles": ["cva"], "exp": time.time() + 300}))
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    assert keycloak_introspection.introspect_token_sync("tok") == SUB
    assert keycloak_introspection.introspect_token_sync("tok") == SUB
    assert post.call_count == 1


def test_reuse_never_outlives_the_token(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("ACTIONS_TOKEN_CHECK_REUSE_SECONDS", "60")
    post = mock.Mock(return_value=_response({"active": True, "sub": SUB, "roles": ["cva"], "exp": time.time() - 1}))
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    keycloak_introspection.introspect_token_sync("tok")
    keycloak_introspection.introspect_token_sync("tok")
    assert post.call_count == 2


def test_a_rejected_token_is_never_reused(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("ACTIONS_TOKEN_CHECK_REUSE_SECONDS", "60")
    post = mock.Mock(return_value=_response({"active": False}))
    monkeypatch.setattr(keycloak_introspection.requests, "post", post)
    assert keycloak_introspection.introspect_token_sync("tok") is None
    assert keycloak_introspection.introspect_token_sync("tok") is None
    assert post.call_count == 2
