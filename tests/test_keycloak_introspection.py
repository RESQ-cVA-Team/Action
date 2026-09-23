from unittest import mock

from src.util import keycloak_introspection

SUB = "user-1"


def _post(monkeypatch, payload: dict):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = payload
    monkeypatch.setattr(keycloak_introspection.requests, "post", mock.Mock(return_value=resp))


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
