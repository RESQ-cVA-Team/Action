import asyncio
from types import SimpleNamespace

import pytest

import rasa_sdk_plugins
from src.util import keycloak_introspection


class _StubApp:
    def __init__(self):
        self.request_hooks = []

    def on_request(self, fn):
        self.request_hooks.append(fn)
        return fn

    def get(self, *_a, **_k):
        return lambda fn: fn

    post = get


@pytest.fixture
def gate():
    app = _StubApp()
    rasa_sdk_plugins.attach_sanic_app_extensions(app)
    return app.request_hooks[0]


def _request(path, token=None, body=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return SimpleNamespace(path=path, headers=headers, json=body)


@pytest.fixture(autouse=True)
def _stub_introspection(monkeypatch):
    monkeypatch.setattr(rasa_sdk_plugins, "introspect_token_sync", lambda token: SUBS.get(token))


def _run(gate, request, _subs=None):
    return asyncio.run(gate(request))


SUBS = {"good": "user-1"}


def test_open_paths_need_no_token(gate):
    assert _run(gate, _request("/health"), SUBS) is None
    assert _run(gate, _request("/version"), SUBS) is None


def test_webhook_without_token_is_401(gate):
    assert _run(gate, _request("/webhook", body={"sender_id": "user-1"}), SUBS).status == 401


def test_webhook_with_inactive_token_is_401(gate):
    assert _run(gate, _request("/webhook", "bad", {"sender_id": "user-1"}), SUBS).status == 401


def test_webhook_sender_must_match_token_sub(gate):
    assert _run(gate, _request("/webhook", "good", {"sender_id": "user-2"}), SUBS).status == 403


def test_webhook_thread_sender_matches_sub(gate):
    body = {"sender_id": "user-1:thread:3", "tracker": {"sender_id": "user-1:thread:3"}}
    assert _run(gate, _request("/webhook", "good", body), SUBS) is None


def test_webhook_tracker_sender_mismatch_is_403(gate):
    body = {"sender_id": "user-1", "tracker": {"sender_id": "user-2"}}
    assert _run(gate, _request("/webhook", "good", body), SUBS).status == 403


def test_webhook_missing_sender_is_400(gate):
    assert _run(gate, _request("/webhook", "good", {}), SUBS).status == 400


def test_debug_requires_any_valid_token(gate):
    assert _run(gate, _request("/debug/fewshot-relevance"), SUBS).status == 401
    assert _run(gate, _request("/debug/fewshot-relevance", "good"), SUBS) is None


def test_sender_sub():
    assert keycloak_introspection.sender_sub("abc:thread:12") == "abc"
    assert keycloak_introspection.sender_sub("abc") == "abc"
