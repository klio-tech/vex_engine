"""The workspace tools: one MCP that acts as well as remembers.

These pin the contracts that matter: the caller's OWN token is what
reaches the workspace (never a service credential), an unattributable
caller cannot file work, an unconfigured deployment degrades instead of
guessing a host, and a caller in several workspaces is asked rather than
guessed for.
"""

from unittest.mock import MagicMock

import app.workspace_tools as workspace_tools
import pytest


class _Request:
    def __init__(self, key="plane_api_deadbeef"):
        self.headers = {"X-Vex-Key": key} if key else {}


def _workspaces(*rows):
    return list(rows)


def _ws(slug, personal=False):
    return {
        "workspace_id": f"ws-{slug}",
        "slug": slug,
        "is_personal": personal,
        "owner_user_id": "user-1",
    }


def test_unconfigured_deployment_degrades(monkeypatch):
    monkeypatch.delenv(workspace_tools.WORKSPACE_API_URL_ENV, raising=False)
    result = workspace_tools._unconfigured()
    assert result["error"] == "not_configured"
    # Memory must be named as unaffected: an agent that reads this should
    # not conclude Klio is down.
    assert "memory" in result["detail"].lower()


def test_unattributable_caller_cannot_file_work(monkeypatch):
    monkeypatch.setattr(
        workspace_tools, "resolve_request_user_id", lambda request: None
    )
    slug, error = workspace_tools.resolve_workspace_slug(_Request(), None)
    assert slug is None
    assert error["error"] == "unauthorized"


def test_single_org_workspace_is_resolved_without_being_named(monkeypatch):
    monkeypatch.setattr(
        workspace_tools, "resolve_request_user_id", lambda request: "user-1"
    )
    monkeypatch.setattr(workspace_tools, "workspace_engine", lambda: MagicMock())
    monkeypatch.setattr(
        workspace_tools,
        "member_workspaces",
        lambda conn, user_id: _workspaces(_ws("acme"), _ws("alice", personal=True)),
    )
    slug, error = workspace_tools.resolve_workspace_slug(_Request(), None)
    assert error is None
    # The personal workspace is never the default target for work items.
    assert slug == "acme"


def test_several_workspaces_are_asked_about_not_guessed(monkeypatch):
    monkeypatch.setattr(
        workspace_tools, "resolve_request_user_id", lambda request: "user-1"
    )
    monkeypatch.setattr(workspace_tools, "workspace_engine", lambda: MagicMock())
    monkeypatch.setattr(
        workspace_tools,
        "member_workspaces",
        lambda conn, user_id: _workspaces(_ws("acme"), _ws("globex")),
    )
    slug, error = workspace_tools.resolve_workspace_slug(_Request(), None)
    assert slug is None
    assert error["error"] == "ambiguous_workspace"
    assert set(error["workspaces"]) == {"acme", "globex"}


def test_a_workspace_you_are_not_in_is_not_found(monkeypatch):
    monkeypatch.setattr(
        workspace_tools, "resolve_request_user_id", lambda request: "user-1"
    )
    monkeypatch.setattr(workspace_tools, "workspace_engine", lambda: MagicMock())
    monkeypatch.setattr(
        workspace_tools,
        "member_workspaces",
        lambda conn, user_id: _workspaces(_ws("acme")),
    )
    slug, error = workspace_tools.resolve_workspace_slug(
        _Request(), "someone-elses-org"
    )
    assert slug is None
    assert error["error"] == "not_found"


@pytest.mark.anyio
async def test_the_callers_own_token_is_what_reaches_the_workspace(monkeypatch):
    """The whole point: actions run as the person, not as a service account."""
    monkeypatch.setenv(
        workspace_tools.WORKSPACE_API_URL_ENV, "https://work.example.test"
    )

    captured = {}

    class _Response:
        status_code = 200
        content = b"{}"

        @staticmethod
        def json():
            return {"ok": True}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, url, params=None, json=None, headers=None):
            captured["headers"] = headers
            captured["url"] = url
            return _Response()

    monkeypatch.setattr(workspace_tools.httpx, "AsyncClient", lambda **kw: _Client())

    payload, error = await workspace_tools._call(
        _Request(key="plane_api_the_callers_token"), "GET", "/api/v1/x/"
    )
    assert error is None
    assert payload == {"ok": True}
    assert captured["headers"]["X-Api-Key"] == "plane_api_the_callers_token"
    assert captured["url"].startswith("https://work.example.test")


@pytest.mark.anyio
async def test_workspace_refusals_are_passed_through_not_masked(monkeypatch):
    monkeypatch.setenv(
        workspace_tools.WORKSPACE_API_URL_ENV, "https://work.example.test"
    )

    class _Response:
        status_code = 403
        content = b""

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, *a, **kw):
            return _Response()

    monkeypatch.setattr(workspace_tools.httpx, "AsyncClient", lambda **kw: _Client())
    payload, error = await workspace_tools._call(_Request(), "GET", "/api/v1/x/")
    assert payload is None
    # The workspace is the authority on authorization; we report its answer.
    assert error["error"] == "forbidden"


def test_slim_drops_empty_fields_but_keeps_real_ones():
    row = {"id": "1", "name": "Ship it", "priority": None, "description_stripped": ""}
    out = workspace_tools._slim(row, ("id", "name", "priority", "description_stripped"))
    assert out == {"id": "1", "name": "Ship it"}


def test_rows_handles_both_paginated_and_bare_payloads():
    assert workspace_tools._rows({"results": [1, 2]}) == [1, 2]
    assert workspace_tools._rows([3]) == [3]
    assert workspace_tools._rows(None) == []


@pytest.fixture
def anyio_backend():
    return "asyncio"
