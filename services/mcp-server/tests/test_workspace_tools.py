"""The workspace tools: one MCP that acts as well as remembers.

These pin the contracts that matter: the caller's OWN token is what
reaches the workspace (never a service credential), an unattributable
caller cannot file work, an unconfigured deployment degrades instead of
guessing a host, and a caller in several workspaces is asked rather than
guessed for.
"""

from unittest.mock import MagicMock

import asyncio

import pytest

from app import workspace_tools


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


class TestCredentialKind:
    """An engine key is a valid Klio credential that the WORKSPACE cannot
    authenticate. Plane answers an unrecognised key with 403 (DRF sends no
    challenge), so the status code alone reads exactly like a permission
    denial -- which is how a whole debugging session went looking for a
    permissions bug that did not exist (2026-09-03)."""

    @pytest.mark.parametrize("key", ["klio_pat_abc", "klio_bot_abc", "plane_api_abc"])
    def test_workspace_keys_are_accepted(self, key):
        assert workspace_tools._is_workspace_key(key) is True

    @pytest.mark.parametrize("key", ["ag_live_abc", "sk-abc", "", "klio_key_abc"])
    def test_other_keys_are_rejected(self, key):
        assert workspace_tools._is_workspace_key(key) is False

    def test_engine_key_is_named_in_the_error(self):
        err = workspace_tools._not_a_workspace_key("ag_live_abc")

        assert err["error"] == "unauthorized"
        assert "engine key" in err["detail"]
        # The error must name the fix; the caller cannot see their own MCP
        # config from inside a tool result.
        assert "klio_pat_" in err["detail"]

    def test_unauthorized_is_not_reported_as_forbidden(self):
        """`forbidden` means "recognised you, refused you". Reporting an
        unusable credential that way sends people to the permissions UI."""
        assert (
            workspace_tools._not_a_workspace_key("ag_live_abc")["error"] != "forbidden"
        )


class TestWorkItemKeys:
    """The key a person reads off the board is accepted as an id."""

    @pytest.mark.parametrize("ref", ["YUMMY-8", "yummy-8", " KLIO-1234 ", "A1-2"])
    def test_board_keys_are_recognised(self, ref):
        assert workspace_tools._is_work_item_key(ref)

    @pytest.mark.parametrize(
        "ref",
        [
            "ff289b42-0307-4de8-a0ab-86fb76bf025b",
            "YUMMY",
            "8",
            "YUMMY-",
            "",
            None,
            12,
        ],
    )
    def test_everything_else_is_a_uuid_or_nonsense(self, ref):
        assert not workspace_tools._is_work_item_key(ref)


class TestPlainText:
    def test_markup_becomes_readable_lines(self):
        assert (
            workspace_tools._plain_text(
                "<p>Purge <strong>old</strong> data</p><ul><li>drop table</li>"
                "<li>vacuum</li></ul>"
            )
            == "Purge old data\ndrop table\nvacuum"
        )

    def test_entities_are_decoded_and_breaks_kept(self):
        assert workspace_tools._plain_text("<p>a &amp; b<br>c</p>") == "a & b\nc"

    def test_empty_markup_is_empty(self):
        assert workspace_tools._plain_text("<p></p>") == ""
        assert workspace_tools._plain_text(None) == ""


class TestShaping:
    def test_state_and_priority_come_back_as_words(self):
        item = workspace_tools._shape_work_item(
            {
                "id": "ff289b42-0307-4de8-a0ab-86fb76bf025b",
                "name": "Purge the old data",
                "state": "state-1",
                "priority": "priority-1",
                "description_html": "<p>Drop the 2019 rows.</p>",
            },
            states={"state-1": "In Progress"},
            priorities={"priority-1": "High"},
        )
        assert item["state"] == "In Progress"
        assert item["priority"] == "High"
        assert item["description"] == "Drop the 2019 rows."
        assert "description_stripped" not in item

    def test_unknown_ids_are_returned_as_they_are(self):
        item = workspace_tools._shape_work_item(
            {"id": "i", "state": "state-x", "priority": "priority-x"},
            states={},
            priorities={},
        )
        assert item["state"] == "state-x"
        assert item["priority"] == "priority-x"

    def test_the_stored_plain_text_wins_over_the_markup(self):
        item = workspace_tools._shape_work_item(
            {
                "id": "i",
                "description_stripped": "already plain",
                "description_html": "<p>markup</p>",
            },
            states={},
            priorities={},
        )
        assert item["description"] == "already plain"

    def test_a_comment_is_attributed_to_a_person(self):
        shaped = workspace_tools._shape_comment(
            {
                "id": "c1",
                "created_at": "2026-09-16T10:00:00Z",
                "created_by": "user-1",
                "comment_html": "<p>Ask Oren first.</p>",
            },
            {"user-1": "Abhishek"},
        )
        assert shaped == {
            "id": "c1",
            "created_at": "2026-09-16T10:00:00Z",
            "author": "Abhishek",
            "comment": "Ask Oren first.",
        }

    def test_an_unknown_author_keeps_their_id(self):
        shaped = workspace_tools._shape_comment(
            {"id": "c1", "created_by": "user-9", "comment_stripped": "hi"}, {}
        )
        assert shaped["author"] == "user-9"

    def test_member_rows_map_ids_to_names(self):
        names = workspace_tools._member_names(
            {
                "results": [
                    {"member": {"id": "u1", "display_name": "Oren"}, "role": 20},
                    {"member": "u2", "display_name": "Paul"},
                    {"member": {"id": "u3", "email": "omar@example.test"}},
                    {"nothing": True},
                ]
            }
        )
        assert names == {"u1": "Oren", "u2": "Paul", "u3": "omar@example.test"}


def _fake_call(responses, calls):
    """Stand in for `_call`, answering by path and recording the order."""

    async def _call(request, method, path, *, params=None, json_body=None):
        calls.append((method, path, params))
        for suffix, answer in responses.items():
            if path.endswith(suffix):
                return answer
        return {}, None

    return _call


ITEM = {
    "id": "ff289b42-0307-4de8-a0ab-86fb76bf025b",
    "name": "Purge the old data for YYT",
    "sequence_id": 8,
    "project_id": "6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
    "state": "state-1",
    "priority": "priority-1",
    "description_html": "<p>Drop everything before 2024.</p>",
}

LOOKUPS = {
    "/activities/": ({"results": []}, None),
    "/states/": ({"results": [{"id": "state-1", "name": "Todo"}]}, None),
    "/priorities/": (
        {
            "results": [
                {"id": "priority-1", "name": "Urgent"},
                {"id": "priority-2", "name": "High"},
            ]
        },
        None,
    ),
    "/members/": (
        {"results": [{"member": {"id": "user-1", "display_name": "Oren"}}]},
        None,
    ),
}


@pytest.mark.anyio
async def test_read_returns_the_body_and_the_thread(monkeypatch):
    calls = []
    responses = dict(LOOKUPS)
    responses["/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"] = (ITEM, None)
    responses["/comments/"] = (
        {
            "results": [
                {
                    "id": "c2",
                    "created_at": "2026-09-16T12:00:00Z",
                    "created_by": "user-1",
                    "comment_html": "<p>Second</p>",
                },
                {
                    "id": "c1",
                    "created_at": "2026-09-16T09:00:00Z",
                    "created_by": "user-1",
                    "comment_html": "<p>First</p>",
                },
            ]
        },
        None,
    )
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, calls))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["work_item"]["description"] == "Drop everything before 2024."
    assert result["work_item"]["state"] == "Todo"
    assert result["work_item"]["priority"] == "Urgent"
    # A thread reads oldest first.
    assert [c["id"] for c in result["comments"]] == ["c1", "c2"]
    assert result["comments"][0]["author"] == "Oren"
    assert calls[0] == (
        "GET",
        "/api/v1/workspaces/yummy/projects/6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/",
        None,
    )


@pytest.mark.anyio
async def test_a_board_key_needs_no_project_id(monkeypatch):
    calls = []
    responses = dict(LOOKUPS)
    responses["/work-items/YUMMY-8/"] = (ITEM, None)
    responses["/comments/"] = ({"results": []}, None)
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, calls))

    result = await workspace_tools.read_work_item(
        _Request(), slug="yummy", project_id=None, work_item_id="yummy-8"
    )

    assert calls[0][1] == "/api/v1/workspaces/yummy/work-items/YUMMY-8/"
    assert result["work_item"]["name"] == "Purge the old data for YYT"
    assert result["comments"] == []
    # Members are fetched regardless now: they name the people in the
    # activity feed, which exists even when nobody commented.
    assert any(path.endswith("/members/") for _, path, _ in calls)


@pytest.mark.anyio
async def test_reading_by_uuid_without_a_project_says_what_to_do(monkeypatch):
    monkeypatch.setattr(workspace_tools, "_call", _fake_call({}, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id=None,
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["error"] == "missing_argument"
    assert "YUMMY-8" in result["detail"]


@pytest.mark.anyio
async def test_read_needs_something_to_read(monkeypatch):
    monkeypatch.setattr(workspace_tools, "_call", _fake_call({}, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id=None,
    )

    assert result["error"] == "missing_argument"


@pytest.mark.anyio
async def test_a_refusal_from_the_workspace_is_passed_through(monkeypatch):
    refusal = {"error": "forbidden", "detail": "This token may not do that."}
    monkeypatch.setattr(
        workspace_tools,
        "_call",
        _fake_call(
            {"/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/": (None, refusal)}, []
        ),
    )

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result == refusal


@pytest.mark.anyio
async def test_an_unreadable_thread_does_not_cost_the_body(monkeypatch):
    responses = dict(LOOKUPS)
    responses["/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"] = (ITEM, None)
    responses["/comments/"] = (None, {"error": "unreachable", "detail": "no answer"})
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["work_item"]["name"] == "Purge the old data for YYT"
    # Silence would read as "no comments", which is a different fact.
    assert result["comments_error"]["error"] == "unreachable"
    assert "comments" not in result


@pytest.mark.anyio
async def test_unresolvable_lookups_still_return_the_item(monkeypatch):
    responses = {
        "/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/": (ITEM, None),
        "/states/": (None, {"error": "unreachable", "detail": "x"}),
        "/priorities/": (None, {"error": "unreachable", "detail": "x"}),
        "/comments/": ({"results": []}, None),
    }
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["work_item"]["state"] == "state-1"
    assert result["work_item"]["priority"] == "priority-1"


@pytest.mark.anyio
async def test_an_empty_answer_is_not_found(monkeypatch):
    monkeypatch.setattr(
        workspace_tools,
        "_call",
        _fake_call({"/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/": ({}, None)}, []),
    )

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["error"] == "not_found"


class TestIdValidation:
    @pytest.mark.parametrize(
        "value",
        [
            "ff289b42-0307-4de8-a0ab-86fb76bf025b",
            " ff289b42-0307-4de8-a0ab-86fb76bf025b ",
            "FF289B42-0307-4DE8-A0AB-86FB76BF025B",
        ],
    )
    def test_uuids_are_accepted(self, value):
        assert workspace_tools._is_uuid(value)

    @pytest.mark.parametrize(
        "value",
        ["../../projects", "not-a-uuid", "", None, 7, "ff289b42-0307-4de8-a0ab"],
    )
    def test_everything_else_is_not_an_id(self, value):
        assert not workspace_tools._is_uuid(value)


@pytest.mark.anyio
async def test_a_ref_that_is_not_an_id_never_reaches_the_workspace(monkeypatch):
    """Ids are interpolated into a path on an authenticated client."""
    calls = []
    monkeypatch.setattr(workspace_tools, "_call", _fake_call({}, calls))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="../../../workspaces/other/projects",
    )

    assert result["error"] == "invalid_argument"
    assert calls == []


@pytest.mark.anyio
async def test_a_project_id_that_is_not_an_id_is_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(workspace_tools, "_call", _fake_call({}, calls))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="../../..",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["error"] == "invalid_argument"
    assert calls == []


@pytest.mark.anyio
async def test_a_key_read_checks_the_project_the_workspace_returned(monkeypatch):
    """The key path takes its project from the payload; it is still a path."""
    calls = []
    monkeypatch.setattr(
        workspace_tools,
        "_call",
        _fake_call(
            {
                "/work-items/YUMMY-8/": (
                    {
                        "id": "ff289b42-0307-4de8-a0ab-86fb76bf025b",
                        "project_id": "../x",
                    },
                    None,
                )
            },
            calls,
        ),
    )

    result = await workspace_tools.read_work_item(
        _Request(), slug="yummy", project_id=None, work_item_id="YUMMY-8"
    )

    assert result["error"] == "invalid_argument"
    assert len(calls) == 1


@pytest.mark.anyio
async def test_the_three_lookups_go_together(monkeypatch):
    """The five lookups do not depend on each other, so the read pays for
    one round trip rather than five."""
    started = []
    finished = []
    finished_when_started = []

    async def _call(request, method, path, *, params=None, json_body=None):
        started.append(path)
        if path.endswith("/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"):
            return ITEM, None
        finished_when_started.append(len(finished))
        # Yield: a sequential implementation would complete each call before
        # the next one starts, and see a rising count above.
        await asyncio.sleep(0)
        finished.append(path)
        if path.endswith("/comments/"):
            return {"results": []}, None
        return {"results": []}, None

    monkeypatch.setattr(workspace_tools, "_call", _call)

    await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert sorted(path.rsplit("/", 2)[-2] for path in started[1:6]) == [
        "activities",
        "comments",
        "members",
        "priorities",
        "states",
    ]
    # Each of the five began while none of them had finished.
    assert finished_when_started == [0, 0, 0, 0, 0]


@pytest.mark.anyio
async def test_padded_ids_reach_the_workspace_clean(monkeypatch):
    """`_is_uuid` parses a stripped value, so the value that goes into the
    path has to be the stripped one too — a space there is a 404."""
    calls = []
    responses = dict(LOOKUPS)
    responses["/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"] = (ITEM, None)
    responses["/comments/"] = ({"results": []}, None)
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, calls))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="  6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f  ",
        work_item_id="  ff289b42-0307-4de8-a0ab-86fb76bf025b  ",
    )

    assert "work_item" in result
    assert all(" " not in path and "%20" not in path for _, path, _ in calls)
    assert calls[0] == (
        "GET",
        "/api/v1/workspaces/yummy/projects/6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f"
        "/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/",
        None,
    )


@pytest.mark.anyio
async def test_a_padded_project_from_the_workspace_is_cleaned_too(monkeypatch):
    calls = []
    responses = dict(LOOKUPS)
    responses["/work-items/YUMMY-8/"] = (
        {
            "id": "ff289b42-0307-4de8-a0ab-86fb76bf025b",
            "project_id": " 6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f ",
            "name": "Purge the old data for YYT",
        },
        None,
    )
    responses["/comments/"] = ({"results": []}, None)
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, calls))

    result = await workspace_tools.read_work_item(
        _Request(), slug="yummy", project_id=None, work_item_id="YUMMY-8"
    )

    assert "work_item" in result
    assert all(" " not in path for _, path, _ in calls)


class TestActivitySentences:
    PEOPLE = {"user-1": "Oren", "user-2": "Paul"}
    PRIORITIES = {"pri-1": "Medium", "pri-2": "High"}

    def describe(self, row):
        return workspace_tools._describe_activity(
            row, names=self.PRIORITIES, people=self.PEOPLE
        )

    def test_creation_reads_as_creation(self):
        assert (
            self.describe({"verb": "created", "field": None}) == "created the work item"
        )

    def test_a_priority_id_becomes_the_word_on_the_board(self):
        assert (
            self.describe(
                {
                    "verb": "updated",
                    "field": "priority",
                    "old_value": "pri-1",
                    "new_value": "pri-2",
                }
            )
            == "changed priority from Medium to High"
        )

    def test_state_values_are_already_words(self):
        assert (
            self.describe(
                {
                    "verb": "updated",
                    "field": "state",
                    "old_value": "Backlog",
                    "new_value": "In Progress",
                }
            )
            == "changed state from Backlog to In Progress"
        )

    def test_an_assignee_added_and_removed(self):
        assert (
            self.describe(
                {"verb": "updated", "field": "assignees", "new_value": "Paul"}
            )
            == "added Paul as assignee"
        )
        assert (
            self.describe(
                {"verb": "updated", "field": "assignees", "old_value": "Paul"}
            )
            == "removed Paul as assignee"
        )

    def test_setting_and_clearing_a_field(self):
        assert (
            self.describe(
                {"verb": "updated", "field": "target_date", "new_value": "2026-09-16"}
            )
            == "set target date to 2026-09-16"
        )
        assert (
            self.describe(
                {"verb": "updated", "field": "target_date", "old_value": "2026-09-16"}
            )
            == "cleared target date (was 2026-09-16)"
        )

    def test_the_description_diff_is_never_repeated_back(self):
        assert (
            self.describe(
                {
                    "verb": "updated",
                    "field": "description",
                    "old_value": "<p></p>",
                    "new_value": "<p class='editor-paragraph'>lots of markup</p>",
                }
            )
            == "updated the description"
        )

    def test_a_row_that_is_not_a_row_is_dropped(self):
        assert workspace_tools._shape_activity("nonsense", names={}, people={}) is None

    def test_the_actor_is_named(self):
        shaped = workspace_tools._shape_activity(
            {
                "created_at": "2026-09-16T17:32:00Z",
                "actor": "user-1",
                "verb": "updated",
                "field": "assignees",
                "new_value": "Abhishek",
            },
            names={},
            people=self.PEOPLE,
        )
        assert shaped == {
            "at": "2026-09-16T17:32:00Z",
            "who": "Oren",
            "field": "assignees",
            "change": "added Abhishek as assignee",
        }


@pytest.mark.anyio
async def test_read_returns_what_happened_to_the_item(monkeypatch):
    """The case this was built for: no comments, four activity rows."""
    responses = dict(LOOKUPS)
    responses["/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"] = (ITEM, None)
    responses["/comments/"] = ({"results": []}, None)
    responses["/activities/"] = (
        {
            "results": [
                {
                    "created_at": "2026-09-16T19:27:00Z",
                    "actor": "user-1",
                    "verb": "updated",
                    "field": "priority",
                    "old_value": "priority-2",
                    "new_value": "priority-1",
                },
                {
                    "created_at": "2026-09-10T19:38:00Z",
                    "actor": "user-1",
                    "verb": "created",
                    "field": None,
                },
            ]
        },
        None,
    )
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["comments"] == []
    # Oldest first, and the priority id resolved to the word on the board.
    assert [row["change"] for row in result["activity"]] == [
        "created the work item",
        "changed priority from High to Urgent",
    ]
    assert {row["who"] for row in result["activity"]} == {"Oren"}


@pytest.mark.anyio
async def test_an_unreadable_activity_feed_does_not_cost_the_body(monkeypatch):
    responses = dict(LOOKUPS)
    responses["/issues/ff289b42-0307-4de8-a0ab-86fb76bf025b/"] = (ITEM, None)
    responses["/comments/"] = ({"results": []}, None)
    responses["/activities/"] = (None, {"error": "unreachable", "detail": "no answer"})
    monkeypatch.setattr(workspace_tools, "_call", _fake_call(responses, []))

    result = await workspace_tools.read_work_item(
        _Request(),
        slug="yummy",
        project_id="6b3a1f2c-1111-4c2d-9f77-0a1b2c3d4e5f",
        work_item_id="ff289b42-0307-4de8-a0ab-86fb76bf025b",
    )

    assert result["work_item"]["name"] == "Purge the old data for YYT"
    assert result["activity_error"]["error"] == "unreachable"
    assert "activity" not in result


def test_assignees_and_author_are_people_not_ids():
    item = workspace_tools._shape_work_item(
        {
            "id": "i",
            "assignees": ["user-1", "user-9"],
            "created_by": "user-2",
        },
        states={},
        priorities={},
        people={"user-1": "Oren", "user-2": "Paul"},
    )
    assert item["assignees"] == ["Oren", "user-9"]
    assert item["created_by"] == "Paul"


def test_an_unresolvable_id_in_the_activity_is_left_as_it_is():
    """Better a raw id than a sentence that invents a value."""
    shaped = workspace_tools._shape_activity(
        {
            "created_at": "2026-09-16T19:27:00Z",
            "verb": "updated",
            "field": "priority",
            "old_value": "priority-gone",
            "new_value": "priority-1",
        },
        names={"priority-1": "Urgent"},
        people={},
    )
    assert shaped["change"] == "changed priority from priority-gone to Urgent"
