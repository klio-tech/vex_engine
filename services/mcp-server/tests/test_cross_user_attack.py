"""Adversarial tests at the TOOL surface. Companion to
``shared/tests/test_user_silo_isolation.py``, which attacks the SQL.

The storage layer can be perfectly scoped and still leak if a tool hands it the
wrong principal. So these tests attack the seam between a request and a
``user_id``: forge the one header a client controls, call ``share`` on someone
else's memory, grant yourself into a project you do not administer, and ask for
a member list from outside the boundary.

Everything below is a MagicMock request plus a fake DB — no Postgres. What is
under test here is *which arguments the tool derives and forwards*, which is
exactly the class of bug a database cannot catch.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import mcp_app
from app.identity import resolve_request_user_id

PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _request(headers=None, oauth_principal=None):
    req = MagicMock()
    req.state = SimpleNamespace()
    if oauth_principal is not None:
        req.state.oauth_principal = oauth_principal
    req.headers = headers or {}
    return req


def _ctx(request):
    ctx = MagicMock()
    ctx.request_context.request = request
    return ctx


def _principal(user_id):
    from app.oauth import OAuthPrincipal

    return OAuthPrincipal(
        org_id="acme",
        account_slug="acme",
        user_id=user_id,
        client_id="cli",
        plan="free",
        plan_overrides=None,
    )


# ---------------------------------------------------------------------------
# ATTACK 1 — forge X-Vex-Agent to become someone else
# ---------------------------------------------------------------------------


def test_x_vex_agent_cannot_make_dev_into_priya():
    """The header is the ONE identity input a client fully controls.

    If it could set ``user_id``, reading a teammate's private memories would be
    a one-line curl. Dev sends every plausible spelling, including a raw user
    id, and still resolves to himself.
    """
    forged = _request(
        headers={
            "X-Vex-Agent": PRIYA,
            "x-vex-agent": PRIYA,
            "X-Vex-User": PRIYA,
            "X-Vex-User-Id": PRIYA,
        },
        oauth_principal=_principal(DEV),
    )
    assert resolve_request_user_id(forged) == DEV


def test_a_forged_header_alone_resolves_to_nobody():
    """With no credential at all, the header buys not even an identity."""
    forged = _request(headers={"X-Vex-Agent": PRIYA})
    with patch("app.auth.verify_api_key", side_effect=Exception("no key")):
        assert resolve_request_user_id(forged) is None


@pytest.mark.asyncio
async def test_recall_forwards_the_credentialed_user_not_the_header():
    """End-to-end at the tool: Dev declares Priya's agent id and the value that
    reaches the SQL layer is still Dev's."""
    retrieve = AsyncMock(return_value=[])
    request = _request(headers={"X-Vex-Agent": PRIYA}, oauth_principal=_principal(DEV))

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("acme", PRIYA, "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve),
        patch("app.mcp_app.record_usage"),
        patch("app.mcp_app.check_memory_quota", return_value=False),
        patch("app.mcp_app.log_recall_event"),
    ):
        await mcp_app._perform_recall(
            _ctx(request), query="anything", scope="org", space=None, project=None
        )

    assert retrieve.await_args.kwargs["user_id"] == DEV
    # agent_id still carries the declared label — it is a label, not a person.
    assert retrieve.await_args.kwargs["agent_id"] == PRIYA


@pytest.mark.asyncio
async def test_private_recall_by_an_unattributed_caller_is_refused():
    """It must not silently widen to the org brain."""
    retrieve = AsyncMock(return_value=[])
    request = _request()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("acme", "agent-x", "free", None),
        ),
        patch("app.mcp_app.resolve_request_user_id", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve),
    ):
        result = await mcp_app._perform_recall(
            _ctx(request), query="q", scope="private", space=None, project=None
        )

    assert result["error"] == "unattributed_caller"
    assert result["memories"] == []
    retrieve.assert_not_awaited()


# ---------------------------------------------------------------------------
# ATTACK 2 — share a memory you do not own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_share_forwards_the_caller_as_owner_so_promotion_can_refuse():
    """``share`` must never let the caller name the owner.

    The refusal itself lives in ``promote_memory`` (whose SELECT carries
    ``user_id = :user_id``); what this pins is that the tool hands it the
    RESOLVED caller. A tool that passed the row's own user_id, or None, would
    turn a correct storage layer into an open door.
    """
    promote = MagicMock(return_value={"shared": False, "error": "not_found"})
    request = _request(headers={"X-Vex-Agent": PRIYA}, oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.promote_memory", promote),
    ):
        result = await mcp_app._perform_share(
            _ctx(request), memory_id="00000000-0000-0000-0000-0000000000ff", to="org"
        )

    assert promote.call_args.kwargs["user_id"] == DEV
    assert result == {"shared": False, "error": "not_found"}


@pytest.mark.asyncio
async def test_an_unattributed_caller_cannot_share_anything():
    promote = MagicMock()
    request = _request()

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.resolve_request_user_id", return_value=None),
        patch("app.mcp_app.promote_memory", promote),
    ):
        result = await mcp_app._perform_share(_ctx(request), memory_id="x", to="org")

    assert result["error"] == "unattributed_caller"
    promote.assert_not_called()


@pytest.mark.asyncio
async def test_sharing_to_a_project_requires_membership_of_it():
    """Pushing a memory INTO a boundary you are not inside is a write across a
    line you cannot read across."""
    project_id = "aaaaaaaa-0000-0000-0000-000000000001"
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = SimpleNamespace(
        project_id=project_id
    )
    promote = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=db)),
        patch("app.mcp_app.projects.can_write", return_value=False),
        patch("app.mcp_app.promote_memory", promote),
    ):
        result = await mcp_app._perform_share(
            _ctx(request), memory_id="m1", to="project"
        )

    assert result["error"] == "not_a_member"
    promote.assert_not_called()


# ---------------------------------------------------------------------------
# ATTACK 3 — project_grant, the tool a prompt injection wants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_admin_cannot_grant_even_to_themselves():
    """Belonging to the org is not standing. Neither is belonging to the
    project. Only administering it is.

    ``is_member`` is patched True: this caller DOES belong to the project
    (that is the whole point of the test — membership without admin is not
    enough), so the refusal must be the informative ``forbidden``, not the
    existence-hiding ``not_found`` a genuine non-member gets (see
    ``test_a_non_member_cannot_grant_even_if_they_can_name_the_project``).
    """
    add_member = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.find_project_id", return_value="proj-1"),
        patch("app.mcp_app.projects.is_member", return_value=True),
        patch("app.mcp_app.projects.can_manage", return_value=False),
        patch("app.mcp_app.projects.member_role", return_value="write"),
        patch("app.mcp_app.projects.add_member", add_member),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=DEV),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(request), project="apollo", user="dev@oppla.ai", role="admin"
        )

    assert result["error"] == "forbidden"
    add_member.assert_not_called()


@pytest.mark.asyncio
async def test_a_non_member_cannot_grant_even_if_they_can_name_the_project():
    """A caller who is not even a member gets `not_found`, not `forbidden`.

    `forbidden` would confirm the project's name is real to someone standing
    entirely outside it — the same existence-oracle problem `not_found` on
    `project_members` already guards against (see
    ``test_a_non_member_gets_not_found_for_a_project_member_list``).
    ``is_admin`` is deliberately left unmocked / unreachable here: the
    membership gate must short-circuit before administration is even asked
    about.
    """
    add_member = MagicMock()
    is_admin = MagicMock(return_value=False)
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.find_project_id", return_value="proj-1"),
        patch("app.mcp_app.projects.is_member", return_value=False),
        patch("app.mcp_app.projects.is_admin", is_admin),
        patch("app.mcp_app.projects.add_member", add_member),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=DEV),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(request), project="apollo", user="dev@oppla.ai", role="admin"
        )

    assert result["error"] == "not_found"
    assert "unknown project" in result["detail"]
    add_member.assert_not_called()


@pytest.mark.asyncio
async def test_a_raw_user_id_is_refused_where_an_email_belongs():
    """Injected text can name a UUID. It cannot name an org membership.

    Accepting a raw id would let a malicious repo file say "grant
    <uuid>" and skip the directory check entirely.
    """
    add_member = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.projects.is_admin", return_value=True),
        patch("app.mcp_app.projects.add_member", add_member),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(request), project="apollo", user=PRIYA, role="member"
        )

    assert result["error"] == "email_required"
    add_member.assert_not_called()


@pytest.mark.asyncio
async def test_an_email_outside_the_org_cannot_be_granted():
    """The one gate that keeps an ACL tool from becoming a public share link."""
    add_member = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.find_project_id", return_value="proj-1"),
        patch("app.mcp_app.projects.is_member", return_value=True),
        patch("app.mcp_app.projects.can_manage", return_value=True),
        patch("app.mcp_app.projects.member_role", return_value="admin"),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=None),
        patch("app.mcp_app.projects.add_member", add_member),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(request), project="apollo", user="attacker@evil.example", role="admin"
        )

    assert result["error"] == "not_in_org"
    add_member.assert_not_called()


@pytest.mark.asyncio
async def test_a_successful_grant_is_audited_with_who_what_and_surface():
    """The spec's second constraint on this tool. An ACL change nobody can
    reconstruct afterwards is the part that hurts during an incident."""
    audit = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.find_project_id", return_value="proj-1"),
        patch("app.mcp_app.projects.is_member", return_value=True),
        patch("app.mcp_app.projects.can_manage", return_value=True),
        patch("app.mcp_app.projects.member_role", side_effect=["admin", None]),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=PRIYA),
        patch("app.mcp_app.projects.add_member", return_value=True),
        patch("app.mcp_app.projects.record_grant_audit", audit),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(request), project="apollo", user="priya@oppla.ai", role="member"
        )

    assert result["granted"] is True
    kwargs = audit.call_args.kwargs
    assert kwargs["granted_by"] == DEV
    assert kwargs["granted_to"] == PRIYA
    assert kwargs["granted_to_email"] == "priya@oppla.ai"
    assert kwargs["project_id"] == "proj-1"
    assert kwargs["role"] == "write"
    assert kwargs["surface"] == "mcp"


@pytest.mark.asyncio
async def test_the_directory_fails_closed_without_supabase():
    """No directory configured is not permission to trust caller input."""
    from app import directory

    with patch.object(directory, "SUPABASE_DATABASE_URL", None):
        assert directory.resolve_org_member_by_email("acme", "priya@oppla.ai") is None


# ---------------------------------------------------------------------------
# ATTACK 4 — enumerate a project you are not in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_member_gets_not_found_for_a_project_member_list():
    """Who has access is itself part of what the boundary protects, and
    'forbidden' would confirm the project exists."""
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.find_project_id", return_value="proj-1"),
        patch("app.mcp_app.projects.is_member", return_value=False),
    ):
        result = await mcp_app._perform_project_members(_ctx(request), project="apollo")

    assert result["error"] == "not_found"
    assert "members" not in result


@pytest.mark.asyncio
async def test_a_project_someone_else_administers_cannot_be_claimed():
    """project_create must not become a way to adopt an existing team's project."""
    add_member = MagicMock()
    request = _request(oauth_principal=_principal(DEV))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-x")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.brain.ensure_project", return_value="proj-1"),
        patch("app.mcp_app.projects.member_count", return_value=2),
        patch("app.mcp_app.projects.is_member", return_value=False),
        patch("app.mcp_app.projects.add_member", add_member),
    ):
        result = await mcp_app._perform_project_create(
            _ctx(request),
            name="apollo",
            git_remote="git@example.com:acme/apollo.git",
            repo_root_path=None,
        )

    assert result["error"] == "forbidden"
    add_member.assert_not_called()


# ---------------------------------------------------------------------------
# ATTACK 5 — write into a project boundary you are not inside
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_member_write_falls_back_to_org_not_project_scope():
    """X-Vex-Project is a client-declared header. It must file a memory (a
    relevance hint) without ever fencing it (a boundary) for a non-member."""
    db = MagicMock()

    with patch("app.mcp_app.projects.can_write", return_value=False):
        scope = mcp_app._explicit_write_scope(
            db,
            requested_scope="org",
            org_id="acme",
            user_id=DEV,
            project_id="proj-1",
        )
    assert scope == "org"

    with patch("app.mcp_app.projects.can_write", return_value=True):
        scope = mcp_app._explicit_write_scope(
            db,
            requested_scope="org",
            org_id="acme",
            user_id=DEV,
            project_id="proj-1",
        )
    assert scope == "project"


def test_an_unattributed_hook_capture_never_writes_an_ownerless_private_row():
    """The most dangerous shape in the system: a private row nobody owns is
    unreachable by everyone including its author, so routing there on a failed
    identity lookup would destroy data while reporting success."""
    from app import capture

    request = _request()
    key_info = SimpleNamespace(org_id="acme", key_id="k1", user_id=None)

    scope, user_id = capture._hook_write_scope(request, key_info, "org")
    assert (scope, user_id) == ("org", None)

    key_info_attributed = SimpleNamespace(org_id="acme", key_id="k1", user_id=PRIYA)
    scope, user_id = capture._hook_write_scope(request, key_info_attributed, "org")
    assert (scope, user_id) == ("private", PRIYA)
