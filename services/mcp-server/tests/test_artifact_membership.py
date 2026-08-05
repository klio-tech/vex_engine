"""Do the artifact tools respect project membership? Real Postgres.

``test_artifact_project_scope.py`` proves the *card* an artifact writes stops
at the project: a non-member's ``recall`` cannot see it. That is only half the
surface. The artifact tools also take an ``artifact_id`` or a ``project`` name
DIRECTLY, and an id is not a secret — it appears in a card, in a prior tool
result, in a log line, and is guessable in principle. Every one of those
entrypoints used to skip ``projects.is_member`` entirely:

* ``artifact_get`` read any artifact in the org by id, project or not;
* ``artifact_put`` filed new bytes INTO a project the caller was not a member
  of, and retired a project artifact through ``supersedes``;
* ``forget`` retracted a project artifact by id.

The rule under test is the product decision, unchanged: project scope is
**strict**, with **no org-admin override**. And refusals are shaped so they
teach nothing: a caller who is not a member gets exactly what they would get
for an id that was never issued (``not_found``) or a project that does not
exist (``unknown_project``). An error that distinguished "exists but forbidden"
from "does not exist" would be an existence oracle over precisely the rows the
project boundary exists to hide.

Real Postgres because the boundary is ``project_members`` and a mocked session
cannot prove a join is effective. Every adversarial assertion is paired with a
positive control in the same test: the member performs the identical call and
must succeed, so a refusal can never be an artifact of a broken fixture.
"""

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app import mcp_app
from app.mcp_app import (
    _perform_artifact_get,
    _perform_artifact_put,
    _perform_forget,
)
from shared import projects
from shared.testing import database_url, requires_db

pytestmark = requires_db("the artifact project-membership boundary tests")

DB_URL = database_url()

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM

ORG = "acme"
# Two people in one org. PRIYA is a member of the project; DEV is not.
PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"

MB = 1024 * 1024


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in orgs:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM artifacts WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str, name: str = "apollo") -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, :name)
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
        },
    )
    db.commit()
    return project_id


def _artifact(
    db, *, org_id: str, project_id: str | None, title: str = "launch checklist"
) -> str:
    """An active, inline artifact. Inline so ``artifact_get`` needs no S3."""
    artifact_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO artifacts (
                id, org_id, agent_id, title, summary, kind, mime_type,
                size_bytes, checksum, inline_text, version, status, project_id
            ) VALUES (
                :id, :org_id, 'agent-1', :title, 'a summary', 'document',
                'text/plain', 6, :checksum, 'secret', 1, 'active', :project_id
            )
            """
        ),
        {
            "id": artifact_id,
            "org_id": org_id,
            "title": title,
            "checksum": uuid.uuid4().hex,
            "project_id": project_id,
        },
    )
    db.commit()
    return artifact_id


class _NoCloseSession:
    """The tool bodies own their session and ``close()`` it in ``finally``.

    The fixture's connection has to outlive the call so the test can assert
    against the same rows, so closing is the one thing this proxy intercepts.
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - deliberately inert
        pass


def _ctx(project_header: str | None = None):
    headers = {"x-vex-agent": "agent-1"}
    if project_header is not None:
        headers["X-Vex-Project"] = project_header
    request = SimpleNamespace(headers=headers, state=SimpleNamespace())
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


@contextmanager
def _as(db, user_id: str | None, *, org_id: str):
    """Run a tool body as ``user_id`` in ``org_id`` against the real database.

    Only auth and the session factory are seams. ``projects.is_member`` — the
    thing under test — is emphatically NOT patched: it runs its real SQL
    against the real ``project_members`` rows the test inserted.
    """
    plan = SimpleNamespace(
        artifact_max_file_bytes=10 * MB,
        artifact_storage_bytes=100 * MB,
    )
    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=(org_id, "agent-1", "free", None),
        ),
        patch("app.mcp_app.resolve_principal", return_value=(org_id, "agent-1")),
        patch("app.mcp_app.resolve_request_user_id", return_value=user_id),
        patch("app.mcp_app.get_plan_config", return_value=plan),
        patch("app.mcp_app.SessionLocal", lambda: _NoCloseSession(db)),
        patch(
            "shared.memory.embed_text",
            new=AsyncMock(return_value=FIXED_EMBEDDING),
        ),
    ):
        yield


def _member(db, *, org_id: str, project_id: str, user_id: str) -> None:
    assert projects.add_member(
        db,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        granted_by=None,
    )
    db.commit()


# ---------------------------------------------------------------------------
# ATTACK 1 — read a project artifact by id as a non-member
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_member_cannot_artifact_get_a_project_artifact(db):
    org = _org(db)
    project = _project(db, org_id=org)
    _member(db, org_id=org, project_id=project, user_id=PRIYA)
    artifact_id = _artifact(db, org_id=org, project_id=project)

    with _as(db, DEV, org_id=org):
        refused = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)

    assert refused == {"error": "not_found"}, (
        "a non-member must get the same answer as for an id that was never "
        f"issued; got {refused}"
    )

    # Positive control: the member reads the identical artifact.
    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)

    assert allowed.get("text") == "secret", (
        f"positive control failed: a member must still read the bytes; got {allowed}"
    )


@pytest.mark.asyncio
async def test_org_artifact_is_still_readable_by_anyone_in_the_org(db):
    """The gate is project membership, not a new org-wide restriction."""
    org = _org(db)
    artifact_id = _artifact(db, org_id=org, project_id=None)

    with _as(db, DEV, org_id=org):
        result = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)

    assert result.get("text") == "secret", (
        f"an unfiled artifact must stay org-readable; got {result}"
    )


@pytest.mark.asyncio
async def test_unattributed_caller_cannot_read_a_project_artifact(db):
    """``user_id=None`` is not a member of anything — it is not a wildcard."""
    org = _org(db)
    project = _project(db, org_id=org)
    _member(db, org_id=org, project_id=project, user_id=PRIYA)
    artifact_id = _artifact(db, org_id=org, project_id=project)

    with _as(db, None, org_id=org):
        refused = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)

    assert refused == {"error": "not_found"}, (
        f"an unattributable caller must be refused; got {refused}"
    )

    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)
    assert allowed.get("text") == "secret", "positive control failed"


# ---------------------------------------------------------------------------
# ATTACK 2 — file bytes INTO a project you are not a member of
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_member_cannot_artifact_put_into_a_project(db):
    """The worst arm: an unconditional write into someone else's project."""
    org = _org(db)
    project = _project(db, org_id=org, name="apollo")
    _member(db, org_id=org, project_id=project, user_id=PRIYA)

    with _as(db, DEV, org_id=org):
        refused = await _perform_artifact_put(
            _ctx(),
            title="t",
            summary="s",
            kind="document",
            content="intruder",
            project="apollo",
        )

    assert refused["stored"] is False
    assert refused["error"] == "unknown_project", (
        "a project the caller cannot see must be indistinguishable from one "
        f"that does not exist; got {refused}"
    )
    landed = db.execute(
        text("SELECT COUNT(*) FROM artifacts WHERE org_id = :o AND project_id = :p"),
        {"o": org, "p": project},
    ).scalar()
    assert landed == 0, "the refused write still created an artifact row"

    # Positive control: the member files into the same project and it lands.
    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_put(
            _ctx(),
            title="t",
            summary="s",
            kind="document",
            content="legitimate",
            project="apollo",
        )
    assert allowed["stored"] is True, f"positive control failed; got {allowed}"
    landed = db.execute(
        text("SELECT COUNT(*) FROM artifacts WHERE org_id = :o AND project_id = :p"),
        {"o": org, "p": project},
    ).scalar()
    assert landed == 1


@pytest.mark.asyncio
async def test_non_member_cannot_file_via_the_project_header(db):
    """``X-Vex-Project`` is client-supplied too, and takes the same gate.

    The header path fails OPEN for an unresolvable identifier (org-wide), and
    that is deliberate. It must not fail open HERE: filing org-wide an artifact
    the caller asked to file into a project lands the write wider than the
    caller was told, which is the defect, not the remedy.
    """
    org = _org(db)
    project = _project(db, org_id=org, name="apollo")
    _member(db, org_id=org, project_id=project, user_id=PRIYA)

    with _as(db, DEV, org_id=org):
        refused = await _perform_artifact_put(
            _ctx(project_header="apollo"),
            title="t",
            summary="s",
            kind="document",
            content="intruder-via-header",
        )

    assert refused["stored"] is False, f"header path must refuse; got {refused}"
    assert refused["error"] == "unknown_project"
    total = db.execute(
        text("SELECT COUNT(*) FROM artifacts WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert total == 0, "the refused header write still created an artifact row"

    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_put(
            _ctx(project_header="apollo"),
            title="t",
            summary="s",
            kind="document",
            content="legitimate-via-header",
        )
    assert allowed["stored"] is True, f"positive control failed; got {allowed}"


@pytest.mark.asyncio
async def test_non_member_cannot_confirm_a_project_upload(db):
    """The confirm branch takes a bare ``artifact_id`` and trusts its row.

    The pending row carries the project chosen when the upload was REQUESTED.
    Ids travel — in tool results, in logs — so a second caller holding one must
    not be able to complete a stranger's upload into a project they have no
    standing in. Object storage is the only seam here; the row, the project and
    the membership table are all real.
    """
    org = _org(db)
    project = _project(db, org_id=org)
    _member(db, org_id=org, project_id=project, user_id=PRIYA)

    pending_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO artifacts (
                id, org_id, agent_id, title, summary, kind, mime_type,
                version, status, project_id
            ) VALUES (
                :id, :org_id, 'agent-1', 't', 's', 'document', 'text/plain',
                1, 'pending', :project_id
            )
            """
        ),
        {"id": pending_id, "org_id": org, "project_id": project},
    )
    db.commit()

    storage = SimpleNamespace(
        head_size=lambda *a, **k: 5,
        checksum=lambda *a, **k: uuid.uuid4().hex,
        object_key=lambda *a, **k: "k",
        storage_uri=lambda *a, **k: "s3://b/k",
        get_artifact_s3_client=lambda: object(),
    )

    with _as(db, DEV, org_id=org), patch("app.mcp_app.artifact_storage", storage):
        refused = await _perform_artifact_put(
            _ctx(), kind="document", artifact_id=pending_id
        )

    assert refused["stored"] is False, f"expected a refusal; got {refused}"
    assert refused["error"] == "not_found"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": pending_id}
    ).scalar()
    assert status == "pending", "a non-member activated a project upload"

    # Positive control: the member confirms the identical pending row.
    with _as(db, PRIYA, org_id=org), patch("app.mcp_app.artifact_storage", storage):
        allowed = await _perform_artifact_put(
            _ctx(), kind="document", artifact_id=pending_id
        )
    assert allowed["stored"] is True, f"positive control failed; got {allowed}"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": pending_id}
    ).scalar()
    assert status == "active"


@pytest.mark.asyncio
async def test_non_member_cannot_supersede_a_project_artifact(db):
    """``supersedes`` retires a row. Retiring is a write across the boundary."""
    org = _org(db)
    project = _project(db, org_id=org)
    _member(db, org_id=org, project_id=project, user_id=PRIYA)
    victim = _artifact(db, org_id=org, project_id=project)

    with _as(db, DEV, org_id=org):
        refused = await _perform_artifact_put(
            _ctx(),
            title="t",
            summary="s",
            kind="document",
            content="replacement",
            supersedes=victim,
        )

    assert refused["stored"] is False, f"expected a refusal; got {refused}"
    assert refused["error"] == "not_found"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": victim}
    ).scalar()
    assert status == "active", "a non-member retired a project artifact"

    # Positive control: the member supersedes the same artifact.
    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_put(
            _ctx(),
            title="t",
            summary="s",
            kind="document",
            content="replacement",
            supersedes=victim,
        )
    assert allowed["stored"] is True, f"positive control failed; got {allowed}"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": victim}
    ).scalar()
    assert status == "superseded"


# ---------------------------------------------------------------------------
# ATTACK 3 — forget a project artifact by id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_member_cannot_forget_a_project_artifact(db):
    org = _org(db)
    project = _project(db, org_id=org)
    _member(db, org_id=org, project_id=project, user_id=PRIYA)
    artifact_id = _artifact(db, org_id=org, project_id=project)

    with _as(db, DEV, org_id=org):
        refused = await _perform_forget(_ctx(), memory_id=artifact_id)

    assert refused.get("forgotten") is False, f"expected a refusal; got {refused}"
    assert refused.get("error") == "not_found"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": artifact_id}
    ).scalar()
    assert status == "active", "a non-member retracted a project artifact"

    # Positive control: the member forgets the identical artifact.
    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_forget(_ctx(), memory_id=artifact_id)
    assert allowed.get("forgotten") is True, f"positive control failed; got {allowed}"
    status = db.execute(
        text("SELECT status FROM artifacts WHERE id = :i"), {"i": artifact_id}
    ).scalar()
    assert status == "retracted"


@pytest.mark.asyncio
async def test_org_artifact_is_still_forgettable(db):
    """Unfiled artifacts keep their existing org-wide behaviour."""
    org = _org(db)
    artifact_id = _artifact(db, org_id=org, project_id=None)

    with _as(db, DEV, org_id=org):
        result = await _perform_forget(_ctx(), memory_id=artifact_id)

    assert result.get("forgotten") is True, f"got {result}"


# ---------------------------------------------------------------------------
# No admin override — the product decision, pinned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_project_admin_of_another_project_gets_nothing(db):
    """Membership is per project. Administering one grants nothing elsewhere.

    This is the ``no admin override`` decision in its reachable form: there is
    no role anywhere that widens access to a project's artifacts beyond that
    project's own member list.
    """
    org = _org(db)
    theirs = _project(db, org_id=org, name="apollo")
    mine = _project(db, org_id=org, name="gemini")
    _member(db, org_id=org, project_id=theirs, user_id=PRIYA)
    projects.add_member(
        db,
        org_id=org,
        project_id=mine,
        user_id=DEV,
        role=projects.ROLE_ADMIN,
        granted_by=None,
    )
    db.commit()
    artifact_id = _artifact(db, org_id=org, project_id=theirs)

    with _as(db, DEV, org_id=org):
        refused = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)

    assert refused == {"error": "not_found"}, f"admin override leaked; got {refused}"

    with _as(db, PRIYA, org_id=org):
        allowed = await _perform_artifact_get(_ctx(), artifact_id=artifact_id)
    assert allowed.get("text") == "secret", "positive control failed"


def test_the_helper_is_wired_into_all_three_tools():
    """A structural guard: the gate exists and every arm consults it.

    Cheap insurance against a future refactor that keeps the helper but drops
    a call site — the arms are far apart in one large module.
    """
    source = mcp_app.__file__
    with open(source, encoding="utf-8") as fh:
        body = fh.read()
    assert body.count("_project_permits(") >= 4, (
        "expected the membership gate on artifact_get, artifact_put (project "
        "resolution and supersedes) and forget"
    )
