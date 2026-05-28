"""Real end-to-end tests for the Klio brain MCP service (KB-F2).

Unlike ``test_recall_tool.py`` / ``test_write_tools.py`` (which invoke the tool
callables directly with a fake ``Context`` and mock every DB/embedding seam),
this module drives the *actual* MCP client over the *actual* Streamable-HTTP
transport, in-process, against the *real* Postgres test DB. It reuses the
proven in-process ASGI transport pattern from
``test_mcp_protocol.py::test_tools_list_over_streamable_http_transport``:

    transport → auth bridge → tool body → ``shared.brain`` → real Postgres.

Because the FastAPI app runs in-process via ``httpx.ASGITransport``, a
``unittest.mock.patch`` applied in the test DOES affect the server's handlers
(same process, same module objects). We patch EXACTLY TWO seams and leave
everything else real:

1. **Auth** — ``app.auth.get_validator`` returns a fake validator mapping
   ``"key-A" -> KeyInfo(org_id=ORG_A, ...)`` and ``"key-B" -> KeyInfo(org_id=
   ORG_B, ...)``. This bypasses the real DB key lookup + quota/rate/agent
   checks (unit-tested elsewhere, KB-C2b) while keeping the real
   ``resolve_principal`` / ``verify_api_key`` / header-reading logic — so the
   org isolation we assert is the genuine ``org_id`` scoping that flows from
   the validated key through the tool to the DB query.

2. **Embeddings** — ``shared.memory.embed_text`` returns a FIXED 1024-element
   vector. All of ``write_memory`` / ``find_similar_memory`` /
   ``retrieve_memories_scoped`` call ``embed_text``, so a written row and a
   later query share the same vector (cosine distance 0 → the row is
   recalled deterministically). The column is ``vector(1024)``, so the list
   MUST be length 1024.

Dedup caveat: ``write_scoped_memory`` dedups via ``find_similar_memory`` on
``session_id`` + ``org_id``. Each MCP connection gets its own
``mcp-session-id``, so two identical-vector remembers in ONE session would
dedup. We therefore use DISTINCT content per remember and remember once per
fact.

Isolation: unique per-test ``org_id``s (UUID-suffixed) plus a fixture that
DELETEs every row created for those orgs in teardown (even on failure), so no
cross-test pollution and no dependence on pre-existing DB state.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import text

# The mcp-server conftest puts both ``app`` and ``shared`` on sys.path. The
# config default DATABASE_URL already points at the live test DB, but we pin
# it explicitly (autouse fixture below) so this never depends on ambient env.
from shared.auth import KeyInfo

# A fixed embedding vector of the exact column dimension (vector(1024)). A
# written row and any later query both embed to THIS vector, so cosine
# distance is 0 and the row is always recalled — making the round-trip
# deterministic without a live embedding backend.
_FIXED_EMBEDDING = [0.1] * 1024

# Live test DB (alembic head 021: session_memories has space_id/project_id;
# spaces/projects/brain_curator_state exist). The app's config default already
# resolves to this, but we set it explicitly so app.db.SessionLocal in the
# in-process server connects here regardless of ambient env.
_TEST_DATABASE_URL = "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard"


@pytest.fixture(autouse=True)
def _pin_database_url():
    """Pin DATABASE_URL to the live test DB for the test process.

    ``app.config.DATABASE_URL`` is read at import time and the engine in
    ``app.db`` is built from it, so this is belt-and-suspenders: the default
    already matches, but pinning the env var documents the dependency and
    guards against an ambient override leaking in from the shell.
    """
    prev = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev


def _make_fake_validator(org_a: str, org_b: str):
    """Build a fake KeyValidator whose ``.validate`` maps test keys to orgs.

    Returns a stand-in object exposing only ``.validate(api_key)`` — the one
    method ``verify_api_key`` calls — so the real key/quota/rate DB path is
    bypassed while ``resolve_principal``'s real logic (header read, org
    derivation) stays exercised end-to-end.
    """
    key_to_info = {
        "key-A": KeyInfo(org_id=org_a, key_id="k-a", scopes=["memory"]),
        "key-B": KeyInfo(org_id=org_b, key_id="k-b", scopes=["memory"]),
    }

    class _FakeValidator:
        def validate(self, api_key: str) -> KeyInfo:
            from shared.auth import AuthError

            info = key_to_info.get(api_key)
            if info is None:
                raise AuthError(status_code=401, detail="invalid key")
            return info

    return _FakeValidator()


@pytest.fixture
def orgs():
    """Unique per-test org ids plus DB cleanup of every row they created.

    Yields ``(ORG_A, ORG_B)``. In teardown (run even on test failure), DELETEs
    all ``session_memories`` rows for both orgs so tests never pollute each
    other or leave residue in the shared test DB.
    """
    org_a = f"e2e-org-a-{uuid.uuid4()}"
    org_b = f"e2e-org-b-{uuid.uuid4()}"
    try:
        yield org_a, org_b
    finally:
        # Open a direct session (NOT the in-process server's) for cleanup so
        # teardown is independent of any patching still in scope.
        from app.db import SessionLocal

        db = SessionLocal()
        try:
            db.execute(
                text(
                    "DELETE FROM session_memories WHERE org_id IN (:a, :b)"
                ),
                {"a": org_a, "b": org_b},
            )
            db.commit()
        finally:
            db.close()


def _result_dict(call_tool_result) -> dict:
    """Extract the tool's dict payload from a ``CallToolResult``.

    Our tools return a dict. The MCP SDK surfaces it as ``structuredContent``
    (the parsed object) and also mirrors it as JSON text in
    ``content[0].text``. Prefer the structured form; fall back to json-parsing
    the first text block so the helper is robust to either shape.
    """
    structured = getattr(call_tool_result, "structuredContent", None)
    if structured is not None:
        return structured
    content = getattr(call_tool_result, "content", None) or []
    for block in content:
        text_payload = getattr(block, "text", None)
        if text_payload is not None:
            return json.loads(text_payload)
    raise AssertionError(
        f"could not extract a dict from CallToolResult: {call_tool_result!r}"
    )


@asynccontextmanager
async def _mcp_session(api_key: str, agent_id: str):
    """Open a real MCP ClientSession over the in-process Streamable-HTTP app.

    Builds a FRESH ``create_app()`` per connection (the session manager's
    ``run()`` is one-shot per app instance — mirrors test_mcp_protocol.py),
    routes the SDK client's HTTP through ``httpx.ASGITransport`` so no socket
    is opened, and sets the per-org auth headers F2 adds (``X-Vex-Key`` /
    ``X-Vex-Agent``). The caller drives ``session.call_tool(...)`` inside the
    ``with`` block.
    """
    from app.main import create_app

    app = create_app()
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"X-Vex-Key": api_key, "X-Vex-Agent": agent_id},
    )
    async with app.router.lifespan_context(app):
        async with http_client:
            async with streamable_http_client(
                "http://testserver/mcp", http_client=http_client
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session


def _insert_verification_row(org_id: str, *, session_id: str, content: str) -> str:
    """Directly INSERT a verification-style (scope='session') row for an org.

    Binds the fixed 1024-vector as a pgvector literal exactly the way
    ``write_memory`` does (``str([...])`` bound as a plain param). Returns the
    new row id. Used to prove that an org-scope recall excludes scope='session'
    rows at real query-execution time (not merely in the SQL text).
    """
    from app.db import SessionLocal

    db = SessionLocal()
    row_id = str(uuid.uuid4())
    try:
        db.execute(
            text(
                """
                INSERT INTO session_memories (
                    id, session_id, org_id, agent_id, execution_id,
                    memory_type, content, embedding, confidence,
                    sequence_number, scope, status, metadata,
                    created_at, updated_at
                ) VALUES (
                    :id, :session_id, :org_id, :agent_id, NULL,
                    'observation', :content, :embedding, NULL,
                    NULL, 'session', 'active', '{}'::jsonb,
                    NOW(), NOW()
                )
                """
            ),
            {
                "id": row_id,
                "session_id": session_id,
                "org_id": org_id,
                "agent_id": "verif-agent",
                "content": content,
                "embedding": str(_FIXED_EMBEDDING),
            },
        )
        db.commit()
    finally:
        db.close()
    return row_id


@pytest.fixture
def _patched_seams(orgs):
    """Patch the two non-deterministic seams (auth + embeddings) for the test.

    * ``app.auth.get_validator`` → fake validator mapping key-A/key-B to the
      unique per-test orgs (resets the ``app.auth._validator`` singleton so a
      previously-built real validator is never consulted).
    * ``shared.memory.embed_text`` → AsyncMock returning the fixed 1024-vector.

    Everything else (transport, resolve_principal, brain writes/reads, the DB)
    is REAL. Yields ``(ORG_A, ORG_B)``.
    """
    org_a, org_b = orgs
    fake_validator = _make_fake_validator(org_a, org_b)
    embed_mock = AsyncMock(return_value=list(_FIXED_EMBEDDING))

    import app.auth as auth_mod

    # Reset the module singleton so nothing stale is reused, then patch the
    # accessor so verify_api_key never builds/consults a real validator.
    auth_mod._validator = None
    with (
        patch("app.auth.get_validator", return_value=fake_validator),
        patch("shared.memory.embed_text", embed_mock),
    ):
        yield org_a, org_b


# ---------------------------------------------------------------------------
# 1. Round-trip (org A): remember then recall over the real transport + DB.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_then_recall_round_trip(_patched_seams):
    """remember persists a row; a later same-org recall returns its content.

    Proves the full path transport→auth→tool→shared.brain→Postgres works
    end-to-end: the write is stored and the read finds it (the shared fixed
    embedding guarantees the row ranks first / is returned)."""
    org_a, _org_b = _patched_seams
    fact = "JWT refresh rotates tokens via the /auth/refresh endpoint"

    async with _mcp_session("key-A", "agent-a") as session:
        stored = _result_dict(await session.call_tool("remember", {"content": fact}))
        assert stored["stored"] is True, stored
        assert stored.get("id"), f"expected a row id, got {stored!r}"

        recalled = _result_dict(
            await session.call_tool("recall", {"query": "how does JWT refresh work"})
        )

    assert recalled["count"] >= 1, recalled
    contents = [m["content"] for m in recalled["memories"]]
    assert fact in contents, f"remembered fact not recalled: {contents!r}"


# ---------------------------------------------------------------------------
# 2. Org isolation: org B cannot recall org A's memory.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_isolation_org_b_cannot_recall_org_a(_patched_seams):
    """A memory written by org A is invisible to org B's recall.

    org A remembers a fact; org B (a clean unique org with no rows of its own)
    recalls and gets ZERO results — proving org_id scoping holds end-to-end
    through transport→auth→tool→DB, not just in a unit mock."""
    org_a, _org_b = _patched_seams
    fact = "Org A secret: deploy key is rotated every 30 days"

    async with _mcp_session("key-A", "agent-a") as session_a:
        stored = _result_dict(await session_a.call_tool("remember", {"content": fact}))
        assert stored["stored"] is True, stored

    async with _mcp_session("key-B", "agent-b") as session_b:
        recalled_b = _result_dict(
            await session_b.call_tool("recall", {"query": "deploy key rotation"})
        )

    # org B is a fresh unique org with no rows, so this is a clean zero.
    assert recalled_b["count"] == 0, recalled_b
    contents_b = [m["content"] for m in recalled_b["memories"]]
    assert fact not in contents_b, f"org A's row leaked to org B: {contents_b!r}"


# ---------------------------------------------------------------------------
# 3. Result-level coexistence: scope='org' recall excludes scope='session'.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_recall_excludes_verification_session_rows(_patched_seams):
    """A directly-inserted verification (scope='session') row is NOT returned
    by an org-scope recall — proven at real query-execution time.

    This folds in the result-level gap KB-F1 lacked: F1 asserted the scope
    guard at the SQL-text / namespacing level, but here we INSERT an actual
    scope='session' row carrying the SAME fixed embedding (so it WOULD rank
    top if the scope filter were a no-op) and confirm the real query excludes
    it. A brain row written via remember (scope='org') with distinct content
    is still recalled, proving the filter discriminates by scope, not by
    accident of ranking."""
    org_a, _org_b = _patched_seams
    verif_content = "VERIFICATION ROW: ephemeral session observation, not brain"
    brain_fact = "Brain fact: the API gateway enforces a 100 req/min rate limit"

    # Insert a verification-style row (scope='session') for org A directly.
    verif_session_id = f"e2e-verif-{uuid.uuid4()}"
    _insert_verification_row(
        org_a, session_id=verif_session_id, content=verif_content
    )

    async with _mcp_session("key-A", "agent-a") as session_a:
        stored = _result_dict(
            await session_a.call_tool("remember", {"content": brain_fact})
        )
        assert stored["stored"] is True, stored

        recalled = _result_dict(
            await session_a.call_tool(
                "recall", {"query": "rate limit policy", "scope": "org"}
            )
        )

    contents = [m["content"] for m in recalled["memories"]]
    # The scope='session' verification row must be excluded at query time...
    assert verif_content not in contents, (
        f"verification (scope='session') row leaked into org recall: {contents!r}"
    )
    # ...while the scope='org' brain row written via remember IS recallable.
    assert brain_fact in contents, f"brain row missing from org recall: {contents!r}"
    # Every returned row is org-scoped (the visibility ladder's org rung).
    assert all(m["scope"] == "org" for m in recalled["memories"]), recalled


# ---------------------------------------------------------------------------
# 4. Agent scope: org rows are visible at agent scope too.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_scope_recall_returns_org_memory(_patched_seams):
    """An org memory (written via remember, scope='org') is visible to an
    agent-scope recall with the agent header set.

    The visibility ladder gives scope='agent' the org rung PLUS the agent's
    own private rows; a remember writes at scope='org', so it surfaces under
    an agent-scope recall — proven end-to-end through the transport."""
    org_a, _org_b = _patched_seams
    fact = "Agent-scope check: the staging URL is staging.example.internal"

    async with _mcp_session("key-A", "agent-a") as session:
        stored = _result_dict(await session.call_tool("remember", {"content": fact}))
        assert stored["stored"] is True, stored

        recalled = _result_dict(
            await session.call_tool(
                "recall", {"query": "what is the staging URL", "scope": "agent"}
            )
        )

    contents = [m["content"] for m in recalled["memories"]]
    assert fact in contents, f"org memory not visible at agent scope: {contents!r}"
