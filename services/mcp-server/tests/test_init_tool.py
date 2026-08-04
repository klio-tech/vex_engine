"""Contract and invariant tests for the `init` onboarding tool.

Two layers, mirroring the house pattern:

* Wiring/contract tests run against the registered tool callable with a fake
  Context and patched principal seams (no DB) — what matters is which
  arguments the tool derives and forwards, and what its schema promises.
* The five security invariants from the spec
  (docs/superpowers/specs/2026-08-03-klio-init-onboarding-design.md) run
  against real Postgres further down, gated by the shared ``requires_db``
  marker: the redaction/dedup door IS the SQL + chokepoint, and a mock would
  pass with no door at all.
"""

import hashlib
import math
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.testing import requires_db
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text

import app.onboarding as onboarding
from app.mcp_app import EXPECTED_TOOL_NAMES, build_mcp_server

USER = "11111111-2222-3333-4444-555555555555"


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _get_tool(name: str = "init"):
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool


def _principal_patches(user_id=USER):
    """Patch the two identity seams at the onboarding module boundary."""
    return (
        patch(
            "app.onboarding.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.onboarding.resolve_request_user_id", return_value=user_id),
        patch("app.onboarding.SessionLocal", MagicMock(return_value=MagicMock())),
    )


# ---------------------------------------------------------------------------
# Inventory + routing contract
# ---------------------------------------------------------------------------


def test_tool_inventory_is_nineteen_and_includes_init():
    """18 -> 19 tools; `init` asserted by name (spec acceptance)."""
    assert "init" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 19
    server = build_mcp_server()
    assert server._tool_manager.get_tool("init") is not None


def test_init_description_contains_routing_phrases():
    """'onboard me' / 'set me up' / 'get started' must all route here.

    The UX contract is a sentence, not a CLI flag: the description is the
    router, so the phrases are pinned.
    """
    desc = (_get_tool().description or "").lower()
    for phrase in ("onboard", "set up", "get started"):
        assert phrase in desc, f"init description missing routing phrase {phrase!r}"


def test_init_schema_requires_phase_and_offers_no_scope_lever():
    """`phase` is the one required argument, and NOTHING in the public schema
    can name a scope — invariant 2 starts at the contract surface."""
    tool = _get_tool()
    schema = tool.parameters
    props = schema.get("properties", {})
    assert "phase" in props
    assert "phase" in schema.get("required", [])
    assert "scope" not in props
    # No property description may promise scope control either.
    for name in props:
        assert "scope" not in name.lower()


# ---------------------------------------------------------------------------
# Phase: start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_returns_versioned_playbook_with_three_steps_and_caps():
    fn = _get_tool().fn
    p = _principal_patches()
    with p[0], p[1], p[2], patch("app.onboarding._count_runs_today", return_value=0):
        result = await fn(_fake_ctx(), phase="start")

    assert result["phase"] == "start"
    playbook = result["playbook"]
    assert playbook["version"] == onboarding.PLAYBOOK_VERSION
    assert len(playbook["steps"]) == 4  # v1.1 added the consent-gated import step
    # Step 3 is the consent gate, not decoration.
    joined = " ".join(s["instructions"] for s in playbook["steps"]).lower()
    assert "not" in joined and "remember" in joined
    # The playbook tells the agent what NOT to submit.
    refusals = " ".join(playbook["do_not_submit"]).lower()
    for banned in ("secret", "personal", "off-limits"):
        assert banned in refusals
    assert result["caps"] == {
        "candidates_per_submit": 25,
        "accepted_per_run": 60,
        "content_chars_per_candidate": 2000,
        "runs_per_day": 3,
    }


@pytest.mark.asyncio
async def test_start_refuses_an_unattributed_caller():
    """init seeds a PRIVATE silo; without a person there is no silo to seed."""
    fn = _get_tool().fn
    p = _principal_patches(user_id=None)
    with p[0], p[1], p[2]:
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "unattributed_caller"


@pytest.mark.asyncio
async def test_start_refuses_a_fourth_run_in_one_day():
    fn = _get_tool().fn
    p = _principal_patches()
    with p[0], p[1], p[2], patch("app.onboarding._count_runs_today", return_value=3):
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "run_limit_reached"
    assert "3" in result["detail"]


@pytest.mark.asyncio
async def test_start_is_refused_cleanly_on_auth_failure():
    from fastapi import HTTPException

    fn = _get_tool().fn
    with patch(
        "app.onboarding.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "unauthorized"


# ---------------------------------------------------------------------------
# Phase: submit — wiring (mocked chokepoint; the real door is tested with
# Postgres in the invariant section below)
# ---------------------------------------------------------------------------


def _stored(memory_id="00000000-0000-0000-0000-000000000001"):
    return {
        "stored": True,
        "deduplicated": False,
        "id": memory_id,
        "memory_type": "note",
        "scope": "private",
        "redacted": False,
        "redaction_kinds": [],
    }


def _submit_patches(user_id=USER, accepted_today=0):
    return (
        *_principal_patches(user_id=user_id),
        patch("app.onboarding.check_memory_quota", return_value=False),
        patch("app.onboarding.record_usage"),
        patch("app.onboarding._count_accepted_today", return_value=accepted_today),
    )


async def _run_submit(
    candidates, *, write_mock=None, accepted_today=0, git_remote=None
):
    fn = _get_tool().fn
    write_mock = write_mock or AsyncMock(return_value=_stored())
    p = _submit_patches(accepted_today=accepted_today)
    with (
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        patch("app.onboarding.brain.write_scoped_memory", write_mock),
        patch("app.onboarding.brain.find_project_id", return_value=None),
    ):
        result = await fn(
            _fake_ctx(), phase="submit", candidates=candidates, git_remote=git_remote
        )
    return result, write_mock


@pytest.mark.asyncio
async def test_submit_with_no_candidates_is_an_actionable_error():
    result, write_mock = await _run_submit(None)
    assert result["error"] == "no_candidates"
    write_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_rejects_a_call_with_more_than_25_candidates():
    """Cap 1: candidates per submit call — reject the CALL, tell the agent
    to prioritise. Nothing is stored."""
    candidates = [
        {"content": f"fact {i}", "kind": "note", "source": "repo"} for i in range(26)
    ]
    result, write_mock = await _run_submit(candidates)
    assert result["error"] == "too_many_candidates"
    assert "25" in result["detail"]
    assert "priorit" in result["detail"].lower()
    write_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_rejects_an_oversize_candidate_but_not_the_batch():
    """Cap 3: content length — reject the CANDIDATE, not the batch."""
    candidates = [
        {"content": "x" * 2001, "kind": "note", "source": "repo"},
        {"content": "a good fact", "kind": "note", "source": "repo"},
    ]
    result, write_mock = await _run_submit(candidates)
    assert write_mock.await_count == 1
    assert len(result["accepted"]) == 1
    assert result["refused"] == [
        {"index": 0, "reason": "content_too_long", "limit": 2000}
    ]


@pytest.mark.asyncio
async def test_submit_refuses_bad_kind_bad_source_and_empty_content_per_candidate():
    candidates = [
        {"content": "c", "kind": "fact", "source": "repo"},  # curator output type
        {"content": "c", "kind": "note", "source": "scraped"},  # never a source
        {"content": "   ", "kind": "note", "source": "repo"},
        "not even a dict",
        {"content": "legit", "kind": "decision", "source": "conversation"},
    ]
    result, write_mock = await _run_submit(candidates)
    assert write_mock.await_count == 1
    reasons = {r["index"]: r["reason"] for r in result["refused"]}
    assert reasons == {
        0: "invalid_kind",
        1: "invalid_source",
        2: "empty_content",
        3: "invalid_candidate",
    }


@pytest.mark.asyncio
async def test_submit_hardcodes_private_scope_and_maps_provenance_by_source():
    """The write call must say scope='private' verbatim, user_id from the
    resolver, EXTRACTED only for source='user', INFERRED otherwise."""
    candidates = [
        {"content": "the user stated this", "kind": "memory", "source": "user"},
        {"content": "derived from repo", "kind": "observation", "source": "repo"},
        {"content": "derived from chat", "kind": "plan", "source": "conversation"},
    ]
    result, write_mock = await _run_submit(candidates)
    assert len(result["accepted"]) == 3
    calls = [c.kwargs for c in write_mock.await_args_list]
    assert [c["provenance"] for c in calls] == ["EXTRACTED", "INFERRED", "INFERRED"]
    for c in calls:
        assert c["scope"] == "private"
        assert c["user_id"] == USER
        assert c["org_id"] == "org-x"
    assert [c["memory_type"] for c in calls] == ["memory", "observation", "plan"]


@pytest.mark.asyncio
async def test_submit_ignores_identity_and_scope_fields_smuggled_in_the_payload():
    """Invariant 2 + 3 at the wiring level: a candidate that names a scope,
    user, or org changes NOTHING about where the row lands."""
    candidates = [
        {
            "content": "smuggle attempt",
            "kind": "note",
            "source": "repo",
            "scope": "org",
            "user_id": "99999999-8888-7777-6666-555555555555",
            "org_id": "victim-org",
        }
    ]
    result, write_mock = await _run_submit(candidates)
    assert len(result["accepted"]) == 1
    kwargs = write_mock.await_args.kwargs
    assert kwargs["scope"] == "private"
    assert kwargs["user_id"] == USER
    assert kwargs["org_id"] == "org-x"


@pytest.mark.asyncio
async def test_submit_refuses_the_call_once_the_run_cap_is_exhausted():
    """Cap 2: 60 accepted per run, server-derived — a client that skips
    start or reconnects cannot reset it."""
    candidates = [{"content": "one more", "kind": "note", "source": "repo"}]
    result, write_mock = await _run_submit(candidates, accepted_today=60)
    assert result["error"] == "run_cap_reached"
    assert "60" in result["detail"]
    write_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_truncates_a_batch_that_crosses_the_run_cap():
    candidates = [
        {"content": f"fact number {i}", "kind": "note", "source": "repo"}
        for i in range(4)
    ]
    result, write_mock = await _run_submit(candidates, accepted_today=58)
    assert write_mock.await_count == 2
    reasons = [r["reason"] for r in result["refused"]]
    assert reasons == ["run_cap_reached", "run_cap_reached"]


@pytest.mark.asyncio
async def test_submit_files_under_the_project_resolved_from_git_remote():
    """Filing goes through the existing find_project_id path — read-only,
    org-scoped, fail-open to unfiled."""
    fn = _get_tool().fn
    write_mock = AsyncMock(return_value=_stored())
    find_mock = MagicMock(return_value="proj-123")
    p = _submit_patches()
    with (
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        patch("app.onboarding.brain.write_scoped_memory", write_mock),
        patch("app.onboarding.brain.find_project_id", find_mock),
    ):
        result = await fn(
            _fake_ctx(),
            phase="submit",
            candidates=[{"content": "c", "kind": "note", "source": "repo"}],
            git_remote="git@github.com:klio-tech/klio.git",
        )
    assert len(result["accepted"]) == 1
    find_mock.assert_called_once()
    assert find_mock.call_args.kwargs["org_id"] == "org-x"
    assert write_mock.await_args.kwargs["project_id"] == "proj-123"


@pytest.mark.asyncio
async def test_submit_reports_dedup_and_redaction_honestly():
    """The response tells the agent what was accepted, deduplicated, and
    redacted — so the agent can report honestly to the user."""
    results = iter(
        [
            {
                "stored": True,
                "deduplicated": False,
                "id": "id-1",
                "memory_type": "note",
                "scope": "private",
                "redacted": True,
                "redaction_kinds": ["stripe_secret_key"],
            },
            {
                "stored": False,
                "deduplicated": True,
                "id": "id-existing",
                "memory_type": "note",
                "scope": "private",
                "redacted": False,
                "redaction_kinds": [],
            },
        ]
    )
    write_mock = AsyncMock(side_effect=lambda *a, **k: next(results))
    candidates = [
        {"content": "has a secret", "kind": "note", "source": "repo"},
        {"content": "already known", "kind": "note", "source": "repo"},
    ]
    result, _ = await _run_submit(candidates, write_mock=write_mock)
    assert result["accepted"] == ["id-1"]
    assert result["deduplicated"] == 1
    assert result["redacted_kinds"] == ["stripe_secret_key"]


@pytest.mark.asyncio
async def test_submit_surfaces_a_quota_block_cleanly():
    from shared.usage import MemoryQuotaExceeded

    fn = _get_tool().fn
    p = _principal_patches()
    with (
        p[0],
        p[1],
        p[2],
        patch("app.onboarding._count_accepted_today", return_value=0),
        patch(
            "app.onboarding.check_memory_quota",
            side_effect=MemoryQuotaExceeded(
                metric="memory", limit=100, plan_used=101, suggested_plan="max"
            ),
        ),
    ):
        result = await fn(
            _fake_ctx(),
            phase="submit",
            candidates=[{"content": "c", "kind": "note", "source": "repo"}],
        )
    assert result["error"] == "quota_exceeded"


# ---------------------------------------------------------------------------
# Phase: finish
# ---------------------------------------------------------------------------


_RUN_SUMMARY = {
    "total": 5,
    "by_kind": {"note": 3, "decision": 2},
    "by_source": {"repo": 4, "user": 1},
    "redacted_kinds": ["email_address"],
}


async def _run_finish(write_mock=None, summary=None):
    fn = _get_tool().fn
    write_mock = write_mock or AsyncMock(
        return_value={"stored": True, "id": "marker-id", "deduplicated": False}
    )
    p = _principal_patches()
    with (
        p[0],
        p[1],
        p[2],
        patch(
            "app.onboarding._summarize_run_today",
            return_value=summary if summary is not None else dict(_RUN_SUMMARY),
        ),
        patch("app.onboarding.brain.write_scoped_memory", write_mock),
    ):
        result = await fn(_fake_ctx(), phase="finish")
    return result, write_mock


@pytest.mark.asyncio
async def test_finish_reports_counts_by_kind_and_source_and_kinds_only_redactions():
    result, _ = await _run_finish()
    assert result["phase"] == "finish"
    assert result["seeded"] == 5
    assert result["by_kind"] == {"note": 3, "decision": 2}
    assert result["by_source"] == {"repo": 4, "user": 1}
    # Kinds only, never contents.
    assert result["redacted_kinds"] == ["email_address"]


@pytest.mark.asyncio
async def test_finish_returns_a_relay_paragraph_the_agent_repeats_verbatim():
    result, _ = await _run_finish()
    relay = result["relay"]
    assert isinstance(relay, str)
    assert "Klio now knows" in relay
    assert "5" in relay


@pytest.mark.asyncio
async def test_finish_writes_the_rerun_marker_note_with_dedup_off():
    """The marker makes re-runs detectable AND countable (3/day derives from
    it), so it must be written even when its text matches a prior run's."""
    result, write_mock = await _run_finish()
    write_mock.assert_awaited_once()
    kwargs = write_mock.await_args.kwargs
    assert kwargs["memory_type"] == "note"
    assert kwargs["scope"] == "private"
    assert kwargs["user_id"] == USER
    assert kwargs["dedup"] is False
    assert kwargs["metadata"]["init_marker"] is True
    assert kwargs["metadata"]["source"] == "init"
    assert kwargs["content"].startswith("Onboarded via init on ")
    assert "5 memories seeded" in kwargs["content"]
    assert result["marker_id"] == "marker-id"


# ===========================================================================
# The five spec invariants — real Postgres, real chokepoint.
#
# Everything below uses the REAL SessionLocal / write_scoped_memory /
# redaction / dedup SQL. Only three seams are faked:
#   * the identity resolvers (this suite is not an auth suite — the identity
#     ATTACK is that the payload must not override what they resolve);
#   * quota metering (has its own suites);
#   * embed_text — a deterministic content-hash vector, so identical content
#     embeds identically (dedup fires at the real threshold) and different
#     content embeds ~orthogonally (no false dedup). No network calls.
# ===========================================================================

needs_db = requires_db("the init tool security-invariant tests")

_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

EMBED_DIM = 1024
CALLER = "aaaaaaaa-1111-2222-3333-444444444444"
VICTIM = "bbbbbbbb-5555-6666-7777-888888888888"


def _deterministic_embedding(content: str) -> list[float]:
    """Unit vector derived from the content hash: equal text -> equal vector
    (cosine 1.0, above the 0.95 dedup threshold); different text -> a fresh
    pseudo-random direction (cosine ~0 in 1024 dims, far below it)."""
    out: list[float] = []
    counter = 0
    while len(out) < EMBED_DIM:
        digest = hashlib.sha256(f"{content}:{counter}".encode()).digest()
        out.extend(b / 255.0 - 0.5 for b in digest)
        counter += 1
    vec = out[:EMBED_DIM]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def dbconn():
    engine = create_engine(_DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            for org_id in orgs:
                conn.execute(
                    sql_text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(dbconn) -> str:
    org_id = f"org_init_{uuid.uuid4().hex[:8]}"
    dbconn.created_org_ids.add(org_id)
    return org_id


async def _init_call(org_id, user_id, phase, candidates=None, git_remote=None):
    """Drive the real tool body against real Postgres.

    Identity comes from the patched resolvers — matching production, where it
    comes from the key/OAuth and NEVER from the payload the agent controls.
    """
    fn = _get_tool().fn
    async_embed = AsyncMock(side_effect=lambda c: _deterministic_embedding(c))
    with (
        patch(
            "app.onboarding.resolve_principal_plan",
            return_value=(org_id, "agent-init", "free", None),
        ),
        patch("app.onboarding.resolve_request_user_id", return_value=user_id),
        patch("app.onboarding.check_memory_quota", return_value=False),
        patch("app.onboarding.record_usage"),
        patch("shared.memory.embed_text", async_embed),
    ):
        return await fn(
            _fake_ctx(), phase=phase, candidates=candidates, git_remote=git_remote
        )


def _rows(dbconn, org_id):
    return dbconn.execute(
        sql_text(
            "SELECT content, scope, user_id, memory_type, provenance, metadata "
            "FROM session_memories WHERE org_id = :o "
            "AND COALESCE(metadata->>'init_marker', 'false') <> 'true' "
            "ORDER BY created_at"
        ),
        {"o": org_id},
    ).fetchall()


# ---------------------------------------------------------------------------
# Invariant 1 — no redaction bypass
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_invariant1_secret_in_a_candidate_is_masked_before_storage(dbconn):
    org = _org(dbconn)
    secret = "sk_live_" + "a1B2c3D4e5F6g7H8i9J0" * 2
    result = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {
                "content": f"deploy uses stripe key {secret} in the worker",
                "kind": "note",
                "source": "repo",
            }
        ],
    )
    assert len(result["accepted"]) == 1
    assert "stripe_secret_key" in result["redacted_kinds"]
    rows = _rows(dbconn, org)
    assert len(rows) == 1
    assert secret not in rows[0].content
    assert "[REDACTED:stripe_secret_key" in rows[0].content


@needs_db
@pytest.mark.asyncio
async def test_invariant1_pii_email_in_a_candidate_is_masked_before_storage(dbconn):
    org = _org(dbconn)
    result = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {
                "content": "escalations go to oncall-root@example-corp.com first",
                "kind": "note",
                "source": "conversation",
            }
        ],
    )
    assert len(result["accepted"]) == 1
    assert "email" in result["redacted_kinds"]
    rows = _rows(dbconn, org)
    assert "oncall-root@example-corp.com" not in rows[0].content
    assert "[REDACTED:email" in rows[0].content


@needs_db
@pytest.mark.asyncio
async def test_invariant1_positive_control_clean_content_stored_verbatim(dbconn):
    """Proves the redaction fixture is not vacuous: content with nothing to
    redact passes through the same door unchanged."""
    org = _org(dbconn)
    clean = "the team prefers trunk-based development with small PRs"
    result = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[{"content": clean, "kind": "note", "source": "conversation"}],
    )
    assert len(result["accepted"]) == 1
    assert result["redacted_kinds"] == []
    rows = _rows(dbconn, org)
    assert rows[0].content == clean
    assert "[REDACTED" not in rows[0].content


# ---------------------------------------------------------------------------
# Invariant 2 — private only
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_invariant2_no_payload_field_can_widen_the_scope(dbconn):
    """Adversarial candidates name every scope spelling a client might try;
    every stored row must still be scope='private'."""
    org = _org(dbconn)
    result = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {"content": "widen me A", "kind": "note", "source": "repo", "scope": "org"},
            {
                "content": "widen me B",
                "kind": "decision",
                "source": "user",
                "scope": "project",
                "visibility": "org",
                "share": True,
            },
            {"content": "an honest one", "kind": "note", "source": "repo"},
        ],
    )
    # Positive control inside the same test: the writes actually happened.
    assert len(result["accepted"]) == 3
    rows = _rows(dbconn, org)
    assert len(rows) == 3
    assert {r.scope for r in rows} == {"private"}


@needs_db
@pytest.mark.asyncio
async def test_invariant2_rows_are_invisible_to_a_teammates_recall(dbconn):
    """The point of 'private': a teammate's ladder never includes these rows."""
    from shared.memory import retrieve_memories_scoped

    org = _org(dbconn)
    await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {
                "content": "the caller's own onboarding context",
                "kind": "note",
                "source": "conversation",
            }
        ],
    )
    async_embed = AsyncMock(side_effect=lambda c: _deterministic_embedding(c))
    with patch("shared.memory.embed_text", async_embed):
        teammate_sees = await retrieve_memories_scoped(
            dbconn,
            org,
            "the caller's own onboarding context",
            scope="org",
            agent_id="agent-other",
            user_id=VICTIM,
            limit=10,
        )
        owner_sees = await retrieve_memories_scoped(
            dbconn,
            org,
            "the caller's own onboarding context",
            scope="private",
            agent_id="agent-init",
            user_id=CALLER,
            limit=10,
        )
    assert teammate_sees == []
    # Positive control: the owner CAN see it, so the empty result above is
    # the boundary, not a broken fixture.
    assert len(owner_sees) == 1


# ---------------------------------------------------------------------------
# Invariant 3 — identity is server-resolved
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_invariant3_payload_identity_fields_never_reach_the_row(dbconn):
    org = _org(dbconn)
    result = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {
                "content": "attribute this to someone else",
                "kind": "note",
                "source": "repo",
                "user_id": VICTIM,
                "org_id": "victim-org",
                "agent_id": "victim-agent",
            }
        ],
    )
    assert len(result["accepted"]) == 1
    rows = _rows(dbconn, org)
    assert len(rows) == 1
    assert str(rows[0].user_id) == CALLER
    # And nothing landed in the org the payload named.
    foreign = dbconn.execute(
        sql_text("SELECT COUNT(*) FROM session_memories WHERE org_id = 'victim-org'")
    ).scalar()
    assert foreign == 0


# ---------------------------------------------------------------------------
# Invariant 4 — caps are server-side
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_invariant4_run_cap_holds_against_a_client_that_ignores_the_playbook(
    dbconn,
):
    """Three max-size batches: 25 + 25 accepted, then 10 accepted + 15
    refused, then the next call is refused outright. Counted from rows, so
    reconnecting or skipping start resets nothing."""
    org = _org(dbconn)

    def batch(tag, n):
        return [
            {
                "content": f"distinct durable fact {tag}-{i:03d} about the system",
                "kind": "note",
                "source": "repo",
            }
            for i in range(n)
        ]

    first = await _init_call(org, CALLER, "submit", candidates=batch("a", 25))
    second = await _init_call(org, CALLER, "submit", candidates=batch("b", 25))
    third = await _init_call(org, CALLER, "submit", candidates=batch("c", 25))
    assert len(first["accepted"]) == 25
    assert len(second["accepted"]) == 25
    assert len(third["accepted"]) == 10
    assert [r["reason"] for r in third["refused"]] == ["run_cap_reached"] * 15

    fourth = await _init_call(org, CALLER, "submit", candidates=batch("d", 1))
    assert fourth["error"] == "run_cap_reached"
    assert len(_rows(dbconn, org)) == 60


@needs_db
@pytest.mark.asyncio
async def test_invariant4_daily_run_limit_counts_real_markers(dbconn):
    """Three finished runs -> the fourth start is refused, derived purely
    from marker notes in the DB."""
    org = _org(dbconn)
    for _ in range(3):
        result = await _init_call(org, CALLER, "finish")
        assert result["phase"] == "finish"
    refused = await _init_call(org, CALLER, "start")
    assert refused["error"] == "run_limit_reached"
    # Positive control: a DIFFERENT user in the same org still may start.
    ok = await _init_call(org, VICTIM, "start")
    assert ok["phase"] == "start"


# ---------------------------------------------------------------------------
# Invariant 5 — idempotence
# ---------------------------------------------------------------------------

_RERUN_CANDIDATES = [
    {
        "content": "the deploy pipeline promotes via Railway from GitHub source",
        "kind": "note",
        "source": "repo",
    },
    {
        "content": "health checks live at /healthz, not /health",
        "kind": "decision",
        "source": "repo",
    },
    {
        "content": "the user prefers status reports as short bullet points",
        "kind": "memory",
        "source": "user",
    },
]


@needs_db
@pytest.mark.asyncio
async def test_invariant5_rerunning_init_on_the_same_context_accepts_zero(dbconn):
    org = _org(dbconn)
    first = await _init_call(org, CALLER, "submit", candidates=_RERUN_CANDIDATES)
    assert len(first["accepted"]) == 3  # positive control: run one stores
    await _init_call(org, CALLER, "finish")

    rerun = await _init_call(org, CALLER, "submit", candidates=_RERUN_CANDIDATES)
    assert rerun["accepted"] == []
    assert rerun["deduplicated"] == 3
    assert len(_rows(dbconn, org)) == 3


# ---------------------------------------------------------------------------
# The full exchange, end to end
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_full_onboarding_exchange_start_submit_finish(dbconn, capsys):
    org = _org(dbconn)
    start = await _init_call(org, CALLER, "start")
    assert start["phase"] == "start"
    assert start["playbook"]["version"] == onboarding.PLAYBOOK_VERSION

    submit = await _init_call(
        org,
        CALLER,
        "submit",
        candidates=[
            {
                "content": "Abhi is building Klio, a memory product on the Vex engine",
                "kind": "memory",
                "source": "conversation",
            },
            {
                "content": "decided: MCP is the primary protocol; skills and hooks layer on top",
                "kind": "decision",
                "source": "conversation",
            },
            {
                "content": "repo convention: many small files, 200-400 lines typical",
                "kind": "note",
                "source": "repo",
            },
            {
                "content": "agents keep getting wrong: docker hub publish is NOT the deploy path",
                "kind": "memory",
                "source": "user",
            },
        ],
    )
    assert len(submit["accepted"]) == 4
    assert submit["refused"] == []

    finish = await _init_call(org, CALLER, "finish")
    assert finish["seeded"] == 4
    assert finish["by_source"] == {"conversation": 2, "repo": 1, "user": 1}
    assert finish["by_kind"] == {"memory": 2, "decision": 1, "note": 1}
    assert "Klio now knows 4 things" in finish["relay"]
    assert finish["marker_id"]

    # Provenance mapping held end to end.
    rows = _rows(dbconn, org)
    prov = {r.content[:20]: r.provenance for r in rows}
    assert prov["agents keep getting "] == "EXTRACTED"
    assert all(
        p == "INFERRED" for c, p in prov.items() if not c.startswith("agents keep")
    )

    print("\n--- init exchange transcript ---")
    print("agent -> init(phase='start')")
    print(f"server -> playbook v{start['playbook']['version']}, caps {start['caps']}")
    print("agent -> init(phase='submit', candidates=[4 items])")
    print(
        f"server -> accepted={len(submit['accepted'])} deduplicated="
        f"{submit['deduplicated']} refused={submit['refused']}"
    )
    print("agent -> init(phase='finish')")
    print(f"server -> relay: {finish['relay']}")


# ---------------------------------------------------------------------------
# v1.1 — the import source, and the relay grammar fix that rode along with it
# ---------------------------------------------------------------------------


def test_relay_pluralizes_a_count_of_one():
    """'Klio now knows 1 things' shipped to production; pin the grammar."""
    from app.onboarding import _relay_paragraph

    one = _relay_paragraph(
        {"by_kind": {"decision": 1}, "by_source": {"user": 1}, "redacted_kinds": []}
    )
    many = _relay_paragraph(
        {
            "by_kind": {"decision": 2, "note": 1},
            "by_source": {"user": 3},
            "redacted_kinds": [],
        }
    )
    assert "1 thing " in one and "1 things" not in one
    assert "3 things" in many


def test_import_is_an_allowed_source_and_lands_inferred():
    """v1.1: a pasted export from another assistant is fuel, but it is
    machine-distilled second-hand — INFERRED, never EXTRACTED."""
    from app.onboarding import ALLOWED_SOURCES, _provenance_for

    assert "import" in ALLOWED_SOURCES
    assert _provenance_for("import") == "INFERRED"
    assert _provenance_for("user") == "EXTRACTED"


def test_playbook_has_an_import_step_with_consent_language():
    """The start playbook must tell the agent how to handle a pasted
    Claude-memory / ChatGPT export: only when the user offers it, never
    solicited by the agent scraping another tool itself."""
    from app.onboarding import PLAYBOOK

    steps = {s["id"]: s for s in PLAYBOOK["steps"]}
    assert "import" in steps
    text = steps["import"]["instructions"].lower()
    assert "paste" in text and "source='import'" in steps["import"]["instructions"]
    assert "do not" in text or "only if" in text or "never" in text
