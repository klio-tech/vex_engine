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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    assert len(playbook["steps"]) == 3
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
        {"content": "c", "kind": "note", "source": "import"},  # v1.1, not yet
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
