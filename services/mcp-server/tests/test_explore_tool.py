"""The ``explore`` tool: an entity brief instead of disconnected rows.

Mirrors the mock style of ``tests/test_recall_tool.py``: the tool callable is
reached via the FastMCP registry and invoked with a hand-built fake Context;
the auth / identity / DB / GraphStore / LLM seams are patched at the
``app.mcp_app`` module boundary. The *storage-level* visibility attacks live
in ``services/shared/tests/test_graph_isolation.py`` against real Postgres;
this file pins the tool wiring on top of them:

* not_found is ONE opaque payload — the same for a nonexistent entity and one
  whose evidence the caller cannot see (never an empty brief, never
  "forbidden": both would confirm the name exists);
* the caller's resolved user is forwarded to ``entity_brief`` (the leak-relevant
  wire), and the org comes from the principal, never from caller input;
* contributors surface as COUNTS ("you", "2 teammates"), never ids;
* the synthesized summary is labelled INFERRED, and an LLM failure degrades to
  a summary-less brief rather than an error.
"""

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

import app.mcp_app as mcp_app
from app.mcp_app import EXPECTED_TOOL_NAMES, build_mcp_server
from shared.memory import MemoryRecord

PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _explore_fn():
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("explore")
    assert tool is not None, "tool 'explore' not registered"
    return tool.fn


def _record(*, content, user_id=None, memory_type="fact", provenance="EXTRACTED"):
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="s",
        org_id="org-x",
        agent_id="a",
        memory_type=memory_type,
        content=content,
        scope="org",
        user_id=user_id,
        provenance=provenance,
        created_at=datetime.now(timezone.utc),
    )


def _brief(
    *,
    memories,
    contributors,
    decisions=None,
    superseded=None,
    provenance=None,
    truncated=False,
):
    return {
        "entity": "pricing",
        "kind": "concept",
        "memories": memories,
        "evidence_count": len(memories),
        # entity_brief bounds its evidence to the newest N and reports whether
        # the bound bit. The tool forwards it verbatim: a client cannot infer
        # it, because a capped brief and a complete one of exactly N carry the
        # same evidence_count.
        "truncated": truncated,
        "decisions": decisions or [],
        "superseded": superseded or [],
        "contributors": contributors,
        "provenance": provenance or {"EXTRACTED": len(memories)},
    }


def _patched(brief, *, user_id=DEV, summary_result={"summary": "a synthesis"}):
    return (
        patch.object(
            mcp_app,
            "resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch.object(mcp_app, "resolve_request_user_id", return_value=user_id),
        patch.object(mcp_app, "SessionLocal", MagicMock()),
        patch.object(mcp_app.graph_store, "entity_brief", return_value=brief),
        patch.object(
            mcp_app,
            "call_llm",
            new_callable=AsyncMock,
            return_value=summary_result,
        ),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_explore_is_in_the_tool_inventory():
    assert "explore" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 22


# ---------------------------------------------------------------------------
# not_found opacity
# ---------------------------------------------------------------------------


async def test_not_found_is_one_opaque_payload():
    """entity_brief() returning None — which covers nonexistent, other-org,
    and all-evidence-invisible identically — must surface as one payload."""
    p1, p2, p3, p4, p5 = _patched(None)
    with p1, p2, p3, p4 as brief_mock, p5:
        result = await _explore_fn()(_fake_ctx(), entity="project-mercury")

    assert result == {"error": "not_found", "entity": "project-mercury"}
    # The viewer wire: the resolved user (not None, not caller-chosen) went in.
    assert brief_mock.call_args.kwargs["viewer_user_id"] == DEV
    assert brief_mock.call_args.kwargs["org_id"] == "org-x"


async def test_unauthorized_is_a_clean_error():
    with patch.object(
        mcp_app,
        "resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await _explore_fn()(_fake_ctx(), entity="pricing")
    assert result["error"] == "unauthorized"


# ---------------------------------------------------------------------------
# The brief payload
# ---------------------------------------------------------------------------


async def test_brief_shape_and_inferred_summary():
    memories = [
        _record(content="flat pricing shipped", user_id=PRIYA, memory_type="decision"),
        _record(content="pricing page redesigned", user_id=DEV),
    ]
    brief = _brief(
        memories=memories,
        contributors=2,
        decisions=[
            {
                "id": str(memories[0].id),
                "content": "flat pricing shipped",
                "created_at": "2026-08-01T00:00:00+00:00",
                "provenance": "EXTRACTED",
            }
        ],
        superseded=[
            {
                "old": "o",
                "old_content": "usage pricing",
                "by": "n",
                "when": "2026-07-01T00:00:00+00:00",
            }
        ],
        provenance={"EXTRACTED": 1, "INFERRED": 1},
    )
    p1, p2, p3, p4, p5 = _patched(brief, user_id=DEV)
    with p1, p2, p3, p4, p5:
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert result["entity"] == "pricing"
    assert result["kind"] == "concept"
    assert result["summary"] == "a synthesis"
    assert result["summary_provenance"] == "INFERRED", (
        "a synthesized summary is a deduction and must say so"
    )
    assert result["decisions"][0]["content"] == "flat pricing shipped"
    assert result["superseded"][0]["old_content"] == "usage pricing"
    assert result["provenance"] == {"EXTRACTED": 1, "INFERRED": 1}
    assert result["evidence_count"] == 2


async def test_contributors_are_counts_and_never_ids():
    memories = [
        _record(content="a", user_id=PRIYA),
        _record(content="b", user_id=DEV),
        _record(content="c", user_id=str(uuid.uuid4())),
    ]
    p1, p2, p3, p4, p5 = _patched(
        _brief(memories=memories, contributors=3), user_id=DEV
    )
    with p1, p2, p3, p4, p5:
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert result["contributors"] == ["you", "2 teammates"]
    serialized = json.dumps(result)
    assert PRIYA not in serialized, "another person's user id leaked"
    assert DEV not in serialized, "user ids never appear, not even the caller's"


async def test_contributors_without_the_caller_is_teammates_only():
    memories = [_record(content="a", user_id=PRIYA)]
    p1, p2, p3, p4, p5 = _patched(
        _brief(memories=memories, contributors=1), user_id=DEV
    )
    with p1, p2, p3, p4, p5:
        result = await _explore_fn()(_fake_ctx(), entity="pricing")
    assert result["contributors"] == ["1 teammate"]


async def test_truncation_is_forwarded_to_the_caller():
    """A capped brief must say so; the client cannot work it out otherwise.

    The store bounds evidence to the newest N. Every count in the payload —
    contributors, provenance, decisions — is then computed over that window,
    and evidence_count is identical for a capped entity and a complete one of
    exactly N. So the flag has to travel, in both states.
    """
    memories = [_record(content="a", user_id=PRIYA)]

    p1, p2, p3, p4, p5 = _patched(
        _brief(memories=memories, contributors=1, truncated=True)
    )
    with p1, p2, p3, p4, p5:
        capped = await _explore_fn()(_fake_ctx(), entity="pricing")

    p1, p2, p3, p4, p5 = _patched(
        _brief(memories=memories, contributors=1, truncated=False)
    )
    with p1, p2, p3, p4, p5:
        whole = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert capped["truncated"] is True
    assert whole["truncated"] is False, (
        "positive control: the field is forwarded, not hardcoded"
    )


async def test_llm_failure_degrades_to_a_summaryless_brief():
    memories = [_record(content="a", user_id=PRIYA)]
    p1, p2, p3, p4, p5 = _patched(
        _brief(memories=memories, contributors=1), summary_result=None
    )
    with p1, p2, p3, p4, p5:
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert "error" not in result
    assert result["summary"] is None
    assert result["summary_provenance"] is None
    assert result["evidence_count"] == 1


async def test_raw_memory_rows_are_not_dumped_into_the_payload():
    """The brief is the product; the evidence rows stay server-side. (The
    caller can recall them; explore answering with raw rows would just be a
    second recall with a worse ladder review surface.)"""
    memories = [_record(content="sentinel-content-xyz", user_id=PRIYA)]
    p1, p2, p3, p4, p5 = _patched(_brief(memories=memories, contributors=1))
    with p1, p2, p3, p4, p5:
        result = await _explore_fn()(_fake_ctx(), entity="pricing")
    assert "memories" not in result


# ---------------------------------------------------------------------------
# Metering and audit — explore is a read path over user memory
#
# `recall` checks the org's quota, records the usage and writes a
# brain_recall_events row. `explore` did none of the three while calling an
# LLM over up to 20 memory contents per invocation: unbounded spend, and the
# one read path with no audit trail at all.
# ---------------------------------------------------------------------------


def _metered(brief, *, user_id=DEV, summary_result={"summary": "a synthesis"}, db=None):
    """`_patched` plus the metering seams, each individually assertable."""
    session_local = MagicMock()
    if db is not None:
        session_local.return_value = db
    return (
        patch.object(
            mcp_app,
            "resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch.object(mcp_app, "resolve_request_user_id", return_value=user_id),
        patch.object(mcp_app, "SessionLocal", session_local),
        patch.object(mcp_app.graph_store, "entity_brief", return_value=brief),
        patch.object(
            mcp_app, "call_llm", new_callable=AsyncMock, return_value=summary_result
        ),
    )


async def test_explore_is_metered_against_the_recall_quota():
    """A successful explore must consume recall allowance, like recall does."""
    brief = _brief(memories=[_record(content="a", user_id=PRIYA)], contributors=1)
    p1, p2, p3, p4, p5 = _metered(brief)
    with (
        p1,
        p2,
        p3,
        p4,
        p5,
        patch.object(mcp_app, "check_memory_quota", return_value=True) as quota,
        patch.object(mcp_app, "record_usage") as usage,
        patch.object(mcp_app, "log_recall_event"),
    ):
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert "error" not in result
    assert quota.call_count == 1, "explore never checked the org's quota"
    assert quota.call_args.args[1] == "org-x"
    assert quota.call_args.args[3] == "recall", "explore must meter as a recall"
    assert usage.call_count == 1, "explore never recorded its usage"
    assert usage.call_args.args[1:] == ("org-x", "recall")
    # over_cap must be carried through, not re-derived or dropped.
    assert usage.call_args.kwargs["over_cap"] is True


async def test_explore_over_quota_is_blocked_before_the_read_and_the_llm():
    """The block must land BEFORE the graph read and the LLM call.

    Otherwise the quota is a receipt for spend already incurred, which is the
    opposite of a cap.
    """
    brief = _brief(memories=[_record(content="a", user_id=PRIYA)], contributors=1)
    p1, p2, p3, p4, p5 = _metered(brief)
    with (
        p1,
        p2,
        p3,
        p4 as brief_mock,
        p5 as llm,
        patch.object(
            mcp_app,
            "check_memory_quota",
            side_effect=mcp_app.MemoryQuotaExceeded(
                metric="recall", limit=1000, plan_used=1000, suggested_plan="pro"
            ),
        ),
        patch.object(mcp_app, "record_usage") as usage,
    ):
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert result["error"] == "quota_exceeded"
    assert result["suggested_plan"] == "pro"
    assert brief_mock.call_count == 0, "a capped org still read the graph"
    assert llm.call_count == 0, "a capped org still spent an LLM call"
    assert usage.call_count == 0, "a blocked call must not consume allowance"


async def test_explore_writes_a_recall_audit_event():
    """Every read of user memory leaves a trace. explore left none."""
    brief = _brief(
        memories=[_record(content="a", user_id=PRIYA), _record(content="b")],
        contributors=1,
    )
    p1, p2, p3, p4, p5 = _metered(brief)
    with (
        p1,
        p2,
        p3,
        p4,
        p5,
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage"),
        patch.object(mcp_app, "log_recall_event") as audit,
    ):
        await _explore_fn()(_fake_ctx(), entity="pricing")

    assert audit.call_count == 1, "explore wrote no audit row"
    kwargs = audit.call_args.kwargs
    assert kwargs["org_id"] == "org-x"
    assert kwargs["agent_id"] == "agent-y"
    assert kwargs["query_text"] == "pricing", "the audit must say WHAT was read"
    assert kwargs["result_count"] == 2, "the audit must say how much was returned"
    assert "explore" in kwargs["source"], (
        f"the audit must name the read path; got {kwargs['source']!r}"
    )


async def test_a_not_found_explore_is_audited_too():
    """not_found is the enumeration path — the one that most needs a trace.

    A caller probing entity names to find which exist would otherwise leave
    the audit table empty.
    """
    p1, p2, p3, p4, p5 = _metered(None)
    with (
        p1,
        p2,
        p3,
        p4,
        p5,
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage") as usage,
        patch.object(mcp_app, "log_recall_event") as audit,
    ):
        result = await _explore_fn()(_fake_ctx(), entity="project-mercury")

    assert result == {"error": "not_found", "entity": "project-mercury"}
    assert audit.call_count == 1, "a probe for a nonexistent entity left no trace"
    assert audit.call_args.kwargs["query_text"] == "project-mercury"
    assert audit.call_args.kwargs["result_count"] == 0
    assert usage.call_count == 1, "a probe still costs the org a recall"


async def test_an_audit_write_failure_never_breaks_explore():
    """Preserves recall's property: the audit swallows and rolls back.

    Routed through the same `log_recall_event` helper rather than a parallel
    INSERT, so a failing audit cannot take down the read — or the billing
    write that follows it.
    """
    db = MagicMock()
    db.execute.side_effect = RuntimeError("audit table is on fire")
    brief = _brief(memories=[_record(content="a", user_id=PRIYA)], contributors=1)
    p1, p2, p3, p4, p5 = _metered(brief, db=db)
    with (
        p1,
        p2,
        p3,
        p4,
        p5,
        patch.object(mcp_app, "check_memory_quota", return_value=False),
    ):
        result = await _explore_fn()(_fake_ctx(), entity="pricing")

    assert "error" not in result, "a failed audit write broke the read"
    assert result["evidence_count"] == 1
    assert db.rollback.called, "the audit must roll back the poisoned transaction"
