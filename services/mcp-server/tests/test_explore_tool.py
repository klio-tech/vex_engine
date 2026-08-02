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


def _brief(*, memories, contributors, decisions=None, superseded=None, provenance=None):
    return {
        "entity": "pricing",
        "kind": "concept",
        "memories": memories,
        "evidence_count": len(memories),
        "decisions": decisions or [],
        "superseded": superseded or [],
        "contributors": contributors,
        "provenance": provenance or {"EXTRACTED": len(memories)},
    }


def _patched(brief, *, user_id=DEV, summary_result={"summary": "a synthesis"}):
    return (
        patch.object(mcp_app, "resolve_principal", return_value=("org-x", "agent-y")),
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
    assert len(EXPECTED_TOOL_NAMES) == 18


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
        "resolve_principal",
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
