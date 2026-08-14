"""Unit tests for shared.claims: the pure-Python half of the claim contract.

Scope, deliberately: a MagicMock ``db`` can prove the *shape* of what gets
sent to Postgres (SQL text, params, return handling) and the branch decisions
that live entirely in Python (embed-failure fallback, UUID validation on
release). It cannot prove tenancy isolation or the pgvector similarity math
land correctly against a real server -- that lives in
``services/migrations/tests/test_044_work_claims.py`` (DB-gated) alongside
this repo's other schema/isolation split (see ``test_forget_tool.py``'s
docstring for the same division of labour on ``retract_memory``).
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import claims


@pytest.mark.anyio
async def test_create_claim_returns_semantic_collisions():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = [
        {
            "id": "c-old",
            "description": "refactor auth",
            "agent_id": "B",
            "similarity": 0.91,
        }
    ]
    with patch.object(claims, "embed_text", new=AsyncMock(return_value=[0.1] * 4)):
        result = await claims.create_claim(
            db, org_id="o", description="fix login token expiry", agent_id="A"
        )
    assert result["collisions"][0]["agent_id"] == "B"
    sql = " ".join(str(db.execute.call_args_list[-1].args[0]).split())
    assert "released_at IS NULL" in sql and "expires_at > now()" in sql
    assert "org_id = :org_id" in sql


@pytest.mark.anyio
async def test_create_claim_without_embedding_still_registers():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []
    with patch.object(claims, "embed_text", new=AsyncMock(return_value=None)):
        result = await claims.create_claim(
            db, org_id="o", description="x", agent_id="A"
        )
    assert result["claim_id"]
    assert result["collisions"] == []
    # No collision query was ever issued -- only the INSERT.
    assert db.execute.call_count == 1


@pytest.mark.anyio
async def test_create_claim_embed_raising_still_registers():
    """embed_text is documented to never raise (it catches its own backend
    errors and returns None), but create_claim guards the case anyway. A
    raise must degrade exactly like a None return: the claim still
    registers, with no collision detection."""
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []
    with patch.object(
        claims, "embed_text", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        result = await claims.create_claim(
            db, org_id="o", description="x", agent_id="A"
        )
    assert result["claim_id"]
    assert result["collisions"] == []


@pytest.mark.anyio
async def test_create_claim_insert_carries_clamped_ttl():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []
    with patch.object(claims, "embed_text", new=AsyncMock(return_value=None)):
        await claims.create_claim(
            db, org_id="o", description="x", agent_id="A", ttl_minutes=9999
        )
    insert_params = db.execute.call_args_list[0].args[1]
    assert insert_params["ttl"] == claims._MAX_TTL_MINUTES


def test_release_is_owner_scoped():
    db = MagicMock()
    db.execute.return_value.rowcount = 1
    assert claims.release_claim(
        db, org_id="o", claim_id=str(uuid.uuid4()), agent_id="A"
    )
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    assert "agent_id = :agent_id" in sql and "org_id = :org_id" in sql


def test_release_rejects_malformed_claim_id_before_any_query():
    db = MagicMock()
    assert (
        claims.release_claim(db, org_id="o", claim_id="not-a-uuid", agent_id="A")
        is False
    )
    db.execute.assert_not_called()


def test_release_no_matching_row_returns_false():
    db = MagicMock()
    db.execute.return_value.rowcount = 0
    assert (
        claims.release_claim(db, org_id="o", claim_id=str(uuid.uuid4()), agent_id="A")
        is False
    )


def test_live_claims_query_is_org_scoped_and_excludes_embedding():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = [
        {
            "id": "c1",
            "description": "fix login",
            "agent_id": "A",
            "user_id": None,
            "expires_at": "2026-08-13T00:00:00Z",
        }
    ]
    result = claims.live_claims(db, org_id="o")
    assert result[0]["id"] == "c1"
    assert "embedding" not in result[0]
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    assert "org_id = :org_id" in sql
    assert "released_at IS NULL" in sql and "expires_at > now()" in sql
