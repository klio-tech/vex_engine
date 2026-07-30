"""Live-stack proof: agent A stores, agent B discovers and reads.

Requires postgres and minio from docker-compose (migrations at head, plus the
``klio-artifacts`` bucket) and is entirely skipped unless ``KLIO_E2E`` is set,
so CI without a stack stays green::

    docker compose up -d postgres minio
    (from services/migrations) DATABASE_URL=... alembic upgrade head
    (from services/mcp-server) KLIO_E2E=1 pytest tests/test_artifact_e2e.py -v

Everything except one seam is real: the tools are invoked through the actual
FastMCP tool registry, auth resolves through the real ``KeyValidator`` against
a throwaway org + API key this module inserts directly into ``organizations``
(see ``_e2e_org`` below), every DB statement hits the real Postgres started by
docker-compose, and object storage hits the real MinIO ``klio-artifacts``
bucket.

The one patched seam is ``shared.memory.embed_text``: this dev environment has
no live embedding-provider credentials (no ``TOGETHER_API_KEY`` /
``LITELLM_API_URL`` — see ``.env.example``), and the embedding backend is an
external paid API this test suite must not depend on to run. This mirrors the
*existing* precedent in ``tests/test_e2e.py`` (its module docstring: "Embeddings
... FIXED 1024-element vector ... All of write_memory / find_similar_memory /
retrieve_memories_scoped call embed_text"). Patching it does not weaken this
test: the fixed vector still flows through a REAL pgvector column, a REAL
cosine-distance ORDER BY, and a REAL org-scoped WHERE clause — the exact
mechanics of discovery this task exists to prove. Only the external LLM call
itself is stubbed.

Everything else in this module is genuinely live:

* ``test_agent_b_recalls_an_artifact_agent_a_wrote`` — the brief's core
  scenario: agent A stores, agent B (a DIFFERENT agent, never told the
  artifact exists) discovers it via semantic recall, reads its text, and gets
  a presigned download URL.
* ``test_gc_guarded_delete_against_real_postgres`` — exercises the per-row
  ``WHERE id = :id AND status = 'pending' RETURNING id`` guarded DELETE (a new
  idiom with no other precedent in this codebase) against real Postgres: one
  abandoned upload is collected (row + object both gone), one that was
  confirmed in the same instant the sweep ran survives untouched (row AND
  object both still there).
* ``test_check_constraint_rejects_invalid_body`` — proves
  ``ck_artifacts_one_body`` is actually enforced by Postgres, not just present
  in a migration file: activating a row with both bodies, or neither, is
  rejected by a real INSERT/UPDATE.
* ``test_retention_excludes_active_artifact_card`` — backdates both an
  artifact card and an ordinary memory row past a 1-day retention window, runs
  the real ``enforce_memory_retention()`` SQL function, and asserts the plain
  memory row is swept while the artifact card survives (migrations 028/029) —
  proving the exclusion actually fires, not merely that nothing old enough to
  test was ever created.
"""

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_E2E"), reason="set KLIO_E2E=1 with a live stack"
)

# A fixed embedding vector of the exact column dimension (vector(1024)). See
# the module docstring: this is the same seam test_e2e.py patches, for the
# same reason (no live embedding-provider credentials in this dev environment).
_FIXED_EMBEDDING = [0.1] * 1024


# ---------------------------------------------------------------------------
# Throwaway org + dev API key, provisioned directly against local Postgres.
# ---------------------------------------------------------------------------


@pytest.fixture
def _e2e_org():
    """Insert a throwaway org + a real ``memory``-scoped API key.

    ``KeyValidator`` (``shared.auth``) reads keys from the ``organizations
    .api_keys`` JSONB column — there is no separate ``api_keys`` table. A dev
    key for a real org is normally minted by the (separate, closed) dashboard
    repo; for a local live-stack run we provision one directly, the same way
    the dashboard would: hash the raw key, insert the JSONB entry with the
    ``memory`` scope (mcp-server's ``KeyValidator`` requires it — see
    ``app.auth.get_validator``), and let ``resolve_principal_plan`` validate it
    through the exact same code path a production request would hit.

    Respects a caller-supplied ``KLIO_E2E_KEY`` / org: if the environment
    already carries a real dev key (e.g. one provisioned by hand against a
    shared dev org, as the task brief describes), that key is used as-is and
    this fixture skips provisioning and teardown for it. Otherwise it
    provisions its own throwaway org + key so the test is runnable standalone
    from a bare ``docker compose up -d postgres minio``.

    Cleans up every row it created (organizations, artifacts, session_memories)
    in a ``finally`` so a failed test never leaves residue in the shared local
    DB. Never touches any org this fixture did not create.
    """
    from app.db import SessionLocal

    org_id = f"e2e-artifacts-{uuid.uuid4()}"
    raw_key = f"klio_e2e_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    externally_provided = bool(os.environ.get("KLIO_E2E_KEY"))
    prev_key = os.environ.get("KLIO_E2E_KEY")

    db = SessionLocal()
    try:
        if not externally_provided:
            db.execute(
                text(
                    """
                    INSERT INTO organizations (org_id, name, api_keys, plan)
                    VALUES (
                        :org_id, :name,
                        CAST(:api_keys AS jsonb),
                        'free'
                    )
                    """
                ),
                {
                    "org_id": org_id,
                    "name": "Klio artifact E2E throwaway org",
                    "api_keys": (
                        '[{"id": "e2e-key", "key_hash": "%s", "scopes": ["memory"], '
                        '"revoked": false}]' % key_hash
                    ),
                },
            )
            db.commit()
            os.environ["KLIO_E2E_KEY"] = raw_key

        # Reset the module-level KeyValidator singleton so a validator built
        # (and cached) by an earlier test module never masks this org's key,
        # and so this org's key is looked up fresh against the DB we just
        # wrote to.
        import app.auth as auth_mod

        if auth_mod._validator is not None:
            auth_mod._validator.close()
        auth_mod._validator = None

        yield org_id
    finally:
        try:
            db.execute(
                text("DELETE FROM session_memories WHERE org_id = :org_id"),
                {"org_id": org_id},
            )
            db.execute(
                text("DELETE FROM artifacts WHERE org_id = :org_id"),
                {"org_id": org_id},
            )
            if not externally_provided:
                db.execute(
                    text("DELETE FROM organizations WHERE org_id = :org_id"),
                    {"org_id": org_id},
                )
            db.commit()

            # Every object key is prefixed with the org id (see
            # shared.artifact_storage.object_key), so this test's real bytes
            # in the real klio-artifacts bucket (inline artifacts materialised
            # for a download URL, GC-test uploads, etc.) never outlive the
            # test even though there is no S3 "delete by prefix" primitive.
            from shared import artifact_storage

            s3 = artifact_storage.get_artifact_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=artifact_storage.ARTIFACT_BUCKET, Prefix=f"{org_id}/"
            ):
                for obj in page.get("Contents", []):
                    try:
                        s3.delete_object(
                            Bucket=artifact_storage.ARTIFACT_BUCKET, Key=obj["Key"]
                        )
                    except Exception:
                        pass
        finally:
            db.close()
            if not externally_provided:
                if prev_key is None:
                    os.environ.pop("KLIO_E2E_KEY", None)
                else:
                    os.environ["KLIO_E2E_KEY"] = prev_key

            import app.auth as auth_mod

            if auth_mod._validator is not None:
                auth_mod._validator.close()
            auth_mod._validator = None


@pytest.fixture(autouse=True)
def _patched_embeddings():
    """Patch the one non-deterministic, credential-gated seam.

    See the module docstring for why this mirrors ``tests/test_e2e.py``'s
    established pattern rather than weakening the test.
    """
    embed_mock = AsyncMock(return_value=list(_FIXED_EMBEDDING))
    with patch("shared.memory.embed_text", embed_mock):
        yield


def _ctx(agent: str):
    """Build the minimal Context stand-in the tool bodies actually read.

    Matches the shape used across the mcp-server unit tests: a nested
    ``SimpleNamespace`` whose ``.request_context.request.headers`` is a real
    dict. ``ctx.session`` is deliberately absent — ``_attach_client_name``
    swallows the resulting ``AttributeError``, exactly as it does for MCP
    clients that never send an ``initialize`` clientInfo.
    """
    # NOTE: header keys are exact-case here — unlike a real transport request
    # (whose ``.headers`` is a case-insensitive Starlette ``Headers``), this
    # stand-in's ``.headers`` is a plain ``dict``, and ``verify_api_key`` reads
    # it with ``.get("X-Vex-Key")``. This matches the casing every other
    # direct-``.fn``-call test in this suite uses (e.g.
    # ``test_write_tools.py``'s ``_fake_ctx``), not the brief's illustrative
    # lowercase pseudocode, which would 401 against a plain dict.
    return SimpleNamespace(
        request_context=SimpleNamespace(
            request=SimpleNamespace(
                headers={"X-Vex-Key": os.environ["KLIO_E2E_KEY"], "X-Vex-Agent": agent}
            )
        )
    )


# ---------------------------------------------------------------------------
# 1. The brief's core scenario.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_b_recalls_an_artifact_agent_a_wrote(_e2e_org):
    from app.mcp_app import build_mcp_server

    server = build_mcp_server()
    put = server._tool_manager.get_tool("artifact_put").fn
    get = server._tool_manager.get_tool("artifact_get").fn
    recall = server._tool_manager.get_tool("recall").fn

    marker = uuid.uuid4().hex[:8]

    stored = await put(
        _ctx("agent-a"),
        title=f"Rate limiter design {marker}",
        summary=f"Token bucket at the edge, decision {marker}",
        kind="document",
        content="# Rate limiting\n\nToken bucket, 100 rps per key.",
    )
    assert stored["stored"] is True, stored

    # Agent B has never seen this artifact and searches semantically.
    found = await recall(_ctx("agent-b"), query="how do we rate limit", limit=25)
    cards = [
        m
        for m in found.get("memories", [])
        if m.get("memory_type") == "artifact" and marker in m.get("content", "")
    ]
    assert cards, f"agent B could not discover agent A's artifact through recall: {found!r}"

    artifact_id = cards[0]["metadata"]["artifact_id"]
    body = await get(_ctx("agent-b"), artifact_id=artifact_id)
    assert "Token bucket" in body["text"], body

    link = await get(_ctx("agent-b"), artifact_id=artifact_id, mode="url")
    assert link["url"].startswith("http"), link
    assert link["expires_in"] == 900


# ---------------------------------------------------------------------------
# 2. GC guarded delete: the per-row `RETURNING id` idiom, for real.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gc_guarded_delete_against_real_postgres(_e2e_org):
    """Two pending rows, both past the GC's age threshold at sweep time.

    One is a genuine abandoned upload: the sweep must delete its row AND its
    S3 object. The other is confirmed (flipped to 'active') in the instant
    before the sweep runs, simulating a confirm landing mid-sweep: the sweep's
    guarded ``DELETE ... WHERE status = 'pending'`` must decline to remove it,
    and — critically — must never touch its object, because the DELETE
    confirming row-removal is what gates the object delete in
    ``collect_artifacts_once``.
    """
    from app.db import SessionLocal
    from app.scheduler import collect_artifacts_once
    from shared import artifact_storage, artifacts

    org_id = _e2e_org
    db = SessionLocal()
    s3 = artifact_storage.get_artifact_s3_client()

    try:
        # -- Row 1: a genuinely abandoned pending upload. ------------------
        abandoned_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id="agent-a",
            title="Abandoned upload",
            summary="Never confirmed",
            kind="file",
            mime_type="application/octet-stream",
        )
        abandoned_key = artifact_storage.object_key(org_id, abandoned_id, 1)
        artifact_storage.put_text(s3, abandoned_key, "orphaned bytes", "text/plain")

        # -- Row 2: confirmed mid-sweep (simulated by activating it BEFORE
        # the sweep runs, but after it was created old enough to have been a
        # GC candidate had it stayed pending).
        confirmed_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id="agent-a",
            title="Confirmed just in time",
            summary="Activated before the sweep reached it",
            kind="file",
            mime_type="text/plain",
        )
        confirmed_key = artifact_storage.object_key(org_id, confirmed_id, 1)
        measured = artifact_storage.put_text(
            s3, confirmed_key, "confirmed bytes", "text/plain"
        )

        # Backdate BOTH rows' created_at past the 24h abandoned-upload
        # threshold so `collect_abandoned`'s SELECT would treat either as a
        # candidate; this is what actually exercises the age filter for real,
        # rather than trusting a TTL constant never gets checked.
        old_created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        db.execute(
            text("UPDATE artifacts SET created_at = :ts WHERE id IN (:a, :b)"),
            {"ts": old_created_at, "a": abandoned_id, "b": confirmed_id},
        )
        db.commit()

        # Row 2 gets confirmed now — real activate(), real CHECK constraints,
        # real commit — landing "mid-sweep" relative to collect_abandoned's
        # SELECT, which has not run yet.
        activated = artifacts.activate(
            db,
            org_id=org_id,
            artifact_id=confirmed_id,
            size_bytes=measured["size_bytes"],
            checksum=measured["checksum"],
            inline_text=None,
            storage_uri=artifact_storage.storage_uri(org_id, confirmed_id, 1),
        )
        assert activated is not None

        collected = await collect_artifacts_once(lambda: SessionLocal())

        assert collected >= 1

        # Row 1: gone, and its object is gone too.
        remaining = db.execute(
            text("SELECT status FROM artifacts WHERE id = :id"),
            {"id": abandoned_id},
        ).fetchone()
        assert remaining is None, "abandoned pending row must be deleted"
        assert (
            artifact_storage.head(s3, abandoned_key) is None
        ), "abandoned row's object must be deleted"

        # Row 2: survives, status still 'active', object untouched.
        surviving = db.execute(
            text("SELECT status FROM artifacts WHERE id = :id"),
            {"id": confirmed_id},
        ).fetchone()
        assert surviving is not None, "row confirmed mid-sweep must survive"
        assert surviving[0] == "active"
        assert (
            artifact_storage.head(s3, confirmed_key) is not None
        ), "confirmed row's object must NOT be deleted by the sweep"
    finally:
        for key in (abandoned_key, confirmed_key):
            try:
                artifact_storage.delete(s3, key)
            except Exception:
                pass
        db.close()


# ---------------------------------------------------------------------------
# 3. ck_artifacts_one_body: real Postgres, real rejection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_constraint_rejects_invalid_body(_e2e_org):
    """An 'active' row must have EXACTLY ONE of inline_text / storage_uri.

    Both set, and neither set, must each be rejected by Postgres itself
    (``ck_artifacts_one_body``, migration 027) — not merely by application
    discipline in ``shared.artifacts.activate``.
    """
    from app.db import SessionLocal
    from shared import artifacts

    org_id = _e2e_org
    db = SessionLocal()
    try:
        both_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id="agent-a",
            title="Both bodies",
            summary="Should be rejected",
            kind="record",
            mime_type="text/plain",
        )
        with pytest.raises(IntegrityError, match="ck_artifacts_one_body"):
            db.execute(
                text(
                    """
                    UPDATE artifacts SET
                        status = 'active', size_bytes = 10, checksum = 'x',
                        inline_text = 'both', storage_uri = 's3://klio-artifacts/x'
                    WHERE id = :id
                    """
                ),
                {"id": both_id},
            )
            db.commit()
        db.rollback()

        neither_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id="agent-a",
            title="Neither body",
            summary="Should also be rejected",
            kind="record",
            mime_type="text/plain",
        )
        with pytest.raises(IntegrityError, match="ck_artifacts_one_body"):
            db.execute(
                text(
                    """
                    UPDATE artifacts SET
                        status = 'active', size_bytes = 10, checksum = 'x'
                    WHERE id = :id
                    """
                ),
                {"id": neither_id},
            )
            db.commit()
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. Retention excludes an active artifact's card (migrations 028/029).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retention_excludes_active_artifact_card(_e2e_org):
    """A 1-day retention sweep must skip an active artifact's card while
    deleting an ordinary memory row of the same age — proving the exclusion
    in ``enforce_memory_retention`` actually fires (not merely that nothing
    old enough to sweep existed).
    """
    from app.db import SessionLocal
    from app.mcp_app import build_mcp_server
    from shared import artifacts

    org_id = _e2e_org
    db = SessionLocal()
    old_ts = datetime.now(timezone.utc) - timedelta(days=2)

    try:
        server = build_mcp_server()
        put = server._tool_manager.get_tool("artifact_put").fn
        remember = server._tool_manager.get_tool("remember").fn

        marker = uuid.uuid4().hex[:8]
        stored = await put(
            _ctx("agent-a"),
            title=f"Retention-survivor {marker}",
            summary="Must outlive a 1-day retention window while active",
            kind="document",
            content=f"content {marker}",
        )
        assert stored["stored"] is True, stored
        artifact_id = stored["artifact_id"]

        # A different agent than the artifact_put call above: brain_session_id
        # derives the session from (org_id, agent_id) when no mcp-session-id
        # header is sent, and the fixed-embedding seam (module docstring) makes
        # every write's vector cosine-identical. Reusing "agent-a"'s session
        # here would make `remember`'s dedup check find the just-written
        # artifact card as a "similar" existing memory and skip the write
        # entirely — a collision that is an artifact of the patched embedding,
        # not something a real embedding backend would do (mirrors test_e2e.py's
        # documented dedup caveat).
        remembered = await remember(_ctx("agent-c"), content=f"plain fact {marker}")
        assert remembered["stored"] is True, remembered
        plain_memory_id = remembered["id"]

        # Backdate both rows so the retention window actually has to decide
        # something, instead of nothing being old enough to sweep.
        db.execute(
            text(
                "UPDATE session_memories SET created_at = :ts "
                "WHERE org_id = :org_id AND id IN (:card, :plain)"
            ),
            {
                "ts": old_ts,
                "org_id": org_id,
                "card": _artifact_card_id(db, org_id=org_id, artifact_id=artifact_id),
                "plain": plain_memory_id,
            },
        )
        db.commit()

        db.execute(
            text("SELECT enforce_memory_retention(:org_id, 1)"), {"org_id": org_id}
        )
        db.commit()

        card_row = db.execute(
            text(
                "SELECT id FROM session_memories "
                "WHERE org_id = :org_id AND memory_type = 'artifact' "
                "AND metadata->>'artifact_id' = :artifact_id"
            ),
            {"org_id": org_id, "artifact_id": artifact_id},
        ).fetchone()
        assert card_row is not None, "artifact card must survive retention while active"

        plain_row = db.execute(
            text("SELECT id FROM session_memories WHERE id = :id"),
            {"id": plain_memory_id},
        ).fetchone()
        assert plain_row is None, (
            "an ordinary memory row of the same age must be swept — otherwise "
            "this proves nothing was old enough to test, not that the "
            "artifact-card exclusion actually fired"
        )

        # And the card must still be genuinely discoverable through recall,
        # not merely present as a row.
        recall = server._tool_manager.get_tool("recall").fn
        found = await recall(_ctx("agent-b"), query="retention survivor", limit=25)
        cards = [
            m
            for m in found.get("memories", [])
            if m.get("memory_type") == "artifact" and marker in m.get("content", "")
        ]
        assert cards, f"card survived the DB sweep but is not recall-discoverable: {found!r}"

        artifacts.retract(db, org_id=org_id, artifact_id=artifact_id)
    finally:
        db.close()


def _artifact_card_id(db, *, org_id: str, artifact_id: str) -> str:
    row = db.execute(
        text(
            "SELECT id FROM session_memories "
            "WHERE org_id = :org_id AND memory_type = 'artifact' "
            "AND metadata->>'artifact_id' = :artifact_id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    assert row is not None, "artifact_put must have written a card"
    return str(row[0])
