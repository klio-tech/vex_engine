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
external paid API this test suite must not depend on to run.

**Deterministic, input-dependent fake embedding (not a single fixed vector).**
An earlier version of this module patched ``embed_text`` to always return the
SAME vector (mirroring ``tests/test_e2e.py``'s precedent). Code review
correctly flagged that this made the central "agent B discovers agent A's
artifact through recall" assertion unfalsifiable in principle: with only 1-2
rows in a fresh throwaway org and every vector identical, `assert cards` would
pass even if the ``ORDER BY embedding <=> :embedding`` were removed, reversed,
or the blended-score re-rank were completely broken — there was nothing to
discriminate against. ``_hash_embedding`` below fixes this: it derives a
deterministic vector from a SHA-256 hash of the input text (same text -> same
vector, always; different text -> a different, effectively uncorrelated
vector). It is NOT a semantic embedding — two rows about the same topic in
different words land at unrelated vectors — so the tests below route around
that by making a query's text an EXACT match for the target row's own stored
content, while distractor rows carry deliberately different, unrelated text.
That is what makes the ``ORDER BY`` genuinely decide the outcome: the target's
distance is exactly 0, every distractor's is not.

Everything else in this module is genuinely live:

* ``test_agent_b_recalls_an_artifact_agent_a_wrote`` — the brief's core
  scenario: several unrelated memories are planted in the org first (so the
  candidate pool recall must rank against is real, not "the only 1-2 rows
  that exist"), agent A then stores an artifact, and agent B (a DIFFERENT
  agent, never told the artifact exists) must retrieve it under a `limit`
  below the total row count — i.e. it must be SELECTED, not merely
  INCLUDED — reads its text, and gets a presigned download URL. A negative
  assertion confirms no unrelated distractor makes it into the small,
  competitive result set.
* ``test_gc_guarded_delete_against_real_postgres`` — exercises the per-row
  ``WHERE id = :id AND status = 'pending' RETURNING id`` guarded DELETE (a new
  idiom with no other precedent in this codebase) against real Postgres: one
  abandoned upload is collected (row + object both gone), one that was
  confirmed in the same instant the sweep ran survives untouched (row AND
  object both still there).
* ``test_forget_then_sweep_actually_removes_the_bytes_from_minio`` — the one
  assertion that makes "forget really deletes" real rather than claimed. A
  retracted artifact's and a superseded version's objects must be absent from
  the real MinIO bucket after the purge pass, while an ``active`` artifact of
  the same age keeps its bytes. Mocked S3 cannot prove this: it would report a
  delete that never left the process.
* ``test_purge_collects_an_inline_artifacts_materialised_object`` — the M1
  case: ``artifact_get(mode='url')`` on an inline artifact writes an object the
  row never references, and the purge pass must still collect it.
* ``test_check_constraint_rejects_invalid_body`` — proves
  ``ck_artifacts_one_body`` is actually enforced by Postgres, not just present
  in a migration file: activating a row with both bodies, or neither, is
  rejected by a real INSERT/UPDATE.
* ``test_retention_excludes_active_artifact_card`` — backdates both an
  artifact card and an ordinary memory row past a 1-day retention window, runs
  the real ``enforce_memory_retention()`` SQL function, and asserts the plain
  memory row is swept while the artifact card survives (migrations 028/029) —
  proving the exclusion actually fires, not merely that nothing old enough to
  test was ever created — then re-confirms the card is still recall-reachable
  by an exact-content-match query.

See ``.superpowers/sdd/2026-07-30-klio-artifacts/task-11-report.md`` for the
verification that this recall assertion can now actually fail: a control run
with the ranking deliberately reversed, shown failing, then restored and shown
passing.
"""

import hashlib
import math
import os
import random
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

# pgvector column is vector(1024) (see migration 017).
_EMBED_DIM = 1024


def _hash_embedding(text_input: str) -> list[float]:
    """Deterministic, input-dependent stand-in for a real embedding.

    Same text -> same vector, always (repeatable, so a query re-run behaves
    identically). Different text -> a different vector with overwhelming
    probability (1024 floats seeded from a 256-bit hash; a collision between
    two distinct texts is astronomically unlikely). This is what gives the
    ``ORDER BY embedding <=> :embedding`` something real to discriminate on —
    see the module docstring for why a single fixed vector (the prior version
    of this module) made the central recall assertion unfalsifiable.

    Deliberately NOT semantic: this has no notion of "similar meaning",
    only "identical text". The tests route around that by querying with text
    that exactly matches the target row's stored content.
    """
    seed = hashlib.sha256(text_input.encode("utf-8")).digest()
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(_EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


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

    See the module docstring: this now uses a deterministic, input-dependent
    fake embedding (``_hash_embedding``), not a single fixed vector, so
    ranking has something real to discriminate on.
    """
    embed_mock = AsyncMock(side_effect=lambda text_input: _hash_embedding(text_input))
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
# 1. The brief's core scenario, made genuinely competitive.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_b_recalls_an_artifact_agent_a_wrote(_e2e_org):
    from app.mcp_app import build_mcp_server
    from shared.artifacts import card_content

    server = build_mcp_server()
    put = server._tool_manager.get_tool("artifact_put").fn
    get = server._tool_manager.get_tool("artifact_get").fn
    recall = server._tool_manager.get_tool("recall").fn
    remember = server._tool_manager.get_tool("remember").fn

    marker = uuid.uuid4().hex[:8]

    # Plant distractors in the SAME org BEFORE the target artifact exists, so
    # the candidate pool recall ranks against is real (6 rows), not "the only
    # 1-2 rows in a fresh throwaway org". Each is deliberately unrelated in
    # content to the target, so its hash-derived vector lands far from the
    # query's.
    distractor_marker = uuid.uuid4().hex[:8]
    distractor_facts = [
        f"The office coffee machine is being replaced next week {distractor_marker}",
        f"Q3 offsite is in Denver this year {distractor_marker}",
        f"The design system's primary typeface is Inter {distractor_marker}",
        f"Marketing's campaign tracker lives in Airtable {distractor_marker}",
        f"Intern onboarding has twelve steps {distractor_marker}",
    ]
    for fact in distractor_facts:
        r = await remember(_ctx("agent-distractor"), content=fact)
        assert r["stored"] is True, r

    title = f"Rate limiter design {marker}"
    summary = f"Token bucket at the edge, decision {marker}"
    kind = "document"
    content = "# Rate limiting\n\nToken bucket, 100 rps per key."

    stored = await put(
        _ctx("agent-a"), title=title, summary=summary, kind=kind, content=content
    )
    assert stored["stored"] is True, stored

    # The query text is an EXACT match for the card row's own stored content
    # (see ``shared.artifacts.card_content`` — this is exactly what
    # ``_write_artifact_card`` writes). With a hash-derived, non-semantic
    # embedding this is what guarantees the target's vector distance is
    # exactly 0 while every distractor's unrelated text lands elsewhere —
    # what actually makes the ORDER BY decide the outcome, rather than
    # everything in a tiny org just happening to fit under `limit`.
    query = card_content(title, summary, kind, len(content.encode("utf-8")))

    # limit=1 against a 6-row org (1 target + 5 distractors): only the single
    # best-ranked row can come back at all, so the target must be SELECTED
    # under maximum competition, not merely INCLUDED because the result set
    # happened to cover every row. A random distractor's hash-derived vector
    # has expected cosine similarity ~0 against the query (observed ~0.05 in
    # practice); the target's is exactly 1.0 (identical text) — nowhere near
    # a coin flip, so this is not relying on luck to exclude the distractors.
    found = await recall(_ctx("agent-b"), query=query, limit=1)
    memories = found.get("memories", [])
    assert len(memories) <= 1, found

    cards = [
        m
        for m in memories
        if m.get("memory_type") == "artifact" and marker in m.get("content", "")
    ]
    assert cards, f"agent B could not discover agent A's artifact through recall: {found!r}"

    # Negative assertion: an unrelated distractor must NOT be in the small,
    # competitive result set. Without this, a ranking that returns
    # "everything" (or an arbitrary subset with no real ordering) could still
    # satisfy the assertion above by coincidence.
    returned_contents = " ".join(m.get("content", "") for m in memories)
    assert distractor_marker not in returned_contents, (
        "an unrelated distractor made it into a small, competitive recall "
        f"result — ranking is not actually discriminating: {found!r}"
    )

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
# 2b. forget really deletes: the object is GONE from MinIO after the sweep.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_then_sweep_actually_removes_the_bytes_from_minio(_e2e_org):
    """The assertion that makes "forget really deletes" real rather than claimed.

    ``forget`` only sets ``status='retracted'``. Every observable surface then
    says the artifact is gone — the card is hidden, ``artifact_get`` returns
    not_found, ``org_storage_bytes`` stops counting it — while the object it
    points at stays in the bucket. The purge pass is what closes that gap, and
    nothing short of asking real object storage whether the bytes are still
    there can prove it: a mocked S3 would happily report a delete that never
    reached MinIO.

    Both the retracted and the superseded case are covered, plus the negative
    control: an artifact that is still ``active`` must NOT lose its bytes to
    the same sweep.
    """
    from app.db import SessionLocal
    from app.mcp_app import build_mcp_server
    from app.scheduler import collect_artifacts_once
    from shared import artifact_storage, artifacts

    org_id = _e2e_org
    db = SessionLocal()
    s3 = artifact_storage.get_artifact_s3_client()
    keys = []

    try:
        server = build_mcp_server()
        forget = server._tool_manager.get_tool("forget").fn
        get = server._tool_manager.get_tool("artifact_get").fn

        def _upload(title: str, body: str) -> tuple[str, str]:
            """Create a real active, S3-backed artifact and return (id, key)."""
            artifact_id = artifacts.create_pending(
                db,
                org_id=org_id,
                agent_id="agent-a",
                title=title,
                summary=f"{title} — live-stack purge fixture",
                kind="file",
                mime_type="text/plain",
            )
            key = artifact_storage.object_key(org_id, artifact_id, 1)
            keys.append(key)
            measured = artifact_storage.put_text(s3, key, body, "text/plain")
            activated = artifacts.activate(
                db,
                org_id=org_id,
                artifact_id=artifact_id,
                size_bytes=measured["size_bytes"],
                checksum=measured["checksum"],
                inline_text=None,
                storage_uri=artifact_storage.storage_uri(org_id, artifact_id, 1),
            )
            assert activated is not None
            # A card, so the "card must not resurface" assertion has something
            # real to check.
            db.execute(
                text(
                    """
                    INSERT INTO session_memories
                        (org_id, agent_id, session_id, content, memory_type,
                         metadata, scope)
                    VALUES
                        (:org_id, 'agent-a', :session_id, :content, 'artifact',
                         CAST(:metadata AS jsonb), 'org')
                    """
                ),
                {
                    "org_id": org_id,
                    "session_id": f"purge-{artifact_id}",
                    "content": f"[artifact] {title}",
                    "metadata": '{"artifact_id": "%s", "source": "mcp"}' % artifact_id,
                },
            )
            db.commit()
            return artifact_id, key

        forgotten_id, forgotten_key = _upload("Forgotten PDF", "customer bytes")
        superseded_id, superseded_key = _upload("Old version", "stale bytes")
        survivor_id, survivor_key = _upload("Still wanted", "keep me")

        # Real ``forget`` through the real tool: retracts the row, hides the card.
        result = await forget(_ctx("agent-a"), memory_id=forgotten_id)
        assert result["forgotten"] is True, result
        assert result["kind"] == "artifact"

        artifacts.mark_superseded(db, org_id=org_id, artifact_id=superseded_id)

        # Right now the bytes are still there — that is the bug this test pins.
        assert artifact_storage.head_size(s3, forgotten_key) is not None

        # Both dead rows are past the grace window; the survivor is untouched.
        db.execute(
            text("UPDATE artifacts SET created_at = :ts WHERE id IN (:a, :b, :c)"),
            {
                "ts": datetime.now(timezone.utc)
                - timedelta(hours=artifacts.PURGE_GRACE_HOURS + 1),
                "a": forgotten_id,
                "b": superseded_id,
                "c": survivor_id,
            },
        )
        db.commit()

        await collect_artifacts_once(lambda: SessionLocal())

        # THE assertion: the bytes are gone from real object storage.
        assert artifact_storage.head_size(s3, forgotten_key) is None, (
            "forget reported success and the sweep ran, but the object is "
            "still in MinIO — the deletion request was not honoured"
        )
        assert artifact_storage.head_size(s3, superseded_key) is None, (
            "a superseded version's bytes must be destroyed too"
        )

        # Negative control: an active artifact of the same age keeps its bytes.
        assert artifact_storage.head_size(s3, survivor_key) is not None, (
            "the purge pass destroyed an ACTIVE artifact's bytes"
        )
        assert (await get(_ctx("agent-a"), artifact_id=survivor_id, mode="url"))[
            "url"
        ].startswith("http")

        # Rows and cards for the purged artifacts are gone; the survivor's stay.
        for dead_id in (forgotten_id, superseded_id):
            assert (
                db.execute(
                    text("SELECT 1 FROM artifacts WHERE id = :id"), {"id": dead_id}
                ).fetchone()
                is None
            ), f"{dead_id} row must be removed by the purge"
            assert (
                db.execute(
                    text(
                        "SELECT 1 FROM session_memories WHERE org_id = :org_id "
                        "AND memory_type = 'artifact' "
                        "AND metadata->>'artifact_id' = :id"
                    ),
                    {"org_id": org_id, "id": dead_id},
                ).fetchone()
                is None
            ), f"{dead_id} card must not survive its artifact"

        assert (
            db.execute(
                text("SELECT status FROM artifacts WHERE id = :id"),
                {"id": survivor_id},
            ).fetchone()[0]
            == "active"
        )
    finally:
        for key in keys:
            try:
                artifact_storage.delete(s3, key)
            except Exception:
                pass
        db.close()


@pytest.mark.asyncio
async def test_purge_collects_an_inline_artifacts_materialised_object(_e2e_org):
    """M1, live: ``mode='url'`` on an INLINE artifact writes a real object that
    the row never references (``storage_uri`` stays NULL). Because the purge
    pass derives the key instead of reading that column, the object is still
    collected when the artifact is forgotten.
    """
    from app.db import SessionLocal
    from app.mcp_app import build_mcp_server
    from app.scheduler import collect_artifacts_once
    from shared import artifact_storage, artifacts

    org_id = _e2e_org
    db = SessionLocal()
    s3 = artifact_storage.get_artifact_s3_client()

    try:
        server = build_mcp_server()
        put = server._tool_manager.get_tool("artifact_put").fn
        get = server._tool_manager.get_tool("artifact_get").fn
        forget = server._tool_manager.get_tool("forget").fn

        marker = uuid.uuid4().hex[:8]
        stored = await put(
            _ctx("agent-a"),
            title=f"Inline doc {marker}",
            summary="Materialises an object when downloaded",
            kind="document",
            content=f"inline body {marker}",
        )
        assert stored["stored"] is True, stored
        artifact_id = stored["artifact_id"]

        link = await get(_ctx("agent-a"), artifact_id=artifact_id, mode="url")
        assert link["url"].startswith("http"), link

        key = artifact_storage.object_key(org_id, artifact_id, 1)
        assert artifact_storage.head_size(s3, key) is not None, (
            "mode='url' should have materialised an object"
        )
        # The row really is still inline — this is what made the object
        # uncollectable before the purge derived its key.
        row = artifacts.get_artifact(db, org_id=org_id, artifact_id=artifact_id)
        assert row["inline_text"] is not None
        assert row["storage_uri"] is None

        assert (await forget(_ctx("agent-a"), memory_id=artifact_id))["forgotten"]
        db.execute(
            text("UPDATE artifacts SET created_at = :ts WHERE id = :id"),
            {
                "ts": datetime.now(timezone.utc)
                - timedelta(hours=artifacts.PURGE_GRACE_HOURS + 1),
                "id": artifact_id,
            },
        )
        db.commit()

        await collect_artifacts_once(lambda: SessionLocal())

        assert artifact_storage.head_size(s3, key) is None, (
            "an inline artifact's materialised object outlived the artifact"
        )
    finally:
        try:
            artifact_storage.delete(
                s3, artifact_storage.object_key(org_id, artifact_id, 1)
            )
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
    from shared.artifacts import card_content

    org_id = _e2e_org
    db = SessionLocal()
    old_ts = datetime.now(timezone.utc) - timedelta(days=2)

    try:
        server = build_mcp_server()
        put = server._tool_manager.get_tool("artifact_put").fn
        remember = server._tool_manager.get_tool("remember").fn

        marker = uuid.uuid4().hex[:8]
        title = f"Retention-survivor {marker}"
        summary = "Must outlive a 1-day retention window while active"
        kind = "document"
        content = f"content {marker}"

        stored = await put(
            _ctx("agent-a"), title=title, summary=summary, kind=kind, content=content
        )
        assert stored["stored"] is True, stored
        artifact_id = stored["artifact_id"]

        # Now that embed_text is input-dependent (not a single fixed vector —
        # see the module docstring), this plain memory's distinct text no
        # longer collides with the artifact card's under dedup
        # (``find_similar_memory``, session+org scoped, threshold 0.95): two
        # different texts hash to uncorrelated vectors, so cosine similarity
        # is not above threshold. Using the SAME agent as the artifact_put
        # call above (previously required a different agent/session to dodge
        # a fixed-embedding dedup collision) is therefore safe again.
        remembered = await remember(_ctx("agent-a"), content=f"plain fact {marker}")
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

        # And the card must still be genuinely reachable through recall, not
        # merely present as a row — exact-content-match query, same technique
        # as the headline discovery test above.
        recall = server._tool_manager.get_tool("recall").fn
        query = card_content(title, summary, kind, len(content.encode("utf-8")))
        found = await recall(_ctx("agent-b"), query=query, limit=5)
        cards = [
            m
            for m in found.get("memories", [])
            if m.get("memory_type") == "artifact" and marker in m.get("content", "")
        ]
        assert cards, f"card survived the DB sweep but is not recall-reachable: {found!r}"

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
