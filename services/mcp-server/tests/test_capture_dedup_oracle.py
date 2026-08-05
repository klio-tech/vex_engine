"""Can a passive capture be used to probe someone else's private rows? Real Postgres.

``write_scoped_memory`` returns the MATCHING ROW'S ID and memory_type on a
dedup hit. For ``scope='private'`` the dedup window is deliberately narrowed to
``user_id = :user_id`` precisely because of that — ``find_similar_memory``'s
docstring calls the predicate a privacy control, not a correctness one.

Every OTHER scope falls back to the historical session-scoped window:
``session_id = :session_id AND org_id = :org_id``, with no ``scope`` and no
``user_id`` predicate. And the capture path builds that session id from the
CLIENT-SUPPLIED ``session_id`` field. So an unattributable caller — one whose
key resolves to no person, whose writes therefore land at ``org`` scope — could
name the bucket an attributed user's captures land in, submit a guess, and
learn from the response whether that user had already stored near-identical
text, plus the id of their private row.

The fix namespaces the bucket by the resolved principal, so an unattributed
caller and an attributed one can never share one. The client's session id still
groups a single Claude Code session's writes, which is all it was ever for.

Real Postgres because the dedup window is a WHERE clause. ``embed_text`` is
patched to a content-derived vector so identical text is an exact match and
different text is not — the dedup threshold must genuinely decide the outcome.
"""

import hashlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.capture import _capture_session_id
from shared import brain
from shared.testing import database_url, requires_db

pytestmark = requires_db("the capture dedup-window tests")

DB_URL = database_url()
EMBED_DIM = 1024

PRIYA = "11111111-2222-3333-4444-555555555555"
AGENT = "agent-1"
SECRET = "priya is interviewing at a competitor on thursday"


def _content_embedding(content: str) -> list[float]:
    """Same text -> same vector; different text -> a different one."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(EMBED_DIM)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


def _mock_embed():
    async def _embed(content):
        return _content_embedding(content)

    return patch("shared.memory.embed_text", new=AsyncMock(side_effect=_embed))


@pytest.fixture()
def db():
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
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


async def _capture(db, *, org_id, content, user_id, client_session_id, scope):
    """One passive capture, wired exactly as ``capture_event`` wires it."""
    with _mock_embed():
        return await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=AGENT,
            content=content,
            memory_type="note",
            session_id=_capture_session_id(
                org_id, AGENT, client_session_id, user_id=user_id
            ),
            scope=scope,
            user_id=user_id,
        )


@pytest.mark.asyncio
async def test_an_unattributed_capture_cannot_dedup_against_a_private_row(db):
    """The oracle: same bucket name, and the answer names the victim's row."""
    org = _org(db)
    shared_session = "claude-code-session-7"

    stored = await _capture(
        db,
        org_id=org,
        content=SECRET,
        user_id=PRIYA,
        client_session_id=shared_session,
        scope="private",
    )
    assert stored["stored"] is True, stored

    # The attacker names the same client session id and guesses the content.
    probe = await _capture(
        db,
        org_id=org,
        content=SECRET,
        user_id=None,
        client_session_id=shared_session,
        scope="org",
    )

    assert probe.get("deduplicated") is not True, (
        "an unattributable caller learned that a near-identical memory already "
        f"exists, and was handed its id: {probe}"
    )
    assert probe.get("id") != stored["id"]
    assert probe["stored"] is True

    # Positive control: dedup still works WITHIN one principal's own bucket,
    # so the assertion above is about the boundary and not about a dedup that
    # stopped functioning.
    again = await _capture(
        db,
        org_id=org,
        content=SECRET,
        user_id=PRIYA,
        client_session_id=shared_session,
        scope="private",
    )
    assert again.get("deduplicated") is True, (
        f"positive control failed: a repeat capture must still dedup; got {again}"
    )
    assert again["id"] == stored["id"]


@pytest.mark.asyncio
async def test_two_attributed_users_do_not_share_a_bucket(db):
    """The same probe between two real people, neither of them anonymous."""
    org = _org(db)
    dev = "99999999-8888-7777-6666-555555555555"
    shared_session = "claude-code-session-9"

    mine = await _capture(
        db,
        org_id=org,
        content=SECRET,
        user_id=PRIYA,
        client_session_id=shared_session,
        scope="org",
    )
    assert mine["stored"] is True

    theirs = await _capture(
        db,
        org_id=org,
        content=SECRET,
        user_id=dev,
        client_session_id=shared_session,
        scope="org",
    )

    assert theirs.get("deduplicated") is not True, (
        f"one user's capture deduped against another's row: {theirs}"
    )
    assert theirs["stored"] is True


@pytest.mark.asyncio
async def test_the_same_principal_and_session_still_dedups(db):
    """Positive control for the whole feature: dedup is not switched off."""
    org = _org(db)
    first = await _capture(
        db,
        org_id=org,
        content="the staging deploy runs at 4pm",
        user_id=PRIYA,
        client_session_id="s1",
        scope="org",
    )
    second = await _capture(
        db,
        org_id=org,
        content="the staging deploy runs at 4pm",
        user_id=PRIYA,
        client_session_id="s1",
        scope="org",
    )
    assert first["stored"] is True
    assert second.get("deduplicated") is True, second
    assert second["id"] == first["id"]


def test_the_bucket_is_namespaced_by_principal_and_bounded():
    """Distinct principals get distinct buckets, and none overflows the column."""
    anon = _capture_session_id("acme", AGENT, "s1", user_id=None)
    priya = _capture_session_id("acme", AGENT, "s1", user_id=PRIYA)
    other = _capture_session_id(
        "acme", AGENT, "s1", user_id="99999999-8888-7777-6666-555555555555"
    )

    assert len({anon, priya, other}) == 3, (anon, priya, other)
    # Same principal, same client session -> same bucket, or dedup never fires.
    assert priya == _capture_session_id("acme", AGENT, "s1", user_id=PRIYA)
    # Different client sessions still separate, as before.
    assert priya != _capture_session_id("acme", AGENT, "s2", user_id=PRIYA)

    # session_memories.session_id is VARCHAR(256); a client controls one of
    # these segments, so the result has to be bounded regardless of input.
    huge = _capture_session_id("o" * 64, "a" * 128, "s" * 4096, user_id=PRIYA)
    assert len(huge) <= 256, len(huge)
    assert huge != _capture_session_id("o" * 64, "a" * 128, "s" * 4096, user_id=None)
