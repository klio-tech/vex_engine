"""The cache is a tenancy boundary, not a convenience store.

An original is whatever the user compressed — a log, a source file, an API
response. It is at least as sensitive as a memory, and these tests attack it
the same way ``services/shared/tests/test_user_silo_isolation.py`` attacks
recall.

Why a real Postgres connection rather than a mocked session: the boundary IS
the SQL. A mocked session returns whatever the test queued and would pass
against a query with no WHERE clause at all. The ``db`` fixture below mirrors
that module's — a real Connection, reachability-gated, with every row the test
created removed afterwards.

The mcp-server's other live-DB module (``test_artifact_e2e.py``) is gated
behind ``KLIO_E2E`` because it also needs MinIO. These tests need only
Postgres, and gating them the same way would let a tenancy boundary go
unverified on every ordinary test run — a skip is not a pass.
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from app.compression_cache import load_original, store_original

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _db_reachable() -> bool:
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable; the compression-cache boundary tests require it",
)

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"
CAROL = "33333333-3333-3333-3333-333333333333"


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards.

    Orgs are per-test and unique so one test's residue can never look like
    another test's leak.
    """
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
                    text("DELETE FROM compression_cache WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


@pytest.fixture()
def acme(db):
    org_id = f"acme_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


@pytest.fixture()
def other(db):
    org_id = f"other_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def test_owner_can_load_their_own(db, acme):
    """Positive control. Without this, a cache that returns nothing to
    anyone would pass every test below."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")
    assert load_original(db, org_id=acme, user_id=ALICE, handle=handle) == "secret log"


def test_another_user_gets_nothing(db, acme):
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")
    assert load_original(db, org_id=acme, user_id=BOB, handle=handle) is None


def test_a_third_user_in_the_same_org_gets_nothing(db, acme):
    """Same org, no relationship to the owner, no special role."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")
    assert load_original(db, org_id=acme, user_id=CAROL, handle=handle) is None


def test_a_user_with_their_own_entry_still_cannot_read_anothers(db, acme):
    """Holding a valid entry of your own must not widen what you can read.

    The attack this catches: a predicate that admits any row in the org once
    the caller is known to have entries there.
    """
    alice_handle = store_original(db, org_id=acme, user_id=ALICE, text="alice's log")
    bob_handle = store_original(db, org_id=acme, user_id=BOB, text="bob's log")

    assert load_original(db, org_id=acme, user_id=BOB, handle=bob_handle) == "bob's log"
    assert load_original(db, org_id=acme, user_id=BOB, handle=alice_handle) is None


def test_another_org_gets_nothing(db, acme, other):
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")
    assert load_original(db, org_id=other, user_id=ALICE, handle=handle) is None


def test_unattributed_caller_cannot_read_an_owned_entry(db, acme):
    """user_id=None must not degrade into match-anything."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")
    assert load_original(db, org_id=acme, user_id=None, handle=handle) is None


def test_expired_entry_is_not_returned(db, acme):
    """An expired handle returns None; the caller reports `expired`."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="x", ttl_seconds=-1)
    assert load_original(db, org_id=acme, user_id=ALICE, handle=handle) is None


def test_unknown_handle_returns_none(db, acme):
    assert load_original(db, org_id=acme, user_id=ALICE, handle="kc_nope") is None


def test_identical_content_dedups_within_an_org(db, acme):
    a = store_original(db, org_id=acme, user_id=ALICE, text="same bytes")
    b = store_original(db, org_id=acme, user_id=ALICE, text="same bytes")
    assert a == b


def test_identical_content_does_not_dedup_across_orgs(db, acme, other):
    a = store_original(db, org_id=acme, user_id=ALICE, text="same bytes")
    b = store_original(db, org_id=other, user_id=ALICE, text="same bytes")
    assert a != b, "one tenant's content must never satisfy another's lookup"


def test_two_users_in_one_org_can_each_store_the_same_bytes(db, acme):
    """Dedup must not lock the second user out of their own content.

    A handle salted by org alone would collide here: the first writer's row
    wins the primary key, and the second user — holding a handle the API just
    returned them — reads back nothing. See the module under test for why the
    salt is (org_id, user_id).
    """
    alice_handle = store_original(db, org_id=acme, user_id=ALICE, text="same bytes")
    bob_handle = store_original(db, org_id=acme, user_id=BOB, text="same bytes")

    assert load_original(db, org_id=acme, user_id=ALICE, handle=alice_handle) == (
        "same bytes"
    )
    assert (
        load_original(db, org_id=acme, user_id=BOB, handle=bob_handle) == "same bytes"
    )


def test_restoring_an_original_is_byte_for_byte(db, acme):
    """The whole point of the cache: what comes out is what went in."""
    original = "```py\ndef f():\n    return '/etc/hosts'\n```\nERR 1.2.3\né—🚀"
    handle = store_original(db, org_id=acme, user_id=ALICE, text=original)
    assert load_original(db, org_id=acme, user_id=ALICE, handle=handle) == original


def test_storing_again_refreshes_the_expiry(db, acme):
    """A re-compressed original must not inherit the first write's deadline."""
    handle = store_original(
        db, org_id=acme, user_id=ALICE, text="hot file", ttl_seconds=-1
    )
    assert load_original(db, org_id=acme, user_id=ALICE, handle=handle) is None

    again = store_original(db, org_id=acme, user_id=ALICE, text="hot file")
    assert again == handle
    assert load_original(db, org_id=acme, user_id=ALICE, handle=again) == "hot file"
