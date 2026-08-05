"""run_retention also enforces memory retention per org."""

import importlib
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db


def _load_run_retention():
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts"),
    )
    return importlib.import_module("run_retention")


def test_calls_enforce_memory_retention_with_memory_days():
    mod = _load_run_retention()
    calls = []

    class FakeConn:
        def execute(self, stmt, params=None):
            text_sql = getattr(stmt, "text", str(stmt))
            if params:
                calls.append((text_sql, params))
            result = MagicMock()
            result.fetchall.return_value = [("acct", "pro", None)]
            result.fetchone.return_value = ("org-1",)
            return result

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_engine = MagicMock()
    fake_engine.connect.return_value = FakeConn()

    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=fake_engine),
    ):
        mod.main()

    assert any(
        "enforce_memory_retention" in sql and params.get("days") == 30
        for sql, params in calls
    )


def test_hidden_raw_ttl_sweep_runs_with_the_configured_ttl():
    """run_retention must sweep recall_hidden rows on a fixed short TTL.

    This used to assert that the strings ``"recall_hidden = TRUE"`` and
    ``"RAW_OBSERVATION_TTL_DAYS"`` appeared somewhere in the source file — a
    grep that would have passed just as happily against a DELETE with the
    predicate commented out. It now executes the sweep and inspects the
    statement and its bound parameter.

    The destructive half of this knob (``0``/``-1``/garbage) is covered in
    ``test_retention_ttl_guard.py``.
    """
    mod = _load_run_retention()
    statements = []

    class FakeConn:
        def execute(self, stmt, params=None):
            statements.append((getattr(stmt, "text", str(stmt)), params))
            result = MagicMock()
            result.fetchall.return_value = [("acct", "pro", None)]
            result.fetchone.return_value = ("org-1",)
            result.rowcount = 0
            return result

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_engine = MagicMock()
    fake_engine.connect.return_value = FakeConn()

    with (
        patch.dict(
            "os.environ",
            {
                "DATABASE_URL": "x",
                "SUPABASE_DATABASE_URL": "y",
                "RAW_OBSERVATION_TTL_DAYS": "9",
            },
        ),
        patch.object(mod, "create_engine", return_value=fake_engine),
    ):
        mod.main()

    sweeps = [
        (sql, params)
        for sql, params in statements
        if "DELETE" in sql.upper() and "recall_hidden = TRUE" in sql
    ]
    assert len(sweeps) == 1, "exactly one hidden-raw sweep should have run"
    assert sweeps[0][1] == {"days": 9}, (
        "the sweep must be bound to the configured TTL, not a hardcoded one"
    )


def test_sweeps_orphan_graph_edges_for_every_org():
    """Retention deletes from ``session_memories`` only. ``memory_edges`` has
    no foreign keys by design (a forget must never fail because of an edge),
    so nothing collects the edges whose endpoints retention just removed —
    they accumulate forever, and an edge into a deleted or retracted memory is
    exactly what ``sweep_orphan_edges`` calls "forgotten means forgotten, the
    graph included".

    The sweep had ZERO production callers. This pins the wiring, and pins it
    to the GraphStore seam rather than to inline SQL (``memory_edges`` may not
    be named outside ``graph_store.py``).
    """
    mod = _load_run_retention()
    seen_orgs = []

    class FakeConn:
        def execute(self, stmt, params=None):
            result = MagicMock()
            result.fetchall.return_value = [
                ("acct-a", "pro", None),
                ("acct-b", "free", None),
            ]
            result.fetchone.return_value = (f"org-{len(seen_orgs)}",)
            return result

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    conn = FakeConn()
    fake_engine = MagicMock()
    fake_engine.connect.return_value = conn

    def _fake_sweep(db, *, org_id):
        assert db is conn, "the sweep must run on the TimescaleDB connection"
        seen_orgs.append(org_id)
        return 1

    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=fake_engine),
        patch.object(mod.graph_store, "sweep_orphan_edges", _fake_sweep),
    ):
        mod.main()

    assert seen_orgs == ["org-0", "org-1"], (
        "every org whose memories were pruned must have its graph swept"
    )


def test_a_failed_sweep_never_aborts_the_retention_run():
    """The sweep is hygiene, not the job: one org's graph blowing up must not
    leave every later org's retention unenforced."""
    mod = _load_run_retention()
    retention_calls = []

    class FakeConn:
        def execute(self, stmt, params=None):
            text_sql = getattr(stmt, "text", str(stmt))
            if params and "enforce_memory_retention" in text_sql:
                retention_calls.append(params["org_id"])
            result = MagicMock()
            result.fetchall.return_value = [
                ("acct-a", "pro", None),
                ("acct-b", "free", None),
            ]
            result.fetchone.return_value = (f"org-{len(retention_calls)}",)
            return result

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_engine = MagicMock()
    fake_engine.connect.return_value = FakeConn()

    def _boom(db, *, org_id):
        raise RuntimeError("graph engine down")

    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=fake_engine),
        patch.object(mod.graph_store, "sweep_orphan_edges", _boom),
    ):
        mod.main()

    assert retention_calls == ["org-0", "org-1"]


# ---------------------------------------------------------------------------
# compression_cache — the sweep the table was built expecting
#
# Migration 034 gave the table an ``expires_at`` column, an index named for
# the sweep ("the expiry sweep scans by time and must not seq-scan the
# table") and a docstring promising 24-hour retention. Nothing ever deleted a
# row: ``expires_at`` was only ever read as a predicate, so an expired
# original stayed on disk permanently while being unreachable through
# ``load_original``. Content the user cannot get back and cannot get rid of
# is a retention hole, not a growth problem.
# ---------------------------------------------------------------------------


def _fake_engine_over(conn):
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


class _MultiOrgConn:
    """Two accounts, distinct org ids, recording every statement."""

    def __init__(self):
        self.calls = []
        self._orgs = 0

    def execute(self, stmt, params=None):
        sql = getattr(stmt, "text", str(stmt))
        self.calls.append((sql, params))
        result = MagicMock()
        result.fetchall.return_value = [
            ("acct-a", "pro", None),
            ("acct-b", "free", None),
        ]
        # Only the org lookup hands out the next org id; every other statement
        # (retention, sweeps) must not advance the counter or the recorded
        # per-org order becomes meaningless.
        if "FROM organizations" in sql:
            result.fetchone.return_value = (f"org-{self._orgs}",)
            self._orgs += 1
        else:
            result.fetchone.return_value = None
        result.rowcount = 0
        return result

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_sweeps_expired_compression_originals_for_every_org():
    mod = _load_run_retention()
    seen = []

    conn = _MultiOrgConn()
    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=_fake_engine_over(conn)),
        patch.object(mod.graph_store, "sweep_orphan_edges", lambda db, *, org_id: 0),
        patch.object(mod.graph_store, "sweep_orphan_entities", lambda db, *, org_id: 0),
        patch.object(
            mod,
            "sweep_expired_originals",
            lambda db, *, org_id: (seen.append(org_id), 0)[1],
        ),
    ):
        mod.main()

    assert seen == ["org-0", "org-1"], (
        "every org's expired compression originals must be swept"
    )


def test_a_failed_originals_sweep_never_aborts_the_retention_run():
    mod = _load_run_retention()
    later = []

    conn = _MultiOrgConn()

    def _boom(db, *, org_id):
        raise RuntimeError("cache table is on fire")

    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=_fake_engine_over(conn)),
        patch.object(mod.graph_store, "sweep_orphan_edges", lambda db, *, org_id: 0),
        patch.object(
            mod.graph_store,
            "sweep_orphan_entities",
            lambda db, *, org_id: (later.append(org_id), 0)[1],
        ),
        patch.object(mod, "sweep_expired_originals", _boom),
    ):
        mod.main()

    assert later == ["org-0", "org-1"], (
        "one org's cache sweep blowing up left later work undone"
    )


# ---------------------------------------------------------------------------
# entities — the GC that did not exist
# ---------------------------------------------------------------------------


def test_sweeps_orphan_entities_for_every_org():
    """Routed through the GraphStore seam: ``entities`` may not be named in
    SQL outside ``graph_store.py``, and an invariant test enforces it."""
    mod = _load_run_retention()
    seen = []

    conn = _MultiOrgConn()
    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=_fake_engine_over(conn)),
        patch.object(mod.graph_store, "sweep_orphan_edges", lambda db, *, org_id: 0),
        patch.object(mod, "sweep_expired_originals", lambda db, *, org_id: 0),
        patch.object(
            mod.graph_store,
            "sweep_orphan_entities",
            lambda db, *, org_id: (seen.append(org_id), 0)[1],
        ),
    ):
        mod.main()

    assert seen == ["org-0", "org-1"]


def test_entities_are_swept_after_the_edges_that_evidence_them():
    """Order is load-bearing.

    An entity's evidence is its edges. Edges into deleted memories still count
    until ``sweep_orphan_edges`` removes them, so sweeping entities first
    collects nothing and leaves the orphans for another day.
    """
    mod = _load_run_retention()
    order = []

    conn = _MultiOrgConn()
    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=_fake_engine_over(conn)),
        patch.object(mod, "sweep_expired_originals", lambda db, *, org_id: 0),
        patch.object(
            mod.graph_store,
            "sweep_orphan_edges",
            lambda db, *, org_id: (order.append(("edges", org_id)), 0)[1],
        ),
        patch.object(
            mod.graph_store,
            "sweep_orphan_entities",
            lambda db, *, org_id: (order.append(("entities", org_id)), 0)[1],
        ),
    ):
        mod.main()

    assert order == [
        ("edges", "org-0"),
        ("entities", "org-0"),
        ("edges", "org-1"),
        ("entities", "org-1"),
    ]


def test_a_failed_entity_sweep_never_aborts_the_retention_run():
    mod = _load_run_retention()
    retention_orgs = []

    class _Conn(_MultiOrgConn):
        def execute(self, stmt, params=None):
            sql = getattr(stmt, "text", str(stmt))
            if params and "enforce_memory_retention" in sql:
                retention_orgs.append(params["org_id"])
            return super().execute(stmt, params)

    def _boom(db, *, org_id):
        raise RuntimeError("graph engine down")

    with (
        patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}),
        patch.object(mod, "create_engine", return_value=_fake_engine_over(_Conn())),
        patch.object(mod.graph_store, "sweep_orphan_edges", lambda db, *, org_id: 0),
        patch.object(mod, "sweep_expired_originals", lambda db, *, org_id: 0),
        patch.object(mod.graph_store, "sweep_orphan_entities", _boom),
    ):
        mod.main()

    assert retention_orgs == ["org-0", "org-1"]


# ---------------------------------------------------------------------------
# What the originals sweep actually deletes. Real Postgres.
#
# This is a delete of user content, so a mocked session proves nothing: the
# boundary is the SQL. Both halves are asserted — the expired row goes, and
# NOTHING else does, in particular not a live row and not another tenant's.
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_db():
    engine = create_engine(database_url())
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
                    text("DELETE FROM compression_cache WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _cache_org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _store(db, *, org_id: str, user_id, ttl_seconds: int) -> str:
    """Insert one original. A negative ttl produces an already-expired row."""
    handle = uuid.uuid4().hex
    db.execute(
        text(
            """
            INSERT INTO compression_cache (
                id, org_id, user_id, content_hash, inline_text,
                size_bytes, expires_at
            ) VALUES (
                :id, :org, :user, :hash, :body, :size,
                NOW() + make_interval(secs => :ttl)
            )
            """
        ),
        {
            "id": handle,
            "org": org_id,
            "user": user_id,
            "hash": uuid.uuid4().hex,
            "body": "the original bytes",
            "size": 18,
            "ttl": ttl_seconds,
        },
    )
    db.commit()
    return handle


def _handles(db, *, org_id: str) -> set:
    rows = db.execute(
        text("SELECT id FROM compression_cache WHERE org_id = :o"), {"o": org_id}
    ).fetchall()
    return {r[0] for r in rows}


@requires_db("the compression-originals retention sweep")
def test_the_originals_sweep_removes_expired_rows_and_only_those(cache_db):
    mod = _load_run_retention()
    org = _cache_org(cache_db)

    expired = _store(cache_db, org_id=org, user_id="user-1", ttl_seconds=-60)
    live = _store(cache_db, org_id=org, user_id="user-1", ttl_seconds=3600)
    # An unattributed caller's row is a real row with a real TTL; it must be
    # swept on the same terms and not skipped by a NULL-blind predicate.
    expired_null_user = _store(cache_db, org_id=org, user_id=None, ttl_seconds=-60)
    live_null_user = _store(cache_db, org_id=org, user_id=None, ttl_seconds=3600)

    assert mod.sweep_expired_originals(cache_db, org_id=org) == 2
    assert _handles(cache_db, org_id=org) == {live, live_null_user}
    assert expired not in _handles(cache_db, org_id=org)
    assert expired_null_user not in _handles(cache_db, org_id=org)


@requires_db("the compression-originals retention sweep")
def test_the_originals_sweep_never_reaches_into_another_org(cache_db):
    mod = _load_run_retention()
    org_a = _cache_org(cache_db)
    org_b = _cache_org(cache_db)

    a_expired = _store(cache_db, org_id=org_a, user_id="u", ttl_seconds=-60)
    b_expired = _store(cache_db, org_id=org_b, user_id="u", ttl_seconds=-60)

    assert mod.sweep_expired_originals(cache_db, org_id=org_a) == 1
    assert a_expired not in _handles(cache_db, org_id=org_a)
    assert b_expired in _handles(cache_db, org_id=org_b), (
        "one org's retention run deleted another org's originals"
    )


@requires_db("the compression-originals retention sweep")
def test_a_swept_original_is_gone_not_merely_unreadable(cache_db):
    """Positive control on the fixture, and the point of the whole fix.

    Before the sweep existed, an expired row was invisible through
    ``load_original`` while still sitting on disk. "Unreadable" is not
    "deleted", and only the second one is a retention policy.
    """
    mod = _load_run_retention()
    org = _cache_org(cache_db)
    expired = _store(cache_db, org_id=org, user_id="u", ttl_seconds=-60)

    # The fixture is not vacuous: the row genuinely exists before the sweep.
    assert expired in _handles(cache_db, org_id=org)

    mod.sweep_expired_originals(cache_db, org_id=org)

    still_there = cache_db.execute(
        text("SELECT inline_text FROM compression_cache WHERE id = :i"),
        {"i": expired},
    ).fetchone()
    assert still_there is None, "the content is still on disk"
