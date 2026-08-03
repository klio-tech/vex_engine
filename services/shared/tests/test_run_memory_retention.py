"""run_retention also enforces memory retention per org."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch


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


def test_hidden_raw_ttl_sql_present():
    """run_retention must sweep recall_hidden rows on a fixed short TTL."""
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "run_retention.py"
    )
    source = open(path).read()
    assert "recall_hidden = TRUE" in source
    assert "RAW_OBSERVATION_TTL_DAYS" in source


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
