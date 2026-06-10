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
