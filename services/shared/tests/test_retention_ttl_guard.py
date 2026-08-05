"""``RAW_OBSERVATION_TTL_DAYS`` must never be able to wipe the fleet.

The hidden-raw sweep in ``run_retention`` is the only DELETE in this repo with
**no ``org_id`` predicate** — it deliberately sits outside the per-org loop and
is bounded by age alone. That makes the parsed TTL the single value standing
between a cron job and every tenant's ``recall_hidden`` rows.

Executed against Postgres 16, the cutoff ``NOW() - (:days || ' days')::interval``
behaves like this::

     7  -> cutoff 2026-07-29  -> one-second-old row deleted? f
     1  -> cutoff 2026-08-04  -> f
     0  -> cutoff NOW()       -> t
    -1  -> cutoff TOMORROW    -> t

and the DELETE with ``days = 0`` removed 3 of 3 seeded hidden rows spanning two
orgs, including rows one second old.

The dangerous input is not adversarial, it is *plausible*: an operator wanting
to turn the sweep off would reach for ``0`` or ``-1``. So the module both
rejects those with a fatal error before a single DELETE is issued, and offers a
real way to say "off".

The suite that used to cover this asserted the string ``"RAW_OBSERVATION_TTL_DAYS"``
appeared in the source file. That is a grep. These are the tests.
"""

import importlib
import os
import sys
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


# ---------------------------------------------------------------------------
# The parser, directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["0", "-1", "-7", "  0  "])
def test_non_positive_ttl_is_rejected(raw):
    """``0`` and negatives put the cutoff at or after NOW(). Fatal, not clamped.

    Clamping would be worse than rejecting: the operator who typed ``0`` meant
    "stop deleting", and silently substituting 7 would delete a week of curator
    fuel while looking like it obeyed.
    """
    mod = _load_run_retention()
    with pytest.raises(mod.RetentionConfigError) as excinfo:
        mod.parse_raw_observation_ttl_days(raw)
    assert "RAW_OBSERVATION_TTL_DAYS" in str(excinfo.value)


@pytest.mark.parametrize("raw", ["3651", "99999999"])
def test_absurdly_large_ttl_is_rejected(raw):
    mod = _load_run_retention()
    with pytest.raises(mod.RetentionConfigError):
        mod.parse_raw_observation_ttl_days(raw)


@pytest.mark.parametrize("raw", ["", "   ", "7 days", "seven", "7.5", None])
def test_unparseable_ttl_is_rejected_not_silently_defaulted(raw):
    """A malformed value must stop the run, not fall back to 7.

    ``None`` (unset) is the one input that legitimately means "use the
    default"; the empty string is a Railway/compose artefact and is a typo, not
    an intent.
    """
    mod = _load_run_retention()
    if raw is None:
        assert mod.parse_raw_observation_ttl_days(None) == (
            mod.DEFAULT_RAW_OBSERVATION_TTL_DAYS
        )
        return
    with pytest.raises(mod.RetentionConfigError):
        mod.parse_raw_observation_ttl_days(raw)


@pytest.mark.parametrize("raw", ["1", "7", "30", " 30 ", "3650"])
def test_valid_ttl_parses(raw):
    """Positive control: the guard is not simply rejecting everything."""
    mod = _load_run_retention()
    assert mod.parse_raw_observation_ttl_days(raw) == int(raw.strip())


@pytest.mark.parametrize("raw", ["off", "OFF", "never", "disabled", " off "])
def test_the_sweep_can_be_turned_off_explicitly(raw):
    """The operator who reaches for ``0`` needs somewhere safe to land."""
    mod = _load_run_retention()
    assert mod.parse_raw_observation_ttl_days(raw) is None


def test_the_rejection_message_names_the_safe_way_to_disable():
    """An error that only says "no" sends the operator back to ``-1``."""
    mod = _load_run_retention()
    with pytest.raises(mod.RetentionConfigError) as excinfo:
        mod.parse_raw_observation_ttl_days("0")
    assert "off" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ...and through ``main()``, where it has to fire before any DELETE
# ---------------------------------------------------------------------------


class _RecordingConn:
    """Records every statement so a test can assert nothing was deleted."""

    def __init__(self):
        self.calls = []
        self._orgs = 0

    def execute(self, stmt, params=None):
        sql = getattr(stmt, "text", str(stmt))
        self.calls.append((sql, params))
        result = MagicMock()
        result.fetchall.return_value = [("acct-a", "pro", None)]
        if "FROM organizations" in sql:
            result.fetchone.return_value = (f"org-{self._orgs}",)
            self._orgs += 1
        else:
            result.fetchone.return_value = None
        result.rowcount = 0
        return result

    def commit(self):
        self.calls.append(("COMMIT", None))

    def rollback(self):
        self.calls.append(("ROLLBACK", None))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def deletes(self):
        return [sql for sql, _ in self.calls if "DELETE" in sql.upper()]


def _run_main_with(mod, conn, ttl_value):
    env = {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}
    if ttl_value is not None:
        env["RAW_OBSERVATION_TTL_DAYS"] = ttl_value
    engine = MagicMock()
    engine.connect.return_value = conn
    with (
        patch.dict("os.environ", env, clear=False),
        patch.object(mod, "create_engine", return_value=engine),
    ):
        if ttl_value is None:
            os.environ.pop("RAW_OBSERVATION_TTL_DAYS", None)
        mod.main()


@pytest.mark.parametrize("bad", ["0", "-1", "seven", ""])
def test_main_exits_before_issuing_any_statement_on_a_bad_ttl(bad):
    """Fatal *before* any DELETE — including the per-org sweeps, which commit.

    Validating late would still leave the orphan-edge, orphan-entity and
    compression-original sweeps already committed for every org before the
    process noticed the TTL was unusable.
    """
    mod = _load_run_retention()
    conn = _RecordingConn()
    with pytest.raises(SystemExit) as excinfo:
        _run_main_with(mod, conn, bad)
    assert excinfo.value.code == 1
    assert conn.deletes == [], (
        f"a run with RAW_OBSERVATION_TTL_DAYS={bad!r} issued DELETEs: {conn.deletes}"
    )
    assert conn.calls == [], "no statement at all should have been issued"


def test_main_still_sweeps_on_a_valid_ttl():
    """Positive control: the guard did not disable the sweep it protects."""
    mod = _load_run_retention()
    conn = _RecordingConn()
    _run_main_with(mod, conn, "7")
    hidden = [
        (sql, params)
        for sql, params in conn.calls
        if "recall_hidden = TRUE" in sql and "DELETE" in sql.upper()
    ]
    assert hidden, "the hidden-raw sweep did not run"
    assert hidden[0][1] == {"days": 7}


def test_main_skips_the_sweep_when_the_ttl_is_off():
    mod = _load_run_retention()
    conn = _RecordingConn()
    _run_main_with(mod, conn, "off")
    assert not [sql for sql, _ in conn.calls if "recall_hidden = TRUE" in sql], (
        "RAW_OBSERVATION_TTL_DAYS=off must not delete hidden rows"
    )


# ---------------------------------------------------------------------------
# Against the real database, because the cutoff arithmetic is the defect
# ---------------------------------------------------------------------------

needs_db = requires_db("the RAW_OBSERVATION_TTL_DAYS destruction guard")


@pytest.fixture
def seeded_hidden_rows():
    """Two orgs' worth of hidden rows: fresh in both, plus one genuinely old."""
    import uuid

    engine = create_engine(database_url())
    orgs = (f"ttlguard-a-{uuid.uuid4().hex[:8]}", f"ttlguard-b-{uuid.uuid4().hex[:8]}")
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO session_memories
                        (session_id, org_id, agent_id, memory_type, content,
                         recall_hidden, created_at)
                    VALUES
                        ('s', :a, 'agent', 'observation', 'a fresh', TRUE,
                         NOW() - INTERVAL '1 second'),
                        ('s', :b, 'agent', 'observation', 'b fresh', TRUE,
                         NOW() - INTERVAL '1 second'),
                        ('s', :b, 'agent', 'observation', 'b ancient', TRUE,
                         NOW() - INTERVAL '30 days')
                    """
                ),
                {"a": orgs[0], "b": orgs[1]},
            )
            conn.commit()
        yield engine, orgs
    finally:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM session_memories WHERE org_id IN (:a, :b)"),
                {"a": orgs[0], "b": orgs[1]},
            )
            conn.commit()
        engine.dispose()


def _hidden_count(engine, orgs):
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT count(*) FROM session_memories "
                "WHERE recall_hidden AND org_id IN (:a, :b)"
            ),
            {"a": orgs[0], "b": orgs[1]},
        ).scalar()


def _real_main(mod, ttl_value, real_url):
    """Run ``main()`` against real Postgres with a faked Supabase side."""
    supa = MagicMock()
    supa_conn = MagicMock()
    supa_conn.__enter__.return_value = supa_conn
    supa_conn.__exit__.return_value = False
    supa_conn.execute.return_value.fetchall.return_value = []
    supa.connect.return_value = supa_conn

    real = create_engine(real_url)

    def _engine_for(url, **kwargs):
        return real if url == real_url else supa

    try:
        with (
            patch.dict(
                "os.environ",
                {
                    "DATABASE_URL": real_url,
                    "SUPABASE_DATABASE_URL": "supabase://fake",
                    "RAW_OBSERVATION_TTL_DAYS": ttl_value,
                },
            ),
            patch.object(mod, "create_engine", side_effect=_engine_for),
        ):
            mod.main()
    finally:
        real.dispose()


@needs_db
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_zero_or_negative_ttl_no_longer_wipes_every_tenant(seeded_hidden_rows, bad):
    """The finding, end to end: pre-fix this DELETE removed 3 of 3 rows.

    Both seeded orgs are untouched — including the 30-day-old row that a
    *correct* 7-day sweep would legitimately remove — because the run must not
    start at all.
    """
    mod = _load_run_retention()
    engine, orgs = seeded_hidden_rows
    assert _hidden_count(engine, orgs) == 3

    with pytest.raises(SystemExit) as excinfo:
        _real_main(mod, bad, database_url())
    assert excinfo.value.code == 1

    assert _hidden_count(engine, orgs) == 3, (
        f"RAW_OBSERVATION_TTL_DAYS={bad} deleted hidden rows across tenants"
    )


@needs_db
def test_a_valid_ttl_still_sweeps_only_the_expired_rows(seeded_hidden_rows):
    """Positive control against the same fixture, proving it is not vacuous.

    If this passed while the test above also passed on a guard that simply
    refused everything, the guard would be a functional outage rather than a
    fix.
    """
    mod = _load_run_retention()
    engine, orgs = seeded_hidden_rows
    assert _hidden_count(engine, orgs) == 3

    _real_main(mod, "7", database_url())

    assert _hidden_count(engine, orgs) == 2, (
        "the 30-day-old hidden row should have been swept, the fresh ones kept"
    )
