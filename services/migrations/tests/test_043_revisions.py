import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the memory-revisions migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_session_memories_revision_column():
    cols = {c["name"]: c for c in _insp().get_columns("session_memories")}
    assert "revision" in cols
    assert cols["revision"]["nullable"] is False


def test_session_memories_revision_has_a_value_on_every_row():
    # server_default backfills existing rows with no migration-time UPDATE;
    # a NULL here would mean the default did not apply.
    engine = sa.create_engine(database_url())
    with engine.connect() as conn:
        (null_count,) = conn.execute(
            sa.text("SELECT COUNT(*) FROM session_memories WHERE revision IS NULL")
        ).fetchone()
    assert null_count == 0
