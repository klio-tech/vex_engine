"""Migration 037: the GIN full-text index behind the lexical recall arm.

Three things have to be true, and only one of them is "the index exists".

1. The indexed expression is **byte-identical** to the one ``shared.lexical``
   hands the query. Postgres matches an expression index by expression
   equality, so a drift of one character retires the index silently and the
   only symptom is a sequential scan.
2. A write **larger than the bound still succeeds**. Without ``left(content,
   N)`` the index would reject any row whose tsvector exceeds 1 MB, turning
   the addition of search into a write outage.
3. The **downgrade removes it** and the re-upgrade puts it back, so a rollback
   is total.

Like the other migration tests this builds its own scratch database, because
it moves the schema up and down.
"""

import os
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

import pytest
from sqlalchemy import create_engine, text

from shared.lexical import FTS_CONTENT_LIMIT, FTS_INDEX_NAME, fts_document_expression

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

SCRATCH_DB = "klio_mig_037"

MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


def _alembic(url: str, *args: str):
    env = {**os.environ, "DATABASE_URL": url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=MIGRATIONS_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def _require(proc, what: str) -> None:
    assert proc.returncode == 0, f"{what} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def scratch_url():
    admin_url = _with_database(DB_URL, "postgres")
    try:
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:  # pragma: no cover - environment guard
        pytest.skip("Postgres not reachable; migration 037 tests require the test DB")

    url = _with_database(DB_URL, SCRATCH_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))
    try:
        yield url
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))


@pytest.fixture
def db(scratch_url):
    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to head")
    engine = create_engine(scratch_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as c:
        yield c
    engine.dispose()


def _index_definition(conn) -> str | None:
    return conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
        {"n": FTS_INDEX_NAME},
    ).scalar()


def _insert(conn, content: str) -> None:
    conn.execute(
        text(
            "INSERT INTO session_memories "
            "(org_id, agent_id, session_id, scope, status, content, memory_type) "
            "VALUES ('org_037', 'a', 's', 'org', 'active', :c, 'fact')"
        ),
        {"c": content},
    )


def test_the_index_exists_and_is_a_gin_index(db):
    definition = _index_definition(db)

    assert definition is not None, f"{FTS_INDEX_NAME} was not created"
    assert "USING gin" in definition


def test_the_indexed_expression_is_the_one_the_query_uses(db):
    """The drift check, on the stored definition.

    Postgres rewrites what it stores — it quotes ``left`` (a reserved word)
    and pins the configuration as ``'english'::regconfig`` — so the comparison
    undoes exactly those two cosmetic rewrites and nothing else. The bound and
    the column are compared as written.
    """
    definition = _index_definition(db)

    normalised = definition.replace('"', "").replace("::regconfig", "")
    assert fts_document_expression() in normalised, (
        f"indexed expression drifted from shared.lexical:\n{definition}"
    )


def test_the_planner_actually_uses_the_index_for_that_expression(db):
    """The drift check that cannot be fooled by string massaging.

    An expression index Postgres declines to match is indistinguishable from
    no index at all, and the symptom — a sequential scan — is invisible until
    the table is large. ``enable_seqscan = off`` makes the planner state its
    preference rather than picking a scan because the table is tiny.
    """
    _insert(db, "the metrics endpoint is gated by KLIO_METRICS_TOKEN")
    db.execute(text("SET enable_seqscan = off"))

    plan = "\n".join(
        row[0]
        for row in db.execute(
            text(
                "EXPLAIN SELECT id FROM session_memories WHERE "
                + fts_document_expression()
                + " @@ websearch_to_tsquery('english', :q)"
            ),
            {"q": "KLIO_METRICS_TOKEN"},
        )
    )

    assert FTS_INDEX_NAME in plan, f"the planner ignored the index:\n{plan}"


def test_a_write_larger_than_the_bound_still_succeeds(db):
    """The regression that the bound exists to prevent.

    ``repeat()`` would dedupe — a tsvector stores distinct lexemes — so the
    content is built from DISTINCT tokens, which is what actually overflows.
    """
    tokens = " ".join(f"tok{n:07d}" for n in range(140_000))
    assert len(tokens) > FTS_CONTENT_LIMIT

    _insert(db, tokens)

    stored = db.execute(
        text("SELECT max(length(content)) FROM session_memories WHERE org_id='org_037'")
    ).scalar()
    assert stored == len(tokens), "content must be stored whole, not truncated"


def test_the_bound_only_limits_the_index_not_the_stored_row(db):
    """Positive control for the test above: the row is genuinely oversized and
    the index genuinely covers its head, so the write is not passing merely
    because nothing was indexed."""
    head = "unmistakeable_sentinel_lexeme"
    tokens = head + " " + " ".join(f"tok{n:07d}" for n in range(140_000))

    _insert(db, tokens)

    found = db.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE org_id = 'org_037' AND "
            + fts_document_expression()
            + " @@ websearch_to_tsquery('english', :q)"
        ),
        {"q": head},
    ).scalar()
    assert found == 1


def test_downgrade_drops_the_index_and_reupgrade_rebuilds_it(db, scratch_url):
    assert _index_definition(db) is not None

    _require(_alembic(scratch_url, "downgrade", "036"), "downgrade to 036")
    assert _index_definition(db) is None

    _require(_alembic(scratch_url, "upgrade", "037"), "re-upgrade to 037")
    assert _index_definition(db) is not None
