"""Migration 034: the originals cache.

Mirrors test_migration_031.py's conventions — real Postgres, skips without it.
"""

import os

import pytest
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


@pytest.fixture(scope="module")
def conn():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not reachable; migration 034 tests require the live DB")
    with create_engine(DB_URL).connect() as c:
        yield c


def _columns(conn, table):
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def test_compression_cache_exists_with_the_siloing_columns(conn):
    cols = _columns(conn, "compression_cache")
    assert {"id", "org_id", "user_id", "content_hash", "expires_at"} <= set(cols)
    assert cols["org_id"] == "NO", "org_id is the tenancy boundary and cannot be null"


def test_user_id_is_nullable(conn):
    """An unattributed caller still gets a cache entry; it is simply
    unreachable by any user-scoped read, which is the correct outcome."""
    assert _columns(conn, "compression_cache")["user_id"] == "YES"


def test_dedup_index_on_org_and_hash(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'compression_cache'")
    ).fetchall()
    defs = " ".join(r[0] for r in idx)
    assert "content_hash" in defs, "dedup requires an index on the hash"
    assert "org_id" in defs


def test_expiry_index_exists(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'compression_cache'")
    ).fetchall()
    assert any("expires_at" in r[0] for r in idx), "the sweep needs an expiry index"


def test_inline_or_uri_but_the_table_permits_both_null(conn):
    """A pending row exists briefly before its content lands."""
    cols = _columns(conn, "compression_cache")
    assert cols["inline_text"] == "YES"
    assert cols["storage_uri"] == "YES"
