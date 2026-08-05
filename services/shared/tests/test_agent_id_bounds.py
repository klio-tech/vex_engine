"""``agent_id`` is client input on a path that ends in ``VARCHAR(128)``.

It arrives verbatim from the ``X-Vex-Agent`` header, or — on the OAuth path —
is built from ``clientInfo.name``, which is also client-supplied. Neither was
bounded. Executed against real Postgres 16::

    X-Vex-Agent header, 500 chars           -> agent_id len 500
    OAuth slug path, clientInfo.name 500 ch -> agent_id len 515

    INSERT agent_id 515 chars -> (psycopg2.errors.StringDataRightTruncation)
                                 value too long for type character varying(128)
    INSERT agent_id 500 chars -> same

So an over-long header did not degrade, it turned every write into a 500.
"""

import pytest

from shared.auth import MAX_AGENT_ID_LEN, normalize_agent_id

DEFAULT = "mcp-default"


def test_a_short_agent_id_is_returned_unchanged():
    """Positive control, and the property that matters most.

    Real agent ids are short. If normalisation touched them the whole memory
    corpus would re-key onto new principals.
    """
    assert normalize_agent_id("claude-code", default=DEFAULT) == "claude-code"


@pytest.mark.parametrize("length", [129, 200, 515, 10_000])
def test_an_over_long_agent_id_is_bounded(length):
    out = normalize_agent_id("a" * length, default=DEFAULT)
    assert len(out) <= MAX_AGENT_ID_LEN


def test_an_agent_id_at_the_limit_is_untouched():
    """Boundary: exactly 128 fits the column and must not be rewritten."""
    exact = "a" * MAX_AGENT_ID_LEN
    assert normalize_agent_id(exact, default=DEFAULT) == exact


def test_two_different_over_long_ids_do_not_collide():
    """Plain truncation would merge distinct principals into one.

    That is worse than the error it replaces: two tenanted agents sharing an
    ``agent_id`` share an ``agent``-scoped memory bucket. The bound keeps a
    digest of the full value so distinct inputs stay distinct.
    """
    a = normalize_agent_id("x" * 200 + "-alpha", default=DEFAULT)
    b = normalize_agent_id("x" * 200 + "-beta", default=DEFAULT)
    assert a != b
    assert len(a) <= MAX_AGENT_ID_LEN and len(b) <= MAX_AGENT_ID_LEN


def test_bounding_is_deterministic():
    """Dedup and agent-scoped recall both key on this; it cannot drift."""
    long_id = "y" * 300
    assert normalize_agent_id(long_id, default=DEFAULT) == normalize_agent_id(
        long_id, default=DEFAULT
    )


@pytest.mark.parametrize("raw", [None, "", "   ", "\t"])
def test_absent_or_blank_takes_the_default(raw):
    assert normalize_agent_id(raw, default=DEFAULT) == DEFAULT


def test_surrounding_whitespace_is_stripped():
    """``X-Vex-Agent: claude-code`` and `` claude-code `` are one principal."""
    assert normalize_agent_id("  claude-code  ", default=DEFAULT) == "claude-code"


@pytest.mark.requires_db("the agent_id column bound")
def test_a_bounded_agent_id_actually_inserts():
    """The point of the bound is that the write succeeds. Prove it does.

    Without this, a bound that returned something 129 characters long would
    pass every assertion above and still 500 in production.
    """
    from sqlalchemy import create_engine, text

    from shared.testing import database_url

    engine = create_engine(database_url())
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO session_memories "
                    "(session_id, org_id, agent_id, memory_type, content) "
                    "VALUES ('s', 'agentbound-test', :a, 'observation', 'x')"
                ),
                {"a": normalize_agent_id("z" * 515, default=DEFAULT)},
            )
            conn.rollback()
    finally:
        engine.dispose()
