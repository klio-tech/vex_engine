"""``brain_session_id`` takes the client's ``mcp-session-id`` header verbatim.

The result goes into ``session_memories.session_id``, a ``VARCHAR(256)``.
Executed against real Postgres 16::

    mcp-session-id header, 400 chars -> session_id len 411
    INSERT -> (psycopg2.errors.StringDataRightTruncation)
              value too long for type character varying(256)

Note what is **not** a defect here: the reviewers also filed the
``org_id:agent_id`` fallback as an overflow. That is arithmetically false —
``len("klio-brain:") + 64 + 1 + 128 == 204``, inside 256, executed and
confirmed. Only the header path overflows, and only the header path is fixed.
"""

import pytest

from app.session import MAX_BRAIN_SESSION_ID_LEN, brain_session_id


class _Req:
    def __init__(self, sid=None):
        self.headers = {} if sid is None else {"mcp-session-id": sid}


def test_a_normal_session_header_is_used_as_is():
    """Positive control: dedup grouping must keep working unchanged."""
    assert brain_session_id(_Req("abc123"), "org", "agent") == "klio-brain:abc123"


def test_the_principal_fallback_is_unchanged():
    """Also a positive control for the claim this file declines to 'fix'."""
    assert brain_session_id(_Req(), "org", "agent") == "klio-brain:org:agent"


def test_the_worst_case_principal_fallback_already_fits():
    """204 <= 256. The filed finding is refuted, not implemented."""
    out = brain_session_id(_Req(), "o" * 64, "a" * 128)
    assert len(out) == 204
    assert len(out) <= MAX_BRAIN_SESSION_ID_LEN


@pytest.mark.parametrize("length", [246, 400, 10_000])
def test_an_over_long_session_header_is_bounded(length):
    out = brain_session_id(_Req("s" * length), "org", "agent")
    assert len(out) <= MAX_BRAIN_SESSION_ID_LEN


def test_two_different_over_long_headers_do_not_collide():
    """Truncation alone would pool two clients into one dedup bucket.

    Harmless across the tenant boundary (org_id and agent_id are filtered
    separately on every read and write) but it would silently merge two
    connections' dedup state.
    """
    a = brain_session_id(_Req("h" * 300 + "-alpha"), "org", "agent")
    b = brain_session_id(_Req("h" * 300 + "-beta"), "org", "agent")
    assert a != b


def test_bounding_is_deterministic():
    """The id has to be stable across calls or dedup stops grouping."""
    req = _Req("q" * 400)
    assert brain_session_id(req, "org", "agent") == brain_session_id(
        req, "org", "agent"
    )


def test_an_empty_session_header_still_falls_back_to_the_principal():
    assert brain_session_id(_Req(""), "org", "agent") == "klio-brain:org:agent"


@pytest.mark.requires_db("the brain session_id column bound")
def test_a_bounded_session_id_actually_inserts():
    """The bound exists so the write succeeds; prove the write succeeds."""
    from sqlalchemy import create_engine, text

    from shared.testing import database_url

    engine = create_engine(database_url())
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO session_memories "
                    "(session_id, org_id, agent_id, memory_type, content) "
                    "VALUES (:s, 'sessbound-test', 'agent', 'observation', 'x')"
                ),
                {"s": brain_session_id(_Req("s" * 400), "org", "agent")},
            )
            conn.rollback()
    finally:
        engine.dispose()
