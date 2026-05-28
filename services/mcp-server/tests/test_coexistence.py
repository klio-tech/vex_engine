"""Coexistence guard: brain session-id namespaces are reserved & distinct.

This is deliverable (d) of the KB-F1 coexistence guard. Sub-tests (a)/(b)/(c)
— the SQL-level scope guards — live in
``services/shared/tests/test_coexistence.py``; they were kept in the shared
suite because they exercise ``shared.memory`` directly. This sub-test lives in
the mcp-server suite because the helpers it covers (``app.session`` and
``app.curator``) resolve cleanly only under the mcp-server import root.

The contamination this guards against is the OTHER direction of the safety
contract: even though brain rows live in the same ``session_memories`` table as
verification rows, a verification read — ``retrieve_memories(session_id=<real
conversation id>)`` — can NEVER match a brain row, because every brain write
namespaces its synthetic ``session_id`` under a reserved prefix that a real
conversation id will not collide with:

* MCP write tools (remember/observe/plan/decide/note) →
  ``klio-brain:<suffix>`` via :func:`app.session.brain_session_id`.
* The org curator → ``klio-curator:<org_id>`` (``f"{agent_id}:{org_id}"`` with
  the default ``agent_id`` :data:`app.curator.DEFAULT_CURATOR_AGENT_ID`).

These assertions document those two prefixes as reserved namespaces and fail
loudly if either drifts, since a drift toward an un-namespaced (or shorter,
collision-prone) id would re-open the door to verification reads matching brain
rows.
"""

from app.curator import DEFAULT_CURATOR_AGENT_ID
from app.session import brain_session_id

# Reserved synthetic session-id prefixes. A real conversation session id (the
# one verification's retrieve_memories filters on) is never expected to start
# with either of these, which is what keeps the two paths from colliding.
BRAIN_WRITE_PREFIX = "klio-brain:"
CURATOR_PREFIX = "klio-curator:"


class _FakeHeaders:
    """Minimal dict-like header carrier (only ``.get`` is read)."""

    def __init__(self, mapping: dict):
        self._mapping = dict(mapping)

    def get(self, key, default=None):
        return self._mapping.get(key, default)


class _FakeRequest:
    def __init__(self, headers: dict):
        self.headers = _FakeHeaders(headers)


def test_brain_write_session_id_uses_reserved_prefix():
    """Prevents verification→brain collision on the write-tool path.

    Every brain write tool derives its synthetic session_id via
    ``brain_session_id``; it must start with ``klio-brain:`` so a verification
    ``retrieve_memories(session_id=<real conversation id>)`` cannot match it.
    Covered both with and without the ``mcp-session-id`` header (the header and
    the principal-fallback branches).
    """
    with_header = brain_session_id(
        _FakeRequest({"mcp-session-id": "conn-1"}), "org-x", "agent-y"
    )
    assert with_header.startswith(BRAIN_WRITE_PREFIX), with_header
    assert with_header == "klio-brain:conn-1"

    no_header = brain_session_id(_FakeRequest({}), "org-x", "agent-y")
    assert no_header.startswith(BRAIN_WRITE_PREFIX), no_header
    assert no_header == "klio-brain:org-x:agent-y"


def test_curator_session_id_uses_reserved_prefix():
    """Prevents verification→brain collision on the curator path.

    The curator authors synthesised memories under
    ``session_id = f"{DEFAULT_CURATOR_AGENT_ID}:{org_id}"``. With the default
    agent id that renders ``klio-curator:<org>``, a reserved namespace distinct
    from real conversation ids and from the write-tool ``klio-brain:`` prefix —
    so neither can be matched by a verification session read.
    """
    assert DEFAULT_CURATOR_AGENT_ID == "klio-curator"

    # Mirror how curator._synthesize_and_store composes the session id.
    org_id = "org-1"
    curator_session_id = f"{DEFAULT_CURATOR_AGENT_ID}:{org_id}"
    assert curator_session_id.startswith(CURATOR_PREFIX), curator_session_id
    assert curator_session_id == "klio-curator:org-1"


def test_brain_and_curator_namespaces_are_distinct():
    """The two reserved brain namespaces never overlap with each other.

    Keeping the write-tool and curator prefixes distinct means dedup buckets
    and provenance stay separable, and neither can be confused for a real
    conversation session id consumed by the verification path.
    """
    assert BRAIN_WRITE_PREFIX != CURATOR_PREFIX
    assert not BRAIN_WRITE_PREFIX.startswith(CURATOR_PREFIX)
    assert not CURATOR_PREFIX.startswith(BRAIN_WRITE_PREFIX)
