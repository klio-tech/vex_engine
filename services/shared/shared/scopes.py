"""The memory-row scope value set, defined once.

``session_memories.scope`` is governed by ``ck_session_memories_scope``
(migration 031). Before this module the same set was also written out as a
tuple in ``memory.py`` twice, ``brain.py``, ``graph_store.py`` and
``promotion.py`` — six independent copies, none referring to another. A scope
added to five of them is accepted on write and invisible on read, which looks
like data loss rather than a validation gap.

Not to be confused with ``shared.auth.VALID_SCOPES``, which is the unrelated
set of API-key permission scopes.
"""

from __future__ import annotations

#: Every value ``session_memories.scope`` may hold. Must equal the
#: ``ck_session_memories_scope`` CHECK constraint; a test asserts it.
MEMORY_SCOPES = frozenset({"private", "project", "org", "session", "agent"})

#: Scopes a caller may ask a write to land at. 'session' is excluded because
#: ``write_memory`` hardcodes it on INSERT and ``write_scoped_memory`` then
#: UPDATEs the row to its real scope, so it is an implementation detail rather
#: than a destination.
WRITABLE_SCOPES = frozenset({"private", "project", "org", "agent"})

#: Scopes ``retrieve_memories_scoped`` accepts as its ``scope`` argument. It
#: rejects 'session' explicitly and reaches 'project' rows through the
#: membership subquery rather than by being asked for them.
#:
RECALL_SCOPES = frozenset({"org", "private", "agent"})

#: Read MODES ``retrieve_memories_scoped`` accepts. A superset of
#: RECALL_SCOPES: every column-valued scope, plus modes that SELECT existing
#: rows without being a column value themselves.
#:
#: The distinction is load-bearing. Every set above must satisfy "these are
#: values the column accepts" — a scope in one of them that the CHECK
#: constraint rejects is accepted on write and invisible on read, which is
#: what this module exists to prevent. 'agent_only' is not such a value: no
#: row is ever written with it, it selects ``scope='agent'`` rows, and so
#: MEMORY_SCOPES, WRITABLE_SCOPES and the ``ck_session_memories_scope``
#: constraint are all unchanged and no migration is needed.
#:
#: It exists because 'agent' unconditionally unions the org tier into the
#: ladder and so can only WIDEN a read. A consumer that partitions its end
#: users by the client-declared ``X-Vex-Agent`` needs a read that narrows.
RECALL_MODES = RECALL_SCOPES | frozenset({"agent_only"})

#: Destinations ``promote_memory`` will move a private row to. 'private' is
#: the source, never a destination.
PROMOTABLE_SCOPES = frozenset({"org", "project"})

#: Scopes the knowledge graph will attach edges for.
GRAPH_SCOPES = frozenset({"org", "agent"})
