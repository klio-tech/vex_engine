"""Which vector column recall reads, and what prefix each side of the pair gets.

The production embedding model is ``intfloat/multilingual-e5-large-instruct``.
E5 is trained **asymmetrically**: a document must be embedded as
``passage: <text>`` and a search string as ``query: <text>``. The engine sent
bare text for both, on both sides, for the whole life of the corpus. The
measured consequence, over the 18,367 org-visible memories in production:

===================================  ======  ======  ======  ======
query                                   max     p99     p90  median
===================================  ======  ======  ======  ======
how does the proxy inject memories   0.8538  0.8185  0.7971  0.7728
what did we decide about defaults    0.8429  0.8154  0.7975  0.7754
Vedic astrology dasha calculation    0.8528  0.7911  0.7666  0.7431
===================================  ======  ======  ======  ======

The single best match out of 18,367 rows beats a *randomly chosen* one by
0.085 cosine. There is no signal to rank on, no threshold that could mean
anything, and a recency term worth 0.15 trivially dominates a similarity term
whose entire dynamic range is 0.069.

**Why this needs two columns and not one.** Fixing the prefixes changes the
vector for every text. A prefixed query scored against an unprefixed corpus is
not "slightly stale" — it is a different space, and it ranks *worse* than the
consistent-but-wrong status quo. So there is a window, between the first
prefixed query and the last re-embedded row, in which recall is broken. The
only question is how long that window is, and it was measured rather than
guessed: 63,196 embedded rows, and the production LiteLLM gateway sustains
~15.7 embeddings/second at concurrency 16 (0.68 s p50 per call). That is
**≈67 minutes** of re-embedding. An hour of knowingly-degraded recall over
somebody's live memory is not an acceptable deploy window, so the corpus is
re-embedded into a SECOND column and the read path switches atomically when it
is full.

Hence two *spaces*:

``v1``
    column ``embedding``, no prefixes. What every existing row holds. The
    default, so importing this module changes nothing until an operator says
    otherwise.
``v2``
    column ``embedding_v2`` (migration 046), ``passage: ``/``query: ``
    prefixes. Populated by :mod:`scripts.run_embedding_v2_backfill` and by
    every write once the column exists.

``KLIO_EMBEDDING_SPACE`` selects the **read** space. Writes populate both
whenever :func:`dual_write_enabled` (default on), which is what makes the
switch atomic in one direction and reversible in the other: flipping to ``v2``
needs no data movement, and flipping back to ``v1`` finds ``embedding`` still
current rather than a month stale.

Scope note — this covers ``session_memories``, the recall corpus, and nothing
else. ``work_claims`` and ``entities`` embed content against content, so both
sides of those comparisons carry the same prefix or neither, and the asymmetry
this module exists to fix cannot arise there.
"""

from __future__ import annotations

import os

from shared.env_config import parse_bool, parse_str

__all__ = [
    "SPACE_V1",
    "SPACE_V2",
    "SPACES",
    "ROLE_QUERY",
    "ROLE_PASSAGE",
    "ROLES",
    "V1_COLUMN",
    "V2_COLUMN",
    "SPACE_ENV",
    "DUAL_WRITE_ENV",
    "active_space",
    "column_for",
    "dual_write_enabled",
    "prefix_for",
    "apply_role",
    "write_spaces",
]

SPACE_V1 = "v1"
SPACE_V2 = "v2"
SPACES = (SPACE_V1, SPACE_V2)

ROLE_QUERY = "query"
ROLE_PASSAGE = "passage"
ROLES = (ROLE_QUERY, ROLE_PASSAGE)

V1_COLUMN = "embedding"
V2_COLUMN = "embedding_v2"

_COLUMNS = {SPACE_V1: V1_COLUMN, SPACE_V2: V2_COLUMN}

#: E5's mandatory asymmetric prefixes, trailing space included. These are the
#: model's own tokens, not a formatting choice — ``query:`` without the space
#: is a different tokenisation and does not get the trained behaviour.
_V2_PREFIXES = {ROLE_QUERY: "query: ", ROLE_PASSAGE: "passage: "}

SPACE_ENV = "KLIO_EMBEDDING_SPACE"
DUAL_WRITE_ENV = "KLIO_EMBEDDING_DUAL_WRITE"


def active_space() -> str:
    """The space recall READS. ``v1`` unless an operator says otherwise.

    Read per call rather than frozen at import, matching
    :func:`shared.lexical.lexical_enabled`: an operator who flips the switch
    should be believed by a process that already imported this module, because
    the alternative is a restart of every service to change a read path that
    is deliberately designed to be flipped and un-flipped.
    """
    return parse_str(
        os.environ.get(SPACE_ENV), default=SPACE_V1, allowed=SPACES, name=SPACE_ENV
    )


def dual_write_enabled() -> bool:
    """Whether a write populates the non-active space as well. Default on.

    Off is a deliberate post-migration state — once ``embedding`` is dropped
    there is no second column to fill — and NOT a way to save an embedding
    call during the transition. Turning it off while the read space is still
    switchable makes the switch one-way: the space nobody is writing goes
    stale from that moment, and rolling back lands on a corpus missing every
    memory written since.
    """
    return parse_bool(os.environ.get(DUAL_WRITE_ENV), default=True, name=DUAL_WRITE_ENV)


def column_for(space: str | None = None) -> str:
    """The ``session_memories`` vector column backing *space*.

    Returns a module-local literal from a closed set, never caller text: this
    value is interpolated into SQL (a column name cannot be a bind parameter),
    so an unknown space resolves through :func:`active_space`'s validation
    rather than reaching a query.
    """
    return _COLUMNS[space or active_space()]


def prefix_for(role: str, space: str | None = None) -> str:
    """The prefix for *role* in *space*; ``""`` in ``v1``, which has none.

    Raises:
        ValueError: on an unknown role. A typo here is silent otherwise — the
            text embeds fine, just into the wrong half of the space, and the
            only symptom is recall quality nobody can attribute.
    """
    if role not in ROLES:
        raise ValueError(f"unknown embedding role: {role!r}")
    if (space or active_space()) != SPACE_V2:
        return ""
    return _V2_PREFIXES[role]


def apply_role(text_input: str, role: str, space: str | None = None) -> str:
    """*text_input* prefixed for *role*. Identity in ``v1``."""
    return prefix_for(role, space) + text_input


def write_spaces() -> tuple[str, ...]:
    """Spaces a write must populate: the active one first, then the other.

    Active-first is load-bearing. The caller treats the FIRST space as
    required and the rest as best-effort, because a row missing its active
    vector is invisible to recall while a row missing the inactive one is
    merely un-prepared for a switch that has not happened.
    """
    active = active_space()
    if not dual_write_enabled():
        return (active,)
    return (active,) + tuple(s for s in SPACES if s != active)
