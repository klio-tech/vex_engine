"""Recall ranking: how the blend is weighted, how deep it looks, and when it
returns nothing at all.

Three knobs used to be three magic numbers inlined in one SQL string, and each
of them was wrong in a way the others hid.

**The blend.** ``0.85 * cosine + 0.15 * recency`` looks like similarity
dominating four-to-one. Measured against production it is the opposite:
similarity's entire dynamic range, best-in-corpus to median, is
``0.85 * (0.854 - 0.773) = 0.069``, while recency's range, brand-new to
thirty-days-old, is the full ``0.15``. Each day of age was worth 0.0059 of
cosine, so a mediocre memory written last week outranked the best memory in
the database if the latter was ten days older. Freshness was worth more than
relevance, in a system whose entire job is relevance.

Two changes, and they only work together:

1. **Normalise similarity inside the candidate pool** — ``(sim - min) /
   (max - min)`` over the rows the ANN stage actually returned. This is what
   makes the weight mean what it says. Absolute cosine in a corpus this
   compressed carries almost no information; *relative* cosine among the
   nearest neighbours carries all of it.
2. **Cut the recency weight to 0.05.** Not to zero: the tie-break between two
   equally-relevant memories should be the newer one, and in a corpus that
   stores "the container is running on port 54422" as a durable fact, that
   tie-break earns its keep. But it must be a tie-break, not the ranking.

**The pool.** ``limit * 4`` = 32 candidates for a default recall. Recency
could only ever reorder 32 rows the pure-cosine ANN had already chosen, so the
blend did far less than the formula implied, and with selective scope filters
one measured production query under-filled to *four* rows. The pool is now
``max(64, limit * 8)`` and ``hnsw.ef_search`` is raised to match — the ANN
index will not return more candidates than its own search list holds, so
raising the pool without raising ``ef_search`` buys nothing.

**The floor.** There was none, anywhere. Recall returned ``limit`` rows
however bad they were, which is how a session about the Klio proxy was handed
four memories about an unrelated agent's core-memory setting, all scoring 0.85
— indistinguishable, on the number, from a real match.

Two floors, because they fail in opposite directions. An **absolute** floor
(``relevance >= min``) is the one that can silence recall entirely when a
corpus or model shifts, so it is off by default in ``v1``, whose absolute
numbers are meaningless. A **relative** floor (``relevance >= best - delta``)
cannot silence recall — the best row always survives it — and is what removes
the long tail of "nearly as good as the best, which was itself bad". Both are
per-space, because ``v1``'s 0.08-wide band and ``v2``'s prefixed band are not
comparable and one set of numbers cannot serve both.

**Returning zero rows is a feature.** Every caller must treat an empty recall
as a normal answer. Injecting nothing beats injecting four wrong things: the
agent then asks, instead of confidently building on somebody else's project.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from shared.embedding_space import SPACE_V1, SPACE_V2, active_space
from shared.env_config import parse_float, parse_int

__all__ = [
    "SIMILARITY_WEIGHT_ENV",
    "RECENCY_WEIGHT_ENV",
    "CANDIDATE_POOL_FACTOR_ENV",
    "MIN_CANDIDATE_POOL_ENV",
    "EF_SEARCH_ENV",
    "MIN_RELEVANCE_ENV",
    "RELATIVE_FLOOR_ENV",
    "RECENCY_WINDOW_SECONDS",
    "similarity_weight",
    "recency_weight",
    "candidate_pool",
    "ef_search",
    "min_relevance",
    "relative_floor",
    "apply_relevance_floor",
]

SIMILARITY_WEIGHT_ENV = "KLIO_RECALL_SIMILARITY_WEIGHT"
RECENCY_WEIGHT_ENV = "KLIO_RECALL_RECENCY_WEIGHT"
CANDIDATE_POOL_FACTOR_ENV = "KLIO_RECALL_CANDIDATE_POOL_FACTOR"
MIN_CANDIDATE_POOL_ENV = "KLIO_RECALL_MIN_CANDIDATE_POOL"
EF_SEARCH_ENV = "KLIO_RECALL_EF_SEARCH"
MIN_RELEVANCE_ENV = "KLIO_RECALL_MIN_RELEVANCE"
RELATIVE_FLOOR_ENV = "KLIO_RECALL_RELATIVE_FLOOR"

#: The recency ramp, unchanged: a row loses its whole freshness bonus over
#: thirty days and never goes negative.
RECENCY_WINDOW_SECONDS = 30 * 86400

DEFAULT_SIMILARITY_WEIGHT = 0.95
DEFAULT_RECENCY_WEIGHT = 0.05
DEFAULT_CANDIDATE_POOL_FACTOR = 8
DEFAULT_MIN_CANDIDATE_POOL = 64

#: ``hnsw.ef_search`` for the recall statement. pgvector's default is 40, which
#: is below the candidate pool this module now asks for — the index cannot
#: return more rows than its search list held, so the pool would silently
#: under-fill exactly as it did in production.
DEFAULT_EF_SEARCH = 200

#: Absolute cosine floors, per space.
#:
#: ``v1`` is 0.0 — OFF. Its whole corpus lives between 0.68 and 0.87 with the
#: median at 0.77, so any floor that excludes a bad row excludes nearly every
#: row, and any floor that keeps the good rows keeps everything. There is no
#: value here that means anything, and picking one anyway is how a floor
#: silently empties a production recall.
#:
#: ``v2`` is 0.80. E5's prefixed space separates: measured over a 4,000-row
#: production sample, an on-topic row scores 0.86-0.89 against its query while
#: the median row scores 0.78 and off-project noise sits at 0.79-0.82. 0.80 is
#: below every on-topic match measured and above the bulk of the corpus.
_MIN_RELEVANCE_DEFAULTS = {SPACE_V1: 0.0, SPACE_V2: 0.80}

#: Relative floors, per space: a row is dropped when it scores more than this
#: far below the best row in the same result. Cannot empty a recall.
#:
#: ``v1`` is 1.0 — effectively off, for the same reason as above: its whole
#: band is 0.085 wide, so any delta small enough to discriminate is inside the
#: noise.
#:
#: ``v2`` is 0.06, which is roughly the gap the same sample shows between an
#: on-topic cluster and the first off-topic row.
_RELATIVE_FLOOR_DEFAULTS = {SPACE_V1: 1.0, SPACE_V2: 0.06}


def similarity_weight() -> float:
    """Weight on the pool-normalised similarity term, in [0, 1]."""
    return parse_float(
        os.environ.get(SIMILARITY_WEIGHT_ENV),
        default=DEFAULT_SIMILARITY_WEIGHT,
        name=SIMILARITY_WEIGHT_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def recency_weight() -> float:
    """Weight on the recency term, in [0, 1]. A tie-break, not a ranking."""
    return parse_float(
        os.environ.get(RECENCY_WEIGHT_ENV),
        default=DEFAULT_RECENCY_WEIGHT,
        name=RECENCY_WEIGHT_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def candidate_pool(limit: int) -> int:
    """How many ANN neighbours the blend gets to re-rank.

    ``max(floor, limit * factor)`` rather than a bare multiple: the floor is
    what stops a small ``limit`` from producing a pool too shallow for the
    re-rank to do anything, which is the shape the production measurement
    found (``limit=8`` → 32 candidates → one query under-filled to 4 rows).
    """
    factor = parse_int(
        os.environ.get(CANDIDATE_POOL_FACTOR_ENV),
        default=DEFAULT_CANDIDATE_POOL_FACTOR,
        name=CANDIDATE_POOL_FACTOR_ENV,
        minimum=1,
        maximum=64,
    )
    floor = parse_int(
        os.environ.get(MIN_CANDIDATE_POOL_ENV),
        default=DEFAULT_MIN_CANDIDATE_POOL,
        name=MIN_CANDIDATE_POOL_ENV,
        minimum=1,
        maximum=4096,
    )
    return max(floor, limit * factor)


def ef_search() -> int:
    """``hnsw.ef_search`` for the recall statement. ``0`` leaves it alone."""
    return parse_int(
        os.environ.get(EF_SEARCH_ENV),
        default=DEFAULT_EF_SEARCH,
        name=EF_SEARCH_ENV,
        minimum=0,
        maximum=1000,
    )


def min_relevance(space: Optional[str] = None) -> float:
    """Absolute cosine floor for *space*. ``0.0`` disables it."""
    return parse_float(
        os.environ.get(MIN_RELEVANCE_ENV),
        default=_MIN_RELEVANCE_DEFAULTS[space or active_space()],
        name=MIN_RELEVANCE_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def relative_floor(space: Optional[str] = None) -> float:
    """Max cosine gap below the best row. ``1.0`` disables it."""
    return parse_float(
        os.environ.get(RELATIVE_FLOOR_ENV),
        default=_RELATIVE_FLOOR_DEFAULTS[space or active_space()],
        name=RELATIVE_FLOOR_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def apply_relevance_floor(
    records: Sequence[Any], *, space: Optional[str] = None
) -> "list[Any]":
    """Drop rows that are not relevant enough to be worth injecting.

    Both floors are measured against ``record.relevance``, the raw cosine
    similarity. A record whose ``relevance`` is ``None`` is **kept**: that is
    the positive marker for a row the lexical arm alone returned (an exact
    token match, which is evidence of its own kind) or one graph expansion
    added (anchored to a row that already passed). Neither has a cosine score,
    and inventing one to compare against a cosine floor would be inventing the
    evidence the floor is supposed to weigh.

    The two floors are independent, and neither implies the other. The
    relative floor cannot express "everything here is consistently mediocre",
    which is exactly the shape the un-prefixed corpus produced for every
    query; the absolute floor cannot express "one row is far better than the
    rest, so drop the rest". Both are applied, absolute first.

    Returns:
        A new list, order preserved. **Possibly empty**, and that is the
        intended outcome rather than a degradation: see the module docstring.
    """
    resolved = space or active_space()
    absolute = min_relevance(resolved)
    delta = relative_floor(resolved)

    scored = [r for r in records if getattr(r, "relevance", None) is not None]
    if not scored:
        return list(records)

    kept = [
        r
        for r in records
        if getattr(r, "relevance", None) is None or r.relevance >= absolute
    ]

    surviving_scores = [
        r.relevance for r in kept if getattr(r, "relevance", None) is not None
    ]
    if not surviving_scores:
        # Every scored row failed the absolute floor. Whatever is left carries
        # no cosine at all, and a lexical-only or graph row has no anchor to
        # be relative to — returning it alone would be presenting the
        # by-products of a search that found nothing as if they were results.
        return []

    best = max(surviving_scores)
    return [
        r
        for r in kept
        if getattr(r, "relevance", None) is None or r.relevance >= best - delta
    ]
