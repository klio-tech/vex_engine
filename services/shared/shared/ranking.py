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

Two floors, because they fail in opposite directions, and their defaults are
set from measurement rather than taste.

An **absolute** floor (``relevance >= min``) is the one that can silence recall
entirely, and it **ships off**. That is not caution, it is what the numbers
say: over an 11,648-row production sample, a control query about a subject the
corpus contains nothing on ("the mating habits of the Patagonian toothfish")
still found a best match at **0.80**, while real queries' best matches ran
0.78–0.90. The two distributions overlap. There is no absolute value that
rejects the nonsense query without also rejecting real ones, so any number
picked here would be a floor that silently empties production for some queries
and admits garbage for others. The knob exists, documented, at 0.0.

A **relative** floor (``relevance >= best - delta``) cannot silence recall —
the best row always survives it — and it is what actually works on this
corpus. It removes the long tail of "nearly as good as the best, which was
itself mediocre", which is exactly the shape of the reported leak: four rows
within 0.01 of each other, none of them about the session's subject.

**Returning zero rows is a feature.** Every caller must treat an empty recall
as a normal answer. Injecting nothing beats injecting four wrong things: the
agent then asks, instead of confidently building on somebody else's project.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

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

#: The absolute cosine floor. OFF by default — see the module docstring for
#: the measurement, and note that "off" here is a finding rather than a
#: placeholder. Set it only against numbers from
#: ``scripts/eval_recall_ranking.py`` on the corpus it will run over.
DEFAULT_MIN_RELEVANCE = 0.0

#: The relative floor: a row is dropped when it scores more than this far below
#: the best row in the same result. Cannot empty a recall.
#:
#: 0.04 is measured, not chosen. Across the evaluation's seventeen queries the
#: gap between the best row and the first row that is plainly about something
#: else averages 0.03–0.05 once the project fence is applied; below 0.03 the
#: floor starts trimming rows a human would keep, and above 0.06 it stops
#: trimming the tail it exists for.
DEFAULT_RELATIVE_FLOOR = 0.04


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


def min_relevance() -> float:
    """Absolute cosine floor. ``0.0`` disables it, and is the default."""
    return parse_float(
        os.environ.get(MIN_RELEVANCE_ENV),
        default=DEFAULT_MIN_RELEVANCE,
        name=MIN_RELEVANCE_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def relative_floor() -> float:
    """Max cosine gap below the best row. ``1.0`` disables it."""
    return parse_float(
        os.environ.get(RELATIVE_FLOOR_ENV),
        default=DEFAULT_RELATIVE_FLOOR,
        name=RELATIVE_FLOOR_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def apply_relevance_floor(
    records: Sequence[Any], *, exempt_ids: "Optional[set[str]]" = None
) -> "list[Any]":
    """Drop rows that are not relevant enough to be worth injecting.

    Both floors are measured against ``record.relevance``, the raw cosine
    similarity. A record whose ``relevance`` is ``None`` is **kept**: that is
    the positive marker for a row the lexical arm alone returned (an exact
    token match, which is evidence of its own kind) or one graph expansion
    added (anchored to a row that already passed). Neither has a cosine score,
    and inventing one to compare against a cosine floor would be inventing the
    evidence the floor is supposed to weigh.

    *exempt_ids* is the same argument for rows that have a cosine AND lexical
    evidence, and it is not a nicety — without it this function undoes the
    lexical arm entirely. That arm exists because a rare literal has no private
    direction in the embedding space: the row containing ``KLIO_METRICS_TOKEN``
    ranks LAST by cosine among rows merely about tokens, which is precisely
    the position a cosine floor deletes. ``websearch_to_tsquery`` ANDs its
    terms, so a lexical hit means the row contains every word of the query —
    strong evidence, arriving through a channel this floor cannot see.

    The two floors are independent, and neither implies the other. The
    relative floor cannot express "everything here is consistently mediocre",
    which is exactly the shape the un-prefixed corpus produced for every
    query; the absolute floor cannot express "one row is far better than the
    rest, so drop the rest". Both are applied, absolute first.

    Returns:
        A new list, order preserved. **Possibly empty**, and that is the
        intended outcome rather than a degradation: see the module docstring.
    """
    absolute = min_relevance()
    delta = relative_floor()
    exempt = exempt_ids or set()

    def _judged(record: Any) -> bool:
        """Whether this row is subject to the floors at all."""
        return (
            getattr(record, "relevance", None) is not None
            and str(getattr(record, "id", "")) not in exempt
        )

    scored = [r for r in records if _judged(r)]
    if not scored:
        return list(records)

    kept = [r for r in records if not _judged(r) or r.relevance >= absolute]

    surviving_scores = [r.relevance for r in kept if _judged(r)]
    if not surviving_scores:
        # Every JUDGED row failed the absolute floor, so there is no ``best``
        # for the relative floor to be relative to. The rows that were never
        # judged still stand on their own evidence and are returned:
        #
        #   * an exempt row has a lexical hit — the query's every word appears
        #     in it — which is the strongest signal recall has, and it is
        #     precisely the row whose cosine is LOW (a rare literal has no
        #     private direction in the embedding space). Dropping it here
        #     would undo the exemption twenty lines above it, and would do so
        #     only when an operator raises ``KLIO_RECALL_MIN_RELEVANCE`` — so
        #     the symptom is "exact-literal search stopped working", weeks
        #     after the knob was turned, with the knob the last place anyone
        #     looks;
        #   * a ``relevance is None`` row came from the lexical arm alone or
        #     from graph expansion, and the floors have never judged those.
        #
        # What is NOT returned is a judged row that failed the bar. An empty
        # list is still the answer when nothing but failing cosines was found.
        return [r for r in kept if not _judged(r)]

    best = max(surviving_scores)
    return [r for r in kept if not _judged(r) or r.relevance >= best - delta]
