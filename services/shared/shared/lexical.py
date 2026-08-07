"""The lexical recall arm: one full-text expression, its configuration, and
Reciprocal Rank Fusion.

Recall was a single retrieval strategy — an HNSW cosine fetch re-ranked by
recency and expanded one hop through the knowledge graph. Dense vectors are
weakest exactly where an agent's queries concentrate: a caller looking for
``KLIO_METRICS_TOKEN`` or ``unsupported startup parameter`` wants a rare
literal, not a paraphrase, and a rare token has no private direction in the
embedding space — it is compressed toward the neighbourhood of the tokens it
resembles. The row that actually contains the identifier can therefore rank
below several rows that are merely about the same subject.

This module holds the three pieces of that fix which are worth isolating:

**The expression, defined once.** Postgres matches an expression index by
expression *equality*. If the ``CREATE INDEX`` and the query disagree by so
much as the character bound, the index is silently never used and the only
symptom is a sequential scan. :func:`fts_document_expression` is the single
definition, imported by migration ``037`` and by ``shared.memory`` alike, so
there is no second copy that can drift.

**Why the expression is bounded.** ``content`` is unbounded ``TEXT`` — the
existing truncation applies only to embedding *input*, never to stored
content. Postgres refuses a ``tsvector`` over 1 MB, so an index on
``to_tsvector('english', content)`` would turn the first write of a large
artifact from something that works today into a hard ``string is too long for
tsvector`` failure. Measured, not assumed: an insert of 120k distinct tokens
raised at 1,477,986 bytes against the 1,048,575 limit, and succeeded under
``left(content, 100000)`` while still storing all 1,088,894 characters.
Adding search must not break writing.

**Fusion by rank, not by score.** Cosine relevance is bounded in [0, 1];
``ts_rank`` is unbounded and corpus-dependent, so it drifts upward as data
accumulates. Blending them with fixed weights needs a normalisation that
decays invisibly and shows up months later as a ranking regression nobody can
date. RRF uses only ordinal position, so the arms never need a common scale::

    fused(row) = Σ_arms  weight_arm / (k + rank_arm(row))

The invariant that makes the change safe to review falls straight out of that
formula: over a single ranked list, RRF is a strictly decreasing function of
rank, so when the lexical arm contributes nothing the fused order is
*identical* to the pre-change order. Empty query, stopword-only query, kill
switch off, lexical failure — every one of those lands on the old behaviour by
construction rather than by care.

**What the weight controls.** The arm does two distinct things, and the weight
decides whether it does both. It always **reorders the dense candidate pool**,
promoting a row the dense ranking buried: that is the demonstrated case, where
the row containing the literal sat sixth of six and now comes first. Whether it
can also **introduce a row from outside that pool** depends on arithmetic. A
lexical-only row scores ``weight/(k + 1)`` against a dense floor of
``1/(k + pool)``, so below roughly 1.0 it loses to the worst dense candidate and
is truncated away. ``DEFAULT_LEXICAL_WEIGHT`` explains why the default is 1.0
and what measuring 0.5 against production showed. Both behaviours are tested.

Dependencies are stdlib plus :mod:`shared.env_config` (itself stdlib-only) so
that the migrations image, which installs neither SQLAlchemy models nor
litellm, can import this module for the one constant it needs.
"""

from __future__ import annotations

import os
from typing import Hashable, Iterable, Sequence, TypeVar

from shared.env_config import parse_bool, parse_float, parse_int

__all__ = [
    "FTS_CONFIG",
    "FTS_CONTENT_LIMIT",
    "FTS_INDEX_NAME",
    "VECTOR_WEIGHT",
    "ENABLED_ENV",
    "WEIGHT_ENV",
    "RRF_K_ENV",
    "fts_document_expression",
    "fts_query_expression",
    "lexical_enabled",
    "lexical_weight",
    "rrf_k",
    "rrf_scores",
    "reciprocal_rank_fusion",
]

#: The text search configuration. ``english`` and ``simple`` were measured
#: against pg16 and tokenise identifiers identically — both split
#: ``KLIO_METRICS_TOKEN`` into ``klio``/``metrics``/``token`` and both keep
#: ``/healthz`` whole. They differ only in stemming, which is a pure gain on
#: the prose half of the corpus (``decision`` matching ``decisions``) and costs
#: nothing on the identifier half because tokenisation is symmetric across
#: query and document.
FTS_CONFIG = "english"

#: The character bound on the indexed expression. See the module docstring:
#: this is what keeps an oversized write working. Changing it requires a
#: migration that rebuilds the index, because the query expression must stay
#: byte-identical to the indexed one.
FTS_CONTENT_LIMIT = 100_000

#: The index created by migration ``037``.
FTS_INDEX_NAME = "ix_session_memories_content_fts"

#: The dense arm's RRF weight. Fixed at 1.0 — the lexical arm's weight is
#: expressed relative to it, so there is only one knob to reason about.
VECTOR_WEIGHT = 1.0

ENABLED_ENV = "KLIO_LEXICAL_RECALL_ENABLED"
WEIGHT_ENV = "KLIO_LEXICAL_WEIGHT"
RRF_K_ENV = "KLIO_RRF_K"

DEFAULT_LEXICAL_WEIGHT = 1.0
"""Equal to the dense arm's weight, which is the conventional RRF choice.

Set to 0.5 first, and measured against production before shipping. At 0.5 a row
present only in the lexical arm scores ``0.5/61 = 0.0082`` while *every* row the
dense arm returned scores at least ``1/(60 + pool)``, so a lexical-only row lost
to the worst dense candidate and could never survive the final truncation. The
arm could reorder the dense pool but never add to it, which is the one thing it
exists to do.

That is not theoretical. Against production, ``pool_pre_ping`` matched 17 active
rows in the caller's own org and none of them reached the results.

At 1.0 a lexical rank-1 row ties the dense arm's own leader, so a row the dense
arm missed entirely can surface. The single-arm invariant is untouched: when the
lexical arm returns nothing the fused order is still identical to the dense
order, because that property depends on there being one list, not on the weights.
"""

DEFAULT_RRF_K = 60


def fts_document_expression(column: str = "content") -> str:
    """The indexed document expression. Must be identical in index and query."""
    return f"to_tsvector('{FTS_CONFIG}', left({column}, {FTS_CONTENT_LIMIT}))"


def fts_query_expression(parameter: str) -> str:
    """The query expression, binding ``:parameter`` rather than interpolating.

    ``websearch_to_tsquery`` and not ``to_tsquery``: the latter raises a syntax
    error on arbitrary input, and this text arrives from an LLM. A stray ``&``
    must not turn a recall into a 500. ``websearch_to_tsquery`` never raises —
    it degrades to an empty tsquery, which matches nothing, which is the
    single-arm invariant above.
    """
    return f"websearch_to_tsquery('{FTS_CONFIG}', :{parameter})"


# ── configuration ─────────────────────────────────────────────────────────
#
# Read per call rather than frozen at import: an operator who flips the kill
# switch should be believed by a process that has already imported this
# module. The cost is one dict lookup per recall.


def lexical_enabled() -> bool:
    """Whether the lexical arm runs at all. Default on; kill switch."""
    return parse_bool(os.environ.get(ENABLED_ENV), default=True, name=ENABLED_ENV)


def lexical_weight() -> float:
    """The lexical arm's RRF weight, in [0.0, 1.0]. Default 0.5.

    Below the vector arm's 1.0 deliberately: lexical is a corrective signal,
    not the primary one. ``0.0`` is a legitimate value meaning "off", and
    :func:`shared.env_config.parse_float` returns it as such — the bug this
    guards against is ``float(...) or DEFAULT``, which treats the operator's
    explicit zero as absence and turns the arm back on at half weight.
    """
    return parse_float(
        os.environ.get(WEIGHT_ENV),
        default=DEFAULT_LEXICAL_WEIGHT,
        name=WEIGHT_ENV,
        minimum=0.0,
        maximum=1.0,
    )


def rrf_k() -> int:
    """The RRF damping constant, in [1, 1000]. Default 60, the conventional
    value: it keeps a row ranked first in one arm from being swamped by a row
    ranked tenth in both."""
    return parse_int(
        os.environ.get(RRF_K_ENV),
        default=DEFAULT_RRF_K,
        name=RRF_K_ENV,
        minimum=1,
        maximum=1000,
    )


# ── fusion ────────────────────────────────────────────────────────────────

K = TypeVar("K", bound=Hashable)


def rrf_scores(
    arms: Iterable[tuple[float, Sequence[K]]], *, k: int
) -> "dict[K, float]":
    """``{id: Σ weight / (k + rank)}`` over every arm, ranks starting at 1.

    An id repeated inside one arm keeps its best (lowest) rank; a duplicate
    should not exist, and if one does it must not be paid for twice.
    """
    scores: dict[K, float] = {}
    for weight, ranked in arms:
        if weight == 0.0:
            # Not merely a fast path: an arm at zero weight must be
            # indistinguishable from an absent one, including for tie-breaking.
            continue
        seen: set[K] = set()
        for position, identifier in enumerate(ranked, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + position)
    return scores


def reciprocal_rank_fusion(
    arms: Sequence[tuple[float, Sequence[K]]], *, k: int
) -> "list[K]":
    """Fuse ranked lists into one order, best first.

    Ties are broken by first appearance, scanning the arms in the order given
    and each arm best-rank first. Because the vector arm is passed first, an
    exact tie resolves the way it resolved before this change existed. The sort
    is Python's, which is stable, so that first-appearance order survives.
    """
    scores = rrf_scores(arms, k=k)
    if not scores:
        return []

    order: list[K] = []
    seen: set[K] = set()
    for weight, ranked in arms:
        if weight == 0.0:
            continue
        for identifier in ranked:
            if identifier not in seen:
                seen.add(identifier)
                order.append(identifier)

    return sorted(order, key=lambda identifier: -scores[identifier])
