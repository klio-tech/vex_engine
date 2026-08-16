"""The recall blend, the candidate pool, and the floor that may return nothing.

The floor is the part worth testing hardest, because its correct behaviour
looks like a bug: recall returning zero rows. That is the point. A session
about one project handed four confident memories about two others is worse
than a session handed none — the agent that gets nothing asks, and the agent
that gets confidently-wrong context builds on it.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from shared import ranking
from shared.embedding_space import SPACE_ENV, SPACE_V1, SPACE_V2


@dataclass
class _Record:
    """Just enough of a MemoryRecord for the floor: an id and a relevance."""

    id: str
    relevance: Optional[float]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        SPACE_ENV,
        ranking.SIMILARITY_WEIGHT_ENV,
        ranking.RECENCY_WEIGHT_ENV,
        ranking.CANDIDATE_POOL_FACTOR_ENV,
        ranking.MIN_CANDIDATE_POOL_ENV,
        ranking.MIN_RELEVANCE_ENV,
        ranking.RELATIVE_FLOOR_ENV,
        ranking.EF_SEARCH_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


# ── the blend ─────────────────────────────────────────────────────────────


def test_recency_is_a_tie_break_not_the_ranking():
    """The measured defect, as an inequality.

    Under the old 0.85/0.15 weights the recency term's range (0.15) exceeded
    similarity's entire measured range over the corpus (0.069), so freshness
    outranked relevance outright. Whatever the numbers become, this ordering
    is the property that must hold.
    """
    assert ranking.recency_weight() < ranking.similarity_weight()


def test_an_operator_can_retune_the_blend(monkeypatch):
    monkeypatch.setenv(ranking.RECENCY_WEIGHT_ENV, "0.3")
    assert ranking.recency_weight() == pytest.approx(0.3)


def test_an_out_of_range_weight_takes_the_default(monkeypatch):
    monkeypatch.setenv(ranking.RECENCY_WEIGHT_ENV, "7")
    assert ranking.recency_weight() == ranking.DEFAULT_RECENCY_WEIGHT


# ── the candidate pool ────────────────────────────────────────────────────


def test_a_small_limit_still_gets_a_pool_deep_enough_to_re_rank():
    """``limit * 4`` gave a default recall 32 candidates, and one measured
    production query under-filled to four rows. The floor is what stops a
    small limit from making the re-rank cosmetic."""
    assert ranking.candidate_pool(8) >= ranking.DEFAULT_MIN_CANDIDATE_POOL
    assert ranking.candidate_pool(8) > 8 * 4


def test_a_large_limit_scales_past_the_floor():
    assert ranking.candidate_pool(50) == 50 * ranking.DEFAULT_CANDIDATE_POOL_FACTOR


def test_ef_search_is_at_least_as_deep_as_the_default_pool():
    """pgvector will not return more candidates than its search list held, so
    a deeper pool without a deeper ``ef_search`` buys exactly nothing."""
    assert ranking.ef_search() >= ranking.DEFAULT_MIN_CANDIDATE_POOL


# ── the floor ─────────────────────────────────────────────────────────────


def test_v1_has_no_absolute_floor():
    """v1's whole corpus sits in a 0.08-wide band with the median at 0.77.

    There is no value in that band that separates a good row from a bad one,
    so any floor either keeps everything or empties production. Off is the
    only honest setting until the corpus is re-embedded.
    """
    assert ranking.min_relevance(SPACE_V1) == 0.0
    assert ranking.relative_floor(SPACE_V1) == 1.0


def test_v2_has_a_real_absolute_floor():
    assert 0.0 < ranking.min_relevance(SPACE_V2) < 1.0
    assert 0.0 < ranking.relative_floor(SPACE_V2) < 1.0


def test_the_floor_can_return_nothing_and_that_is_the_feature(monkeypatch):
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.9")
    rows = [_Record("a", 0.61), _Record("b", 0.60)]
    assert ranking.apply_relevance_floor(rows) == []


def test_the_relative_floor_never_empties_a_result(monkeypatch):
    """The best row is always within zero of itself, so it always survives.

    This is why there are two floors rather than one. The relative floor
    trims the tail and cannot silence recall; the absolute floor can, which is
    exactly why it is off in v1.
    """
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.0")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "0.001")
    rows = [_Record("a", 0.5), _Record("b", 0.2), _Record("c", 0.1)]
    kept = ranking.apply_relevance_floor(rows)
    assert [r.id for r in kept] == ["a"]


def test_the_absolute_floor_is_not_subsumed_by_the_relative_one(monkeypatch):
    """Both floors are load-bearing; neither implies the other.

    ``low`` is comfortably inside the relative window (0.84 is 0.06 below the
    best row, against a 0.10 delta) and is still dropped, because it is below
    the absolute bar. That is the case the relative floor cannot express: a
    result where everything is *consistently* mediocre, which is precisely
    what the un-prefixed corpus produced for every query.
    """
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.85")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "0.10")
    rows = [_Record("high", 0.90), _Record("mid", 0.86), _Record("low", 0.84)]
    kept = ranking.apply_relevance_floor(rows)
    assert [r.id for r in kept] == ["high", "mid"]


def test_rows_with_no_cosine_survive_alongside_scored_ones(monkeypatch):
    """A lexical-only row matched an exact literal and a graph row is anchored
    to a row that already passed. Neither has a cosine, and inventing one to
    compare against a cosine floor would be inventing the evidence."""
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.5")
    rows = [_Record("vector", 0.9), _Record("lexical", None)]
    kept = ranking.apply_relevance_floor(rows)
    assert [r.id for r in kept] == ["vector", "lexical"]


def test_unscored_rows_do_not_survive_a_total_wipeout(monkeypatch):
    """When every scored row fails the floor, the search found nothing, and
    handing back its by-products would present them as results."""
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.99")
    rows = [_Record("vector", 0.5), _Record("lexical", None)]
    assert ranking.apply_relevance_floor(rows) == []


def test_a_result_with_no_scored_rows_at_all_is_left_alone(monkeypatch):
    """Nothing to measure against, so nothing is judged. This is the
    graph-expansion-only shape, and dropping it would be a floor deciding
    about rows it has no evidence on."""
    monkeypatch.setenv(SPACE_ENV, "v2")
    rows = [_Record("graph-a", None), _Record("graph-b", None)]
    assert len(ranking.apply_relevance_floor(rows)) == 2


def test_the_floor_returns_a_new_list_and_preserves_order(monkeypatch):
    monkeypatch.setenv(SPACE_ENV, "v2")
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.0")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "1.0")
    rows = [_Record("a", 0.9), _Record("b", 0.1), _Record("c", 0.5)]
    kept = ranking.apply_relevance_floor(rows)
    assert kept is not rows
    assert [r.id for r in kept] == ["a", "b", "c"]
