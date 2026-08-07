"""Reciprocal Rank Fusion and the lexical arm's configuration.

Database-free on purpose. Fusion is pure arithmetic over ranks, and the one
property that makes the whole change safe to review — *with the lexical arm
empty the fused order is exactly today's order* — is a property of that
arithmetic alone. Pinning it here means it is checked on every run, not only
when a Postgres happens to be reachable.
"""

import pytest

from shared import lexical


# ── the single-arm invariant ──────────────────────────────────────────────


def test_single_arm_order_is_preserved_over_a_long_permutation():
    """One non-empty arm → the fused order IS that arm's order.

    Asserted over a twelve-element permutation deliberately unrelated to any
    natural ordering. On a two-element list a broken implementation has a 50%
    chance of agreeing by accident; here it has one in 479,001,600.
    """
    ranked = ["k", "c", "j", "a", "f", "b", "l", "e", "h", "d", "i", "g"]

    fused = lexical.reciprocal_rank_fusion([(1.0, ranked)], k=60)

    assert fused == ranked


def test_empty_second_arm_changes_nothing():
    """The lexical arm returning no rows must not perturb the vector order."""
    ranked = ["k", "c", "j", "a", "f", "b", "l", "e", "h", "d", "i", "g"]

    fused = lexical.reciprocal_rank_fusion([(1.0, ranked), (0.5, [])], k=60)

    assert fused == ranked


def test_zero_weighted_arm_changes_nothing():
    """A weight of 0.0 is 'off', and off must be indistinguishable from empty."""
    ranked = ["k", "c", "j", "a", "f", "b", "l", "e", "h", "d", "i", "g"]

    fused = lexical.reciprocal_rank_fusion(
        [(1.0, ranked), (0.0, ["g", "i", "d"])], k=60
    )

    assert fused == ranked


# ── the formula ───────────────────────────────────────────────────────────


def test_scores_are_the_weighted_reciprocal_ranks():
    scores = lexical.rrf_scores([(1.0, ["a", "b"]), (0.5, ["b", "a"])], k=60)

    assert scores["a"] == pytest.approx(1.0 / 61 + 0.5 / 62)
    assert scores["b"] == pytest.approx(1.0 / 62 + 0.5 / 61)


def test_a_row_only_the_lexical_arm_found_can_outrank_a_deep_vector_row():
    """The point of the feature, expressed as arithmetic.

    Lexical rank 1 at weight 0.5 scores 0.5/61 ≈ 0.0082; vector rank 6 scores
    1/66 ≈ 0.0152 — so a *deep* vector row still wins. Lexical rank 1 beats
    vector rank 6 only once the row is absent from the vector arm entirely and
    the vector arm's own tail is longer than k allows. What it does reliably
    beat is nothing at all: a row no arm but lexical returned is now present.
    """
    fused = lexical.reciprocal_rank_fusion(
        [(1.0, ["v1", "v2", "v3"]), (0.5, ["lex"])], k=60
    )

    assert fused[:3] == ["v1", "v2", "v3"]
    assert fused[3] == "lex"


def test_k_damps_the_tail():
    """A larger k flattens the gap between ranks; a smaller k sharpens it."""
    sharp = lexical.rrf_scores([(1.0, ["a", "b"])], k=1)
    flat = lexical.rrf_scores([(1.0, ["a", "b"])], k=1000)

    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


# ── determinism ───────────────────────────────────────────────────────────


def test_ties_break_on_first_appearance_and_are_stable():
    """Two rows can tie exactly. The winner must not depend on dict order."""
    arms = [(1.0, ["a", "b"]), (1.0, ["b", "a"])]

    first = lexical.reciprocal_rank_fusion(arms, k=60)
    again = lexical.reciprocal_rank_fusion(arms, k=60)

    assert first == ["a", "b"], "the earlier arm's leader breaks the tie"
    assert first == again


def test_a_repeated_id_within_one_arm_keeps_its_best_rank():
    scores = lexical.rrf_scores([(1.0, ["a", "b", "a"])], k=60)

    assert scores["a"] == pytest.approx(1.0 / 61)


def test_no_arms_at_all_fuses_to_nothing():
    assert lexical.reciprocal_rank_fusion([], k=60) == []
    assert lexical.reciprocal_rank_fusion([(1.0, [])], k=60) == []


# ── the shared index / query expression ───────────────────────────────────


def test_document_and_index_expressions_are_the_same_string():
    """The index and the query must agree byte-for-byte or the index is dead
    weight — Postgres matches an expression index by expression equality."""
    assert lexical.fts_document_expression() == (
        "to_tsvector('english', left(content, 100000))"
    )
    assert str(lexical.FTS_CONTENT_LIMIT) in lexical.fts_document_expression()


def test_query_expression_binds_a_parameter_rather_than_interpolating():
    assert lexical.fts_query_expression("q") == "websearch_to_tsquery('english', :q)"


# ── configuration ─────────────────────────────────────────────────────────


def test_defaults(monkeypatch):
    for name in (
        lexical.ENABLED_ENV,
        lexical.WEIGHT_ENV,
        lexical.RRF_K_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    assert lexical.lexical_enabled() is True
    assert lexical.lexical_weight() == 1.0
    assert lexical.rrf_k() == 60


def test_a_weight_of_zero_is_honoured_and_not_replaced_by_the_default(monkeypatch):
    """The easy bug: ``float(os.environ[...]) or DEFAULT`` silently turns the
    operator's kill switch back on, because 0.0 is falsy."""
    monkeypatch.setenv(lexical.WEIGHT_ENV, "0.0")

    assert lexical.lexical_weight() == 0.0


def test_a_weight_of_zero_written_as_plain_zero_is_also_honoured(monkeypatch):
    monkeypatch.setenv(lexical.WEIGHT_ENV, "0")

    assert lexical.lexical_weight() == 0.0


def test_malformed_values_degrade_to_the_defaults(monkeypatch):
    monkeypatch.setenv(lexical.ENABLED_ENV, "treu")
    monkeypatch.setenv(lexical.WEIGHT_ENV, "half")
    monkeypatch.setenv(lexical.RRF_K_ENV, "sixty")

    assert lexical.lexical_enabled() is True
    assert lexical.lexical_weight() == 1.0
    assert lexical.rrf_k() == 60


def test_out_of_range_values_degrade_to_the_defaults(monkeypatch):
    monkeypatch.setenv(lexical.WEIGHT_ENV, "9")
    monkeypatch.setenv(lexical.RRF_K_ENV, "0")

    assert lexical.lexical_weight() == 1.0
    assert lexical.rrf_k() == 60

    monkeypatch.setenv(lexical.WEIGHT_ENV, "-1")
    monkeypatch.setenv(lexical.RRF_K_ENV, "10000")

    assert lexical.lexical_weight() == 1.0
    assert lexical.rrf_k() == 60


def test_the_kill_switch_turns_the_arm_off(monkeypatch):
    monkeypatch.setenv(lexical.ENABLED_ENV, "off")

    assert lexical.lexical_enabled() is False


def test_config_is_read_per_call_not_frozen_at_import(monkeypatch):
    """An operator flipping the switch must not need a redeploy to be believed
    by a process that already imported the module."""
    monkeypatch.setenv(lexical.ENABLED_ENV, "0")
    assert lexical.lexical_enabled() is False

    monkeypatch.setenv(lexical.ENABLED_ENV, "1")
    assert lexical.lexical_enabled() is True


def test_the_default_weight_lets_a_lexical_only_row_survive_truncation():
    """Pins the default against silently regressing into a no-op.

    The arm exists to surface a row the dense arm missed. Whether it can is
    arithmetic, not intent: a lexical-only row scores ``weight/(k + 1)`` and the
    worst dense candidate scores ``1/(k + pool)``. Below roughly 1.0 the
    lexical-only row loses and is truncated away, so the arm can only reorder
    what the dense arm already found.

    Shipped at 0.5 first. Against production, ``pool_pre_ping`` matched 17
    active rows in the caller's own org and not one of them reached the results.
    A unit test is the cheap way to keep that from coming back.
    """
    k = lexical.rrf_k()
    pool = 40  # limit 10 * the candidate-pool multiplier of 4

    lexical_only = lexical.lexical_weight() / (k + 1)
    worst_dense = lexical.VECTOR_WEIGHT / (k + pool)

    assert lexical_only > worst_dense, (
        f"a lexical-only row scores {lexical_only:.5f} against a dense floor of "
        f"{worst_dense:.5f}, so it cannot survive truncation: at this weight the "
        "arm can only reorder the dense pool, never add to it"
    )
