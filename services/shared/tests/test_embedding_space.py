"""The two vector spaces: which column is read, which prefix each side gets.

These are cheap unit tests over pure functions, and they are here because the
failure they guard against is silent. An embedding written with the wrong
prefix, or compared against the wrong column, produces numbers that look
completely normal — a cosine in [0, 1], a ranking, a result page. Nothing
raises. The only symptom is recall quality, months later, with no way to
attribute it.
"""

import pytest

from shared import embedding_space as space


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(space.SPACE_ENV, raising=False)
    monkeypatch.delenv(space.DUAL_WRITE_ENV, raising=False)


# ── the default ───────────────────────────────────────────────────────────


def test_the_default_space_is_v1():
    """Importing this module must change nothing until an operator says so.

    The v2 column does not exist until migration 046 runs and holds no vectors
    until the backfill finishes, so anything other than v1 as the default is a
    deploy that breaks recall on arrival.
    """
    assert space.active_space() == space.SPACE_V1
    assert space.column_for() == "embedding"


def test_v1_applies_no_prefix_to_either_role():
    assert space.apply_role("hello", space.ROLE_QUERY) == "hello"
    assert space.apply_role("hello", space.ROLE_PASSAGE) == "hello"


# ── the prefixes ──────────────────────────────────────────────────────────


def test_v2_applies_e5_s_asymmetric_prefixes(monkeypatch):
    monkeypatch.setenv(space.SPACE_ENV, "v2")
    assert space.active_space() == space.SPACE_V2
    assert space.column_for() == "embedding_v2"
    assert space.apply_role("hello", space.ROLE_QUERY) == "query: hello"
    assert space.apply_role("hello", space.ROLE_PASSAGE) == "passage: hello"


def test_the_prefixes_keep_their_trailing_space(monkeypatch):
    """``query:hello`` is a different tokenisation from ``query: hello``.

    E5 was trained on the second one. The space is part of the model's
    contract, not formatting, and dropping it silently forfeits the behaviour
    the whole migration exists to obtain.
    """
    monkeypatch.setenv(space.SPACE_ENV, "v2")
    assert space.prefix_for(space.ROLE_QUERY) == "query: "
    assert space.prefix_for(space.ROLE_PASSAGE) == "passage: "


def test_an_unknown_role_raises_rather_than_embedding_into_nowhere():
    with pytest.raises(ValueError, match="unknown embedding role"):
        space.prefix_for("document")


# ── the env switch ────────────────────────────────────────────────────────


def test_an_unrecognised_space_falls_back_to_v1(monkeypatch):
    """A typo must not reach the SQL.

    The space name selects a COLUMN NAME, which is interpolated into the
    statement because a column cannot be a bind parameter. Falling back is
    therefore not merely friendly, it is the boundary.
    """
    monkeypatch.setenv(space.SPACE_ENV, "v3")
    assert space.active_space() == space.SPACE_V1


def test_the_space_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(space.SPACE_ENV, "V2")
    assert space.active_space() == space.SPACE_V2


def test_a_blank_value_is_an_absence_not_an_instruction(monkeypatch):
    """Railway and docker-compose render a declared-but-unset variable as ""."""
    monkeypatch.setenv(space.SPACE_ENV, "   ")
    assert space.active_space() == space.SPACE_V1


# ── dual write ────────────────────────────────────────────────────────────


def test_dual_write_is_on_by_default_and_puts_the_active_space_first():
    """Order is the contract: the caller treats the first as required.

    A row missing its ACTIVE vector is written and invisible. A row missing
    the inactive one is fully readable and merely un-prepared for a switch
    that has not happened.
    """
    assert space.write_spaces() == (space.SPACE_V1, space.SPACE_V2)


def test_switching_the_read_space_reorders_the_write_spaces(monkeypatch):
    monkeypatch.setenv(space.SPACE_ENV, "v2")
    assert space.write_spaces() == (space.SPACE_V2, space.SPACE_V1)


def test_disabling_dual_write_leaves_only_the_active_space(monkeypatch):
    monkeypatch.setenv(space.DUAL_WRITE_ENV, "0")
    assert space.write_spaces() == (space.SPACE_V1,)


def test_disabling_dual_write_under_v2_still_writes_v2(monkeypatch):
    """The one thing that must never happen: writing neither, or the wrong one.

    Turning dual-write off is a post-migration state, and it must narrow the
    write to the space being READ — never to the one being retired.
    """
    monkeypatch.setenv(space.SPACE_ENV, "v2")
    monkeypatch.setenv(space.DUAL_WRITE_ENV, "0")
    assert space.write_spaces() == (space.SPACE_V2,)
