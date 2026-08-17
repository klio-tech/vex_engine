"""How a search string is phrased to the embedding model.

Cheap unit tests over pure functions, and they are here because the failure
they guard against is silent. Text embedded on the wrong side of the retrieval
pair produces a perfectly normal cosine, a perfectly normal ranking, and a
perfectly normal result page. Nothing raises. The only symptom is recall
quality, months later, with nothing to attribute it to.

The single most important assertion in this file is that a PASSAGE is never
prefixed, in either mode. Every one of the 63,196 vectors in production was
written unprefixed; a passage prefix here would invalidate all of them at once,
and would do it without an error, a migration, or a log line.
"""

import pytest

from shared import embedding_prompt as prompt


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(prompt.TEMPLATE_ENV, raising=False)
    monkeypatch.delenv(prompt.TASK_ENV, raising=False)


# ── the invariant ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("template", ["bare", "instruct"])
def test_a_passage_is_never_prefixed(monkeypatch, template):
    """The corpus-invalidating mistake, asserted in both modes.

    This is also the ``-instruct`` model's own convention: unlike plain
    ``e5-*``, the instruct variants take no document prefix. The two reasons
    agree, which is why this is an invariant and not a setting.
    """
    monkeypatch.setenv(prompt.TEMPLATE_ENV, template)
    assert prompt.prefix_for(prompt.ROLE_PASSAGE) == ""
    assert prompt.apply_role("a stored fact", prompt.ROLE_PASSAGE) == "a stored fact"


# ── the default ───────────────────────────────────────────────────────────


def test_the_default_changes_nothing():
    """A 0.008 aggregate margin improvement, better on 8 of 17 queries and
    worse on 4, is not enough evidence to change production's default. The
    mechanism ships; the flip is an observation away."""
    assert prompt.active_template() == prompt.TEMPLATE_BARE
    assert prompt.prefix_for(prompt.ROLE_QUERY) == ""


# ── the instruct template ─────────────────────────────────────────────────


def test_the_instruct_template_wraps_only_the_query(monkeypatch):
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "instruct")
    prefix = prompt.prefix_for(prompt.ROLE_QUERY)
    assert prefix.startswith("Instruct: ")
    assert prefix.endswith("\nQuery: ")
    assert prompt.DEFAULT_TASK in prefix


def test_the_task_sentence_is_configurable(monkeypatch):
    """E5-instruct conditions on the task text, so it is part of the vector.

    Changing it is a ranking change and should be measured like one — which is
    exactly why it is a named knob rather than a string buried in a prefix.
    """
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "instruct")
    monkeypatch.setenv(prompt.TASK_ENV, "Find the deploy runbook")
    assert prompt.prefix_for(prompt.ROLE_QUERY) == (
        "Instruct: Find the deploy runbook\nQuery: "
    )


def test_a_blank_task_falls_back_to_the_default(monkeypatch):
    """Railway renders a declared-but-unset variable as an empty string, and an
    empty task sentence would produce ``Instruct: \\nQuery: `` — a template
    conditioning the model on nothing at all."""
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "instruct")
    monkeypatch.setenv(prompt.TASK_ENV, "   ")
    assert prompt.DEFAULT_TASK in prompt.prefix_for(prompt.ROLE_QUERY)


# ── the env switch ────────────────────────────────────────────────────────


def test_an_unrecognised_template_falls_back_to_bare(monkeypatch):
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "e5-v3")
    assert prompt.active_template() == prompt.TEMPLATE_BARE


def test_the_template_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "INSTRUCT")
    assert prompt.active_template() == prompt.TEMPLATE_INSTRUCT


def test_a_blank_value_is_an_absence_not_an_instruction(monkeypatch):
    monkeypatch.setenv(prompt.TEMPLATE_ENV, "   ")
    assert prompt.active_template() == prompt.TEMPLATE_BARE


def test_an_unknown_role_raises_rather_than_embedding_into_nowhere():
    with pytest.raises(ValueError, match="unknown embedding role"):
        prompt.prefix_for("document")
