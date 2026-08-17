"""The taxonomy and the durability line.

Two properties are under test and they pull in opposite directions, which is
the whole difficulty:

* the durability rule must actually reject the shapes that filled the corpus
  (2,675 of 5,253 rows on the reference org were literal ``Used tool Bash
  input: {...}`` logs), and
* it must not reject real knowledge. Every string in
  :data:`DURABLE_CORPUS_SAMPLES` is a verbatim memory from production that a
  reader would want back. A regex that starts eating those has failed, however
  many tool logs it catches — better-classified junk is still junk, but
  deleted knowledge is worse than either.
"""

import pytest

from shared.taxonomy import (
    DEFAULT_MEMORY_TYPE,
    MEMORY_TAXONOMY,
    ExtractedMemory,
    coerce_memory_type,
    is_transient,
    parse_typed_items,
)

# Verbatim from production. These MUST survive the filter.
DURABLE_CORPUS_SAMPLES = (
    "Klio production deployments are tracked under the Klio environment in the "
    "vex-hq/Vex-Dashboard repository",
    "Claude Code reaches shared memory through hooks, making proxy wiring "
    "redundant for it.",
    "cacheKeyFor produces different keys for the same query under different projects.",
    "The extension handoff module exposes EXTENSION_READY_MESSAGE_TYPE = "
    "'EXTENSION_READY'.",
    "project scope is effectively dead: 12 project-scope memories in 30d vs "
    "15,865 private and 14,533 org.",
    "GitGuardian scans each commit individually against the config present in "
    "that commit's tree.",
    "The pooled Postgres endpoint runs PgBouncer in transaction mode and "
    "refuses libpq startup parameters.",
    "Deploys go through GitHub source builds rather than Docker Hub images.",
    "The health endpoint is /healthz, not /health.",
)

# Verbatim (or verbatim in shape) from production. These MUST be rejected.
TRANSIENT_CORPUS_SAMPLES = (
    'Used tool Bash input: {"command":"cd /repo && ruff check services/"}',
    'Used tool Read input: {"file_path":"/repo/services/shared/tests/x.py"}',
    "The agent responded with 'pong' to confirm the connection.",
    "The agent then read two source files and reported the domain.",
    "A Supabase container is running on port 54422.",
    "The dev server is listening on port 3000.",
    "No crash occurred during the session and multiple TTS sessions started.",
    "The branch adds 40 new tests, all green.",
    "All tests pass after the change.",
    "fix/home-connection-signals branch was created with HEAD at "
    "17128dbe439d7e45ade739189e8b29794a581fa7",
    "The command failed and returned exit code 1.",
    "No file named '*location-step*' exists under any onboarding directory.",
    "The API is reachable at http://localhost:8787/healthz right now.",
    "ok",
)


@pytest.mark.parametrize("content", DURABLE_CORPUS_SAMPLES)
def test_real_knowledge_survives_the_filter(content):
    assert not is_transient(content), (
        "the deterministic backstop must stay narrow; this is a real memory "
        "a reader would want back"
    )


@pytest.mark.parametrize("content", TRANSIENT_CORPUS_SAMPLES)
def test_session_exhaust_is_rejected(content):
    assert is_transient(content)


def test_empty_and_missing_content_are_transient():
    assert is_transient(None)
    assert is_transient("")
    assert is_transient("   ")


def test_every_taxonomy_label_round_trips():
    for kind in MEMORY_TAXONOMY:
        assert coerce_memory_type(kind) == kind


def test_unknown_and_malformed_kinds_fall_back_rather_than_raise():
    assert coerce_memory_type("wisdom") == DEFAULT_MEMORY_TYPE
    assert coerce_memory_type(None) == DEFAULT_MEMORY_TYPE
    assert coerce_memory_type(17) == DEFAULT_MEMORY_TYPE
    assert coerce_memory_type("") == DEFAULT_MEMORY_TYPE


def test_the_workspace_section_names_are_accepted_as_aliases():
    """A model that echoes the prompt's section labels must not be punished.

    'Constraints' is the UI's name for ``fact``; mapping it to ``observation``
    would put a real constraint in the Recent bucket, which is the exact
    failure this taxonomy exists to end.
    """
    assert coerce_memory_type("Constraints") == "fact"
    assert coerce_memory_type("Decisions") == "decision"
    assert coerce_memory_type("plans") == "plan"


def test_parse_keeps_the_kind_the_model_chose():
    items = parse_typed_items(
        [
            {"kind": "decision", "content": DURABLE_CORPUS_SAMPLES[7]},
            {"kind": "fact", "content": DURABLE_CORPUS_SAMPLES[6]},
        ]
    )
    assert [i.memory_type for i in items] == ["decision", "fact"]


def test_parse_drops_transient_items_without_dropping_the_batch():
    items = parse_typed_items(
        [
            {"kind": "observation", "content": TRANSIENT_CORPUS_SAMPLES[0]},
            {"kind": "fact", "content": DURABLE_CORPUS_SAMPLES[0]},
            {"kind": "observation", "content": TRANSIENT_CORPUS_SAMPLES[3]},
        ]
    )
    assert items == [ExtractedMemory("fact", DURABLE_CORPUS_SAMPLES[0])]


def test_parse_tolerates_malformed_entries_one_at_a_time():
    items = parse_typed_items(
        [
            42,
            {"no_content": True},
            {"kind": "fact", "content": None},
            {"kind": "fact", "content": DURABLE_CORPUS_SAMPLES[1]},
        ]
    )
    assert items == [ExtractedMemory("fact", DURABLE_CORPUS_SAMPLES[1])]


def test_parse_accepts_a_bare_string_as_an_untyped_item():
    """Backwards tolerance for a model that answers with the old shape."""
    items = parse_typed_items([DURABLE_CORPUS_SAMPLES[2]])
    assert items == [ExtractedMemory(DEFAULT_MEMORY_TYPE, DURABLE_CORPUS_SAMPLES[2])]


def test_parse_of_a_non_list_is_the_empty_list():
    assert parse_typed_items("not a list") == []
    assert parse_typed_items(None) == []
