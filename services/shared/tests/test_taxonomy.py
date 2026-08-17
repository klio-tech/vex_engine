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


# ---------------------------------------------------------------------------
# Regression: two ways the backstop ate real knowledge
# ---------------------------------------------------------------------------
#
# Independent review found both by sampling the DROPPED set of a real
# extraction run: roughly 1 in 6 dropped items was durable knowledge. The
# backstop was the cause in two specific, reproducible ways, and neither had a
# test — the original corpus above happened to contain no durable port
# constraint and nothing under 25 characters.


# Durable constraints that happen to mention a port. A port number is not by
# itself machine state: "listens on 5432 by default" is a property of Postgres,
# not of this laptop right now. The old pattern matched on the noun and on a
# too-wide verb alternation (`served|exposed|bound`), so which of these lived
# and which died came down to conjugation luck.
DURABLE_PORT_SAMPLES = (
    "Postgres listens on port 5432 by default.",
    "The Klio proxy is served on port 8787 and must not be reassigned.",
    "The dashboard dev server is bound to port 3000 in every environment.",
    "Supabase exposes its API gateway on port 54321 by convention.",
    "Grafana is exposed on port 3000, which collides with the Next.js dev server.",
)

# Durable facts that are simply short. The extraction prompt explicitly asks
# for "one short, self-contained statement", so a blanket 25-character floor
# selects against exactly what the prompt requests.
SHORT_DURABLE_SAMPLES = (
    "Health path is /healthz",
    "GitHub org: klio-tech",
    "Prod DB is Neon.",
    "Python 3.14 is required",
    "Use pnpm, not npm.",
    "Node 22 is the floor.",
    "Auth uses PKCE.",
    "Deploy target: Railway",
    "Brand font is Inter.",
    "Region is us-east-1.",
)


@pytest.mark.parametrize("content", DURABLE_PORT_SAMPLES)
def test_a_durable_port_constraint_is_not_transient(content):
    """A port constraint is a property of the system, not of this moment.

    Klio's own stored memory includes one (the proxy owns 8787), so a filter
    that eats these eats the product's own operating knowledge.
    """
    assert not is_transient(content)


@pytest.mark.parametrize("content", SHORT_DURABLE_SAMPLES)
def test_a_short_durable_fact_survives_the_floor(content):
    """Brevity is not transience.

    The floor exists for contentless tokens ('ok', 'done', 'pong'), not for
    terse facts. Anything carrying a number, a path, a URL, or an identifier
    is carrying content however short it is.
    """
    assert not is_transient(content)


@pytest.mark.parametrize(
    "content",
    ("ok", "done", "pong", "yes", "it works", "sure thing", "all good now"),
)
def test_contentless_short_strings_are_still_transient(content):
    """The floor still has to do its original job.

    Positive control on the fix: relaxing the floor must not make every
    two-word acknowledgement a memory.
    """
    assert is_transient(content)


# Root causes and negative decisions. These were dropped by the PROMPT rather
# than the regex — a root cause is phrased in past tense about an incident so
# it reads as narration, and TAXONOMY_RULE defined `decision` only
# affirmatively ("resolves alternatives"), which excludes deciding NOT to do
# something. The rule text is the fix, so the test pins the rule text.
def test_the_durability_rule_keeps_root_causes_and_negative_decisions():
    from shared.taxonomy import DURABILITY_RULE

    assert "root cause" in DURABILITY_RULE.lower(), (
        "a root cause found once is durable; the rule must say so, or the "
        "model reads it as incident narration and drops it"
    )
    assert "not to do" in DURABILITY_RULE.lower(), (
        "a decision to exclude something from scope is a decision"
    )
