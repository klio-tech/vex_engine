"""``extract_graph`` must not renumber the facts its own edges point at.

The extraction prompt asks for one JSON object holding facts, entities, and
edges, where each edge names its fact **by index**::

    {"facts": ["...", "..."],
     "edges": [{"src": 0, "about": "auth service"}]}

``src`` is a position in ``facts``. So the list's positions are a wire
protocol, not an implementation detail — and the parser was compacting it::

    facts = [f for f in facts_raw if isinstance(f, str)]

Drop one malformed element and every fact after it slides down a slot while
the edges keep pointing at the old numbers. The edge that meant fact 2 now
means fact 1. That is not a dropped edge, which the pipeline handles safely
everywhere else; it is an edge drawn to the wrong memory — the graph asserting
a relationship nobody stated, stamped INFERRED and indistinguishable from a
correct one. ``entity_brief`` then serves that memory as evidence for an entity
it was never about.

The parser must therefore preserve positions. An invalid element is REPLACED
with an empty string rather than removed: the fact loop already skips falsy
facts (``if not fact: continue``), so nothing is stored for it, its index never
enters ``fact_row_ids``, and any edge naming it is skipped by the existing
missing-src branch. Dropping the edge is the correct outcome; moving it is not.

An adjacent note on why this is reachable rather than theoretical: the same
parser deliberately tolerates malformed entities and edges one at a time, on
the stated principle that a bad item never takes the batch down. A model that
emits ``null`` for one fact in a list of five is exactly the case that
principle exists for, and it is the case that silently corrupted the graph.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.curator import extract_graph


def _llm(payload):
    return patch("app.curator.call_llm", AsyncMock(return_value=payload))


@pytest.mark.asyncio
async def test_a_malformed_fact_does_not_renumber_the_ones_after_it():
    payload = {
        "facts": ["the auth service moved to Rust", None, "billing is unchanged"],
        "entities": [{"name": "billing", "kind": "system"}],
        # The model is pointing at "billing is unchanged", which is at index 2.
        "edges": [{"src": 2, "relates_to": "billing"}],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert graph["facts"][2] == "billing is unchanged", (
        "index 2 must still be the fact the model called index 2; filtering "
        "the None slid it to index 1, so the edge below now attaches to a "
        "fact about the auth service instead"
    )
    assert graph["facts"][0] == "the auth service moved to Rust"
    assert graph["edges"] == [{"src": 2, "relation": "relates_to", "entity": "billing"}]


@pytest.mark.asyncio
async def test_an_invalid_fact_is_blanked_so_nothing_is_stored_for_it():
    """Position preserved, content not invented.

    The replacement has to be falsy: ``_synthesize_and_store`` skips falsy
    facts, so the slot holds the numbering without producing a memory, and no
    edge can attach to it.
    """
    payload = {
        "facts": ["a real fact", 42, {"nested": "junk"}, ["a", "list"]],
        "entities": [],
        "edges": [],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert len(graph["facts"]) == 4, "positions preserved"
    assert graph["facts"][0] == "a real fact"
    assert graph["facts"][1] == ""
    assert graph["facts"][2] == ""
    assert graph["facts"][3] == ""
    assert all(isinstance(f, str) for f in graph["facts"]), (
        "the non-string elements must not survive into the pipeline; only "
        "their positions do"
    )


@pytest.mark.asyncio
async def test_an_edge_pointing_at_a_blanked_fact_is_simply_dropped():
    """The safe failure, preserved: no row to attach to means no edge.

    Positive control on the fix's *other* half. Preserving indices must not
    resurrect an edge whose fact never became a memory — that would attach it
    to nothing, or worse, to whatever the index later resolves to.
    """
    payload = {
        "facts": [None, "a real fact"],
        "entities": [{"name": "billing", "kind": "system"}],
        "edges": [
            {"src": 0, "about": "billing"},  # points at the blank
            {"src": 1, "about": "billing"},  # points at the real fact
        ],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    # Both edges parse — the parser's job is not to know which facts stored.
    assert len(graph["edges"]) == 2
    # But only index 1 can ever enter fact_row_ids, so only that edge is drawn.
    assert graph["facts"][0] == ""
    assert graph["facts"][1] == "a real fact"


@pytest.mark.asyncio
async def test_a_well_formed_extraction_is_unchanged():
    """Positive control: the ordinary path still parses exactly as before."""
    payload = {
        "facts": ["one", "two"],
        "entities": [{"name": "billing", "kind": "system"}],
        "edges": [{"src": 1, "about": "billing"}],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert graph["facts"] == ["one", "two"]
    assert graph["entities"] == [{"name": "billing", "kind": "system"}]
    assert graph["edges"] == [{"src": 1, "relation": "about", "entity": "billing"}]


@pytest.mark.asyncio
async def test_a_non_list_facts_value_is_still_the_empty_list():
    """Unchanged behaviour: a wholly malformed payload yields nothing."""
    with _llm({"facts": "not a list", "entities": [], "edges": []}):
        graph = await extract_graph("notes")
    assert graph["facts"] == []
