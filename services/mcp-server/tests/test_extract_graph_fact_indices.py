"""``extract_graph`` must not renumber the items its own edges point at.

The extraction prompt asks for one JSON object holding memories, entities, and
edges, where each edge names its memory **by index**::

    {"memories": [{"kind": "fact", "content": "..."}, ...],
     "edges": [{"src": 0, "about": "auth service"}]}

``src`` is a position in ``memories``. So the list's positions are a wire
protocol, not an implementation detail — and the parser was compacting it::

    facts = [f for f in facts_raw if isinstance(f, str)]

Drop one malformed element and every item after it slides down a slot while
the edges keep pointing at the old numbers. The edge that meant item 2 now
means item 1. That is not a dropped edge, which the pipeline handles safely
everywhere else; it is an edge drawn to the wrong memory — the graph asserting
a relationship nobody stated, stamped INFERRED and indistinguishable from a
correct one. ``entity_brief`` then serves that memory as evidence for an entity
it was never about.

The parser must therefore preserve positions. An invalid element is REPLACED
with ``None`` rather than removed: the item loop already skips ``None``, so
nothing is stored for it, its index never enters ``fact_row_ids``, and any edge
naming it is skipped by the existing missing-src branch. Dropping the edge is
the correct outcome; moving it is not.

Since the taxonomy change the same replacement covers a second, more common
case: an item the durability rule rejects (``shared.taxonomy.is_transient``).
Those are dropped for content reasons rather than shape reasons, but they must
hold their slot for exactly the same protocol reason.

An adjacent note on why this is reachable rather than theoretical: the same
parser deliberately tolerates malformed entities and edges one at a time, on
the stated principle that a bad item never takes the batch down. A model that
emits ``null`` for one item in a list of five is exactly the case that
principle exists for, and it is the case that silently corrupted the graph.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.curator import extract_graph
from shared.taxonomy import ExtractedMemory

AUTH = "the auth service moved to Rust for the token path"
BILLING = "billing behaviour is unchanged across the migration"


def _item(kind, content):
    return {"kind": kind, "content": content}


def _llm(payload):
    return patch("app.curator.call_llm", AsyncMock(return_value=payload))


@pytest.mark.asyncio
async def test_a_malformed_item_does_not_renumber_the_ones_after_it():
    payload = {
        "memories": [_item("fact", AUTH), None, _item("fact", BILLING)],
        "entities": [{"name": "billing", "kind": "system"}],
        # The model is pointing at the billing item, which is at index 2.
        "edges": [{"src": 2, "relates_to": "billing"}],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert graph["memories"][2] == ExtractedMemory("fact", BILLING), (
        "index 2 must still be the item the model called index 2; filtering "
        "the None slid it to index 1, so the edge below now attaches to a "
        "memory about the auth service instead"
    )
    assert graph["memories"][0] == ExtractedMemory("fact", AUTH)
    assert graph["edges"] == [{"src": 2, "relation": "relates_to", "entity": "billing"}]


@pytest.mark.asyncio
async def test_an_invalid_item_is_nulled_so_nothing_is_stored_for_it():
    """Position preserved, content not invented.

    The replacement has to be ``None``: ``_synthesize_and_store`` skips it, so
    the slot holds the numbering without producing a memory, and no edge can
    attach to it.
    """
    payload = {
        "memories": [_item("fact", AUTH), 42, {"nested": "junk"}, ["a", "list"]],
        "entities": [],
        "edges": [],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert len(graph["memories"]) == 4, "positions preserved"
    assert graph["memories"][0] == ExtractedMemory("fact", AUTH)
    assert graph["memories"][1] is None
    assert graph["memories"][2] is None
    assert graph["memories"][3] is None


@pytest.mark.asyncio
async def test_a_transient_item_holds_its_slot_rather_than_compacting():
    """The durability rule drops content, and must not shift the numbering.

    This is the new common case: the model answers with a well-formed item
    that is session exhaust, ``shared.taxonomy.is_transient`` rejects it, and
    an edge later in the list still names its own index.
    """
    payload = {
        "memories": [
            _item("observation", "Used tool Bash input: {'command': 'ls -la'}"),
            _item("fact", BILLING),
        ],
        "entities": [{"name": "billing", "kind": "system"}],
        "edges": [{"src": 1, "about": "billing"}],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert graph["memories"][0] is None, "tool-log narration is not durable"
    assert graph["memories"][1] == ExtractedMemory("fact", BILLING)
    assert graph["edges"] == [{"src": 1, "relation": "about", "entity": "billing"}]


@pytest.mark.asyncio
async def test_an_edge_pointing_at_a_nulled_item_is_simply_dropped():
    """The safe failure, preserved: no row to attach to means no edge.

    Positive control on the fix's *other* half. Preserving indices must not
    resurrect an edge whose item never became a memory — that would attach it
    to nothing, or worse, to whatever the index later resolves to.
    """
    payload = {
        "memories": [None, _item("fact", BILLING)],
        "entities": [{"name": "billing", "kind": "system"}],
        "edges": [
            {"src": 0, "about": "billing"},  # points at the blank
            {"src": 1, "about": "billing"},  # points at the real item
        ],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    # Both edges parse — the parser's job is not to know which items stored.
    assert len(graph["edges"]) == 2
    # But only index 1 can ever enter fact_row_ids, so only that edge is drawn.
    assert graph["memories"][0] is None
    assert graph["memories"][1] == ExtractedMemory("fact", BILLING)


@pytest.mark.asyncio
async def test_a_well_formed_extraction_is_unchanged():
    """Positive control: the ordinary path still parses exactly as before."""
    payload = {
        "memories": [_item("decision", AUTH), _item("fact", BILLING)],
        "entities": [{"name": "billing", "kind": "system"}],
        "edges": [{"src": 1, "about": "billing"}],
    }

    with _llm(payload):
        graph = await extract_graph("notes")

    assert graph["memories"] == [
        ExtractedMemory("decision", AUTH),
        ExtractedMemory("fact", BILLING),
    ]
    assert graph["entities"] == [{"name": "billing", "kind": "system"}]
    assert graph["edges"] == [{"src": 1, "relation": "about", "entity": "billing"}]


@pytest.mark.asyncio
async def test_a_non_list_memories_value_is_still_the_empty_list():
    """Unchanged behaviour: a wholly malformed payload yields nothing."""
    with _llm({"memories": "not a list", "entities": [], "edges": []}):
        graph = await extract_graph("notes")
    assert graph["memories"] == []
