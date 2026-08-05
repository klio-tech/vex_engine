"""The per-tick entity cap counts entities CREATED, not entities mentioned.

``CURATOR_ENTITY_CAP`` (default 12) is a god-entity guard: no single runaway
extraction may hairball an org's graph in one tick. The guard was charging the
budget for every ``add_entity`` call, and ``add_entity`` is an upsert — so a
name the org already knows, resolved for the tenth time, spent a tenth unit of
an allowance meant for new rows.

The arithmetic is not marginal. A 50-row batch partitions into roughly four
``(space_id, project_id)`` groups, each extraction yields around five entities,
and in a mature graph most of them already exist. That is ~20 references
against a budget of 12. The budget empties, ``_store_graph`` hits its ``break``,
and everything after is dropped — not only the remaining entities but every
edge pointing at one, because an edge whose entity is missing is skipped
silently. The cursor has already advanced past the rows those edges came from.

So the failure is: an org's graph stops growing partway through a tick, roughly
40% of the tick's edges vanish, and nothing in the logs says more than "cap
reached". The fix is one word of SQL — ``(xmax = 0) AS inserted`` — and one
condition at the call site.

Mock-based on the DB seam, real assertions on the budget: what is under test is
the accounting, and ``graph_store.add_entity``'s own SQL is covered against
real Postgres in ``shared/tests/test_graph_store.py``.
"""

from unittest.mock import patch

from shared.graph_store import EntityUpsert

from app.curator import EntityBudget, _store_graph


def _entities(*names):
    return [{"name": n, "kind": "system"} for n in names]


def test_a_repeated_upsert_hit_costs_no_budget():
    budget = EntityBudget(12)
    # Every name resolves to an entity that already existed.
    with patch(
        "app.curator.graph_store.add_entity",
        side_effect=lambda db, **kw: EntityUpsert(f"id-{kw['name']}", False),
    ):
        _store_graph(
            None,
            org_id="org-1",
            entities=_entities(*[f"svc-{i}" for i in range(20)]),
            edges=[],
            fact_row_ids={},
            entity_budget=budget,
        )

    assert budget.remaining == 12, (
        f"20 upsert HITS consumed {12 - budget.remaining} of a budget meant "
        "for new entities; the cap is on creation, not on mention"
    )


def test_an_insert_does_cost_budget():
    """Positive control: the budget is not simply never decremented."""
    budget = EntityBudget(12)
    with patch(
        "app.curator.graph_store.add_entity",
        side_effect=lambda db, **kw: EntityUpsert(f"id-{kw['name']}", True),
    ):
        _store_graph(
            None,
            org_id="org-1",
            entities=_entities("a-svc", "b-svc", "c-svc"),
            edges=[],
            fact_row_ids={},
            entity_budget=budget,
        )

    assert budget.remaining == 9


def test_hits_do_not_exhaust_the_cap_and_strand_later_entities():
    """The consequence, end to end: entity 13+ still resolves, so its edge lands.

    The budget assertion above says the counter is right. This one says the
    counter being right is what keeps the graph intact — the failure was never
    really about a number, it was about the ``break`` the number triggers and
    the edges silently dropped after it.
    """
    budget = EntityBudget(12)
    names = [f"svc-{i}" for i in range(20)]
    # A realistic mix: the first 18 already exist, the last two are new.
    existing = set(names[:18])

    calls: list[str] = []

    def upsert(db, **kw):
        calls.append(kw["name"])
        return EntityUpsert(f"id-{kw['name']}", kw["name"] not in existing)

    edges = [{"src": 0, "relation": "about", "entity": n} for n in names]
    with patch("app.curator.graph_store.add_entity", side_effect=upsert):
        with patch("app.curator.graph_store.add_edge", return_value="edge-1"):
            drawn = _store_graph(
                None,
                org_id="org-1",
                entities=_entities(*names),
                edges=edges,
                fact_row_ids={0: "mem-1"},
                entity_budget=budget,
            )

    assert calls == names, (
        f"the loop broke after {len(calls)} of 20 entities; every edge naming "
        "one of the rest is then dropped with only a 'cap reached' warning"
    )
    assert drawn == 20
    assert budget.remaining == 10, "only the two genuinely new entities charged"


def test_the_cap_still_stops_a_runaway_extraction():
    """Positive control for the guard itself: creation IS still capped.

    Charging only inserts must not turn the god-entity cap off. Twenty NEW
    entities against a budget of 3 must still stop at 3.
    """
    budget = EntityBudget(3)
    calls: list[str] = []

    def upsert(db, **kw):
        calls.append(kw["name"])
        return EntityUpsert(f"id-{kw['name']}", True)

    with patch("app.curator.graph_store.add_entity", side_effect=upsert):
        _store_graph(
            None,
            org_id="org-1",
            entities=_entities(*[f"svc-{i}" for i in range(20)]),
            edges=[],
            fact_row_ids={},
            entity_budget=budget,
        )

    assert len(calls) == 3
    assert budget.remaining == 0
