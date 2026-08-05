"""The curator's two env knobs must not be able to take a tick down or empty it.

``CURATOR_ENTITY_CAP`` and ``CURATOR_RECONCILE_MAX`` were read with a bare
``int(os.environ.get(...))``. Two distinct failures follow from that, and they
fail in opposite directions:

* **A malformed value raises.** ``int("12 ")`` is fine, but ``int("twelve")``
  and ``int("")`` are ``ValueError``. ``_entity_cap()`` is called from
  ``curator_tick`` and from ``_synthesize_and_store``; the raise lands inside
  ``run_curator_once``'s per-pair ``except Exception``, which logs a warning
  and moves on to the next user. So a single typo in one environment variable
  silently disables curation for **every** user, and the only evidence is a
  warning that names the tick, not the variable.

* **A negative value is accepted in silence.** ``CURATOR_ENTITY_CAP=-1`` makes
  ``entity_budget.remaining <= 0`` true on the first entity of every tick, so
  every entity is dropped and — because ``_store_graph`` skips an edge whose
  entity is missing — every edge with it. The graph simply stops growing, with
  no error anywhere. ``CURATOR_RECONCILE_MAX=-1`` likewise disables
  reconciliation entirely.

``shared.env_config.parse_int`` is the guarded parser the rest of the engine
now uses: absent, unparseable, or out of range all fall back to the documented
default, with a WARNING naming the variable and the value. These tests pin that
both knobs go through it, including the ``minimum=1`` bound — without the bound
a negative parses cleanly and the silent-drop failure survives the fix.
"""

import pytest

from app.curator import _entity_cap, _reconcile_budget


# ---------------------------------------------------------------------------
# Positive controls: a well-formed value is still honoured.
#
# Every assertion below is that some input yields the DEFAULT. A parser
# hardcoded to return the default would pass all of them, so the controls come
# first: they prove the fixture reads the environment at all.
# ---------------------------------------------------------------------------


def test_a_valid_entity_cap_is_honoured(monkeypatch):
    monkeypatch.setenv("CURATOR_ENTITY_CAP", "25")
    assert _entity_cap() == 25


def test_a_valid_reconcile_budget_is_honoured(monkeypatch):
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", "7")
    assert _reconcile_budget() == 7


def test_the_documented_defaults_apply_when_unset(monkeypatch):
    monkeypatch.delenv("CURATOR_ENTITY_CAP", raising=False)
    monkeypatch.delenv("CURATOR_RECONCILE_MAX", raising=False)
    assert _entity_cap() == 12
    assert _reconcile_budget() == 50


# ---------------------------------------------------------------------------
# Malformed: must degrade to the default, never raise into the tick.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["twelve", "", "   ", "12.5", "12,"])
def test_a_malformed_entity_cap_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("CURATOR_ENTITY_CAP", bad)
    assert _entity_cap() == 12


@pytest.mark.parametrize("bad", ["fifty", "", "   ", "50s"])
def test_a_malformed_reconcile_budget_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", bad)
    assert _reconcile_budget() == 50


# ---------------------------------------------------------------------------
# Out of range: a cap below 1 is not a cap, it is an off switch nobody asked
# for. Zero and negatives take the default too.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "-100"])
def test_a_non_positive_entity_cap_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("CURATOR_ENTITY_CAP", bad)
    assert _entity_cap() == 12


@pytest.mark.parametrize("bad", ["0", "-1", "-100"])
def test_a_non_positive_reconcile_budget_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", bad)
    assert _reconcile_budget() == 50
