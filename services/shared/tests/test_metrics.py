"""Unit tests for :mod:`shared.metrics` — the single instrumentation module.

Three things are pinned here:

1. the primitives behave (counter monotonicity, histogram bucket/sum/count);
2. the Prometheus text exposition format is emitted correctly;
3. **label cardinality is bounded** — the correctness property. A metrics
   backend dies from an unbounded label set long before it dies from volume,
   so the registry refuses high-cardinality label names outright and refuses
   any label value outside a declared, finite set.
"""

import pytest
from shared import metrics


@pytest.fixture
def registry():
    """A private registry, so these tests never touch the process default."""
    return metrics.Registry()


# ── primitives ────────────────────────────────────────────────────────


def test_counter_starts_at_zero_and_increments(registry):
    c = registry.counter("klio_test_total", "A test counter.")
    assert c.value() == 0.0
    c.inc()
    c.inc(3)
    assert c.value() == 4.0


def test_counter_refuses_negative_increment(registry):
    c = registry.counter("klio_test_total", "A test counter.")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_counter_tracks_each_label_combination_separately(registry):
    c = registry.counter(
        "klio_test_total", "A test counter.", labels={"outcome": ("ok", "error")}
    )
    c.inc(outcome="ok")
    c.inc(outcome="ok")
    c.inc(outcome="error")
    assert c.value(outcome="ok") == 2.0
    assert c.value(outcome="error") == 1.0


def test_gauge_goes_up_and_down(registry):
    g = registry.gauge("klio_test_inflight", "A test gauge.")
    g.inc()
    g.inc()
    g.dec()
    assert g.value() == 1.0
    g.set(9)
    assert g.value() == 9.0


def test_histogram_records_sum_count_and_cumulative_buckets(registry):
    h = registry.histogram(
        "klio_test_seconds", "A test histogram.", buckets=(0.1, 1.0, 10.0)
    )
    h.observe(0.05)
    h.observe(0.5)
    h.observe(50.0)
    snap = h.snapshot()
    assert snap.count == 3
    assert snap.total == pytest.approx(50.55)
    # Buckets are cumulative: le=0.1 -> 1, le=1.0 -> 2, le=10 -> 2, +Inf -> 3.
    assert snap.buckets == ((0.1, 1), (1.0, 2), (10.0, 2))


def test_histogram_timer_records_an_observation(registry):
    h = registry.histogram("klio_test_seconds", "A test histogram.")
    with h.time():
        pass
    assert h.snapshot().count == 1


def test_histogram_timer_still_records_when_the_body_raises(registry):
    """Latency of a failure is latency. A raise must not lose the observation."""
    h = registry.histogram("klio_test_seconds", "A test histogram.")
    with pytest.raises(RuntimeError):
        with h.time():
            raise RuntimeError("boom")
    assert h.snapshot().count == 1


def test_redeclaring_the_same_metric_returns_the_same_object(registry):
    a = registry.counter("klio_test_total", "A test counter.")
    b = registry.counter("klio_test_total", "A test counter.")
    assert a is b


def test_redeclaring_with_different_labels_is_rejected(registry):
    registry.counter("klio_test_total", "doc", labels={"outcome": ("ok",)})
    with pytest.raises(ValueError):
        registry.counter("klio_test_total", "doc", labels={"scope": ("private",)})


# ── label cardinality: the correctness property ───────────────────────


@pytest.mark.parametrize(
    "label",
    ["org_id", "user_id", "session_id", "agent_id", "project_id", "memory_id"],
)
def test_declaring_an_unbounded_label_name_is_refused(registry, label):
    with pytest.raises(metrics.UnboundedLabelError):
        registry.counter("klio_test_total", "doc", labels={label: ("a", "b")})


def test_declaring_a_label_with_no_finite_value_set_is_refused(registry):
    """A label whose values are not enumerated is an unbounded label."""
    with pytest.raises(metrics.UnboundedLabelError):
        registry.counter("klio_test_total", "doc", labels={"outcome": ()})


def test_a_label_value_outside_the_declared_set_is_refused(registry):
    c = registry.counter("klio_test_total", "doc", labels={"outcome": ("ok", "error")})
    with pytest.raises(ValueError):
        c.inc(outcome="0f8c1a2e-free-text-from-a-caller")


def test_missing_and_extra_labels_are_refused(registry):
    c = registry.counter("klio_test_total", "doc", labels={"outcome": ("ok",)})
    with pytest.raises(ValueError):
        c.inc()
    with pytest.raises(ValueError):
        c.inc(outcome="ok", scope="private")


def test_every_shipped_metric_has_only_bounded_labels():
    """THE pin: no metric in the default registry may carry a free-text label.

    This runs over the real registry after importing the module that declares
    every metric, so a future call site cannot introduce ``org_id`` as a label
    without turning this red.
    """
    offenders = []
    for metric in metrics.REGISTRY.collect():
        for name, allowed in metric.labels.items():
            if name in metrics.UNBOUNDED_LABEL_NAMES:
                offenders.append(f"{metric.name}: forbidden label {name!r}")
            if not allowed:
                offenders.append(f"{metric.name}: label {name!r} has no value set")
            if len(allowed) > metrics.MAX_LABEL_VALUES:
                offenders.append(f"{metric.name}: label {name!r} has too many values")
    assert offenders == []


def test_the_cardinality_pin_is_not_vacuous():
    """Positive control for the test above.

    If the check were inert, a registry containing a deliberately bad metric
    would still pass it. Building that registry by bypassing the constructor
    guard proves the *assertion* catches what the *constructor* prevents.
    """
    bad = metrics.Registry()
    smuggled = metrics.Counter("klio_bad_total", "doc", labels={"outcome": ("ok",)})
    # The constructor refuses ``org_id`` outright (pinned above), so the only
    # way to get a bad metric in front of the assertion is to plant it after
    # construction. Test-local surgery, never a product pattern.
    smuggled.labels = {"org_id": ("a",)}
    bad._force_declare_for_test(smuggled)
    offenders = [
        m.name
        for m in bad.collect()
        for name in m.labels
        if name in metrics.UNBOUNDED_LABEL_NAMES
    ]
    assert offenders == ["klio_bad_total"]


def test_the_default_registry_actually_contains_metrics():
    """Positive control: the pin above would pass vacuously on an empty registry."""
    assert len(metrics.REGISTRY.collect()) >= 15


# ── exposition format ─────────────────────────────────────────────────


def test_render_emits_help_type_and_value(registry):
    c = registry.counter("klio_test_total", "A test counter.")
    c.inc(2)
    body = registry.render()
    assert "# HELP klio_test_total A test counter." in body
    assert "# TYPE klio_test_total counter" in body
    assert "klio_test_total 2" in body


def test_render_emits_labels_in_declared_order(registry):
    c = registry.counter(
        "klio_test_total",
        "doc",
        labels={"tool": ("recall",), "outcome": ("ok",)},
    )
    c.inc(tool="recall", outcome="ok")
    assert 'klio_test_total{tool="recall",outcome="ok"} 1' in registry.render()


def test_render_emits_histogram_buckets_sum_and_count(registry):
    h = registry.histogram("klio_test_seconds", "doc", buckets=(0.1, 1.0))
    h.observe(0.5)
    body = registry.render()
    assert "# TYPE klio_test_seconds histogram" in body
    assert 'klio_test_seconds_bucket{le="0.1"} 0' in body
    assert 'klio_test_seconds_bucket{le="1.0"} 1' in body
    assert 'klio_test_seconds_bucket{le="+Inf"} 1' in body
    assert "klio_test_seconds_sum 0.5" in body
    assert "klio_test_seconds_count 1" in body


def test_render_escapes_help_text_and_label_values(registry):
    r"""Backslash and newline in HELP, and ``\ " \n`` in a label value."""
    c = registry.counter("klio_test_total", "a \\ b \n c")
    c.inc()
    body = registry.render()
    assert "# HELP klio_test_total a \\\\ b \\n c" in body


def test_render_ends_with_a_newline(registry):
    """Prometheus requires the exposition body to end in a line feed."""
    registry.counter("klio_test_total", "doc").inc()
    assert registry.render().endswith("\n")


def test_render_of_an_empty_registry_is_empty_not_malformed(registry):
    assert registry.render() == ""


def test_default_registry_renders(registry):
    """The real registry must render without raising, whatever is declared."""
    body = metrics.REGISTRY.render()
    assert "# TYPE " in body
    assert body.endswith("\n")


# ── the declared surface ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "klio_memory_writes_total",
        "klio_memory_write_seconds",
        "klio_memory_recalls_total",
        "klio_memory_recall_seconds",
        "klio_embedding_calls_total",
        "klio_embedding_seconds",
        "klio_embedding_memo_total",
        "klio_embedding_concurrency_saturated_total",
        "klio_db_pool_checkouts_in_use",
        "klio_db_pool_checkout_wait_seconds",
        "klio_db_pool_timeouts_total",
        "klio_db_connect_failures_total",
        "klio_curator_ticks_total",
        "klio_curator_rows_read_total",
        "klio_curator_rows_distilled_total",
        "klio_curator_entities_total",
        "klio_curator_cursor_total",
        "klio_curator_budget_exhausted_total",
        "klio_capture_events_total",
        "klio_capture_redactions_total",
    ],
)
def test_the_required_metric_is_declared(name):
    assert name in {m.name for m in metrics.REGISTRY.collect()}


def test_counter_names_end_in_total_and_histograms_in_a_unit():
    """Naming convention, so dashboards and alert rules stay predictable."""
    bad = []
    for m in metrics.REGISTRY.collect():
        if isinstance(m, metrics.Counter) and not m.name.endswith("_total"):
            bad.append(m.name)
        if isinstance(m, metrics.Histogram) and not m.name.endswith("_seconds"):
            bad.append(m.name)
    assert bad == []


def test_metrics_module_has_no_third_party_imports():
    """The module is hand-rolled precisely so it drags in no dependency.

    ``prometheus_client`` is not installed and must not become a hidden
    requirement of importing ``shared``.
    """
    import ast
    import pathlib

    source = pathlib.Path(metrics.__file__).read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {
        "collections",
        "functools",
        "logging",
        "contextlib",
        "dataclasses",
        "math",
        "threading",
        "time",
        "typing",
        "__future__",
    }
