"""The single instrumentation module for Klio.

Every counter, gauge and histogram in the product is declared here, and call
sites do nothing but increment or observe. That indirection is the point: the
backend can be swapped — for ``prometheus_client``, OpenTelemetry, or a hosted
SDK — by rewriting this file alone, without touching a single call site.

Why hand-rolled
---------------
``prometheus_client`` is not installed in this environment, and adding a
dependency is the user's decision, not this module's. The Prometheus text
exposition format is small and stable, so the registry and the renderer are
implemented here directly. The module imports nothing outside the standard
library, and a test pins that, so importing ``shared`` never grows a new
requirement.

Label cardinality is a correctness property
-------------------------------------------
A time-series backend is destroyed by an unbounded label set long before it is
troubled by volume: one series per ``org_id`` is one series per tenant, forever,
retained. So this registry does not merely *discourage* high-cardinality labels,
it refuses them, in two layers:

* :data:`UNBOUNDED_LABEL_NAMES` — identifier-shaped names are rejected at
  declaration time, before a single sample exists;
* every label must declare the **finite set of values it may take**, and a
  value outside that set raises rather than silently minting a new series.

The second layer is the load-bearing one. A name-based denylist can always be
evaded by inventing a new name; requiring an enumerated value set means an
unbounded label cannot be expressed at all.

Thread safety
-------------
Instrumentation is touched from FastAPI request threads, ``anyio`` worker
threads and the curator's background loop simultaneously. Every mutation takes
a lock. The locks are uncontended in practice (an increment is a dict lookup
and a float add) and are never held across user code.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional, Sequence

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "Registry",
    "UnboundedLabelError",
    "REGISTRY",
    "render",
    "reset_for_test",
]


class UnboundedLabelError(ValueError):
    """Raised when a metric declaration would allow an unbounded label set."""


#: Label names that identify a *principal or record* rather than a *category*.
#: Each one has as many values as the product has rows, so each one is an
#: unbounded label set. Rejected at declaration time.
UNBOUNDED_LABEL_NAMES = frozenset(
    {
        "org_id",
        "user_id",
        "session_id",
        "agent_id",
        "project_id",
        "memory_id",
        "entity_id",
        "artifact_id",
        "key_id",
        "request_id",
        "trace_id",
        "email",
        "query",
        "query_text",
        "content",
        "path",
        "url",
        "ip",
        "user_agent",
    }
)

#: Ceiling on how many values one label may take. Well above any legitimate
#: category (the tool list is the largest at ~18) and well below anything that
#: would trouble a backend. A declaration that needs more is not a category.
MAX_LABEL_VALUES = 64

#: Default histogram buckets, in seconds. Chosen for the shape of the calls
#: actually being measured — an embedding round-trip, a pgvector scan, a pool
#: checkout — which span roughly a millisecond to a minute.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def _validate_labels(labels: Mapping[str, Sequence[str]]) -> None:
    """Reject any label declaration that would allow an unbounded series set.

    Raises:
        UnboundedLabelError: for an identifier-shaped name, an empty value set
            (which means "any value"), or a value set beyond
            :data:`MAX_LABEL_VALUES`.
    """
    for name, allowed in labels.items():
        if name in UNBOUNDED_LABEL_NAMES:
            raise UnboundedLabelError(
                f"label {name!r} identifies a principal or record, not a category; "
                "it would mint one time series per row. Bounded labels only."
            )
        if not allowed:
            raise UnboundedLabelError(
                f"label {name!r} declares no value set, which means any value; "
                "enumerate the values a label may take."
            )
        if len(allowed) > MAX_LABEL_VALUES:
            raise UnboundedLabelError(
                f"label {name!r} declares {len(allowed)} values, above the "
                f"{MAX_LABEL_VALUES} ceiling; that is not a category."
            )


class _Metric:
    """Common state for every metric type: name, help text, declared labels."""

    #: Overridden by each subclass; emitted in the ``# TYPE`` line.
    kind = "untyped"

    def __init__(
        self,
        name: str,
        documentation: str,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        declared = {k: tuple(v) for k, v in (labels or {}).items()}
        _validate_labels(declared)
        self.name = name
        self.documentation = documentation
        #: Ordered mapping of label name -> the finite set of permitted values.
        #: Insertion order is the render order, so a series' identity is stable.
        self.labels: dict[str, tuple[str, ...]] = declared
        self._lock = threading.Lock()

    def _key(self, given: Mapping[str, str]) -> tuple[str, ...]:
        """Validate the supplied labels and return the series key.

        Raises:
            ValueError: on a missing label, an undeclared label, or a value
                outside the label's declared set.
        """
        if set(given) != set(self.labels):
            raise ValueError(
                f"{self.name}: expected labels {sorted(self.labels)}, "
                f"got {sorted(given)}"
            )
        key = []
        for name, allowed in self.labels.items():
            value = given[name]
            if value not in allowed:
                raise ValueError(
                    f"{self.name}: label {name}={value!r} is not one of "
                    f"{list(allowed)}; an undeclared value would mint a new series."
                )
            key.append(value)
        return tuple(key)

    def _samples(self) -> list[tuple[str, dict[str, str], float]]:
        """Return ``(suffix, labels, value)`` triples for the renderer."""
        raise NotImplementedError


class Counter(_Metric):
    """A monotonically increasing count."""

    kind = "counter"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Add ``amount`` to the series identified by ``labels``.

        Raises:
            ValueError: if ``amount`` is negative (a counter cannot go down)
                or the labels do not match the declaration.
        """
        if amount < 0:
            raise ValueError(f"{self.name}: a counter cannot decrease")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        """Current value of one series. Absent series read as zero."""
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _samples(self) -> list[tuple[str, dict[str, str], float]]:
        with self._lock:
            items = list(self._values.items())
        names = list(self.labels)
        return [("", dict(zip(names, key)), val) for key, val in items]


class Gauge(_Metric):
    """A value that goes up and down — an in-flight count, a pool depth."""

    kind = "gauge"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        """Replace the series' value."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Add to the series' value."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        """Subtract from the series' value."""
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        """Current value of one series. Absent series read as zero."""
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    @contextmanager
    def track_inprogress(self, **labels: str) -> Iterator[None]:
        """Hold the gauge up for the duration of the block, raise or not."""
        self.inc(**labels)
        try:
            yield
        finally:
            self.dec(**labels)

    def _samples(self) -> list[tuple[str, dict[str, str], float]]:
        with self._lock:
            items = list(self._values.items())
        names = list(self.labels)
        return [("", dict(zip(names, key)), val) for key, val in items]


@dataclass(frozen=True)
class HistogramSnapshot:
    """An immutable read of one histogram series."""

    #: ``(upper_bound, cumulative_count)`` pairs, excluding the ``+Inf`` bucket.
    buckets: tuple[tuple[float, int], ...]
    #: Total number of observations, i.e. the ``+Inf`` bucket.
    count: int
    #: Sum of all observed values.
    total: float


class Histogram(_Metric):
    """A latency (or size) distribution over fixed buckets."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, documentation, labels)
        self.buckets: tuple[float, ...] = tuple(sorted(buckets))
        if not self.buckets:
            raise ValueError(f"{name}: a histogram needs at least one bucket")
        self._counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation into the series identified by ``labels``."""
        key = self._key(labels)
        with self._lock:
            counts = self._counts.get(key)
            if counts is None:
                counts = [0] * (len(self.buckets) + 1)
                self._counts[key] = counts
                self._sums[key] = 0.0
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[index] += 1
            counts[-1] += 1  # the +Inf bucket, i.e. the observation count
            self._sums[key] = self._sums[key] + value

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        """Observe the wall-clock duration of the block.

        The observation is recorded in a ``finally``, so the latency of a
        *failure* is measured too. A histogram that silently drops the slow,
        exception-raising calls is the one that hides an incident.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self.observe(time.monotonic() - started, **labels)

    def snapshot(self, **labels: str) -> HistogramSnapshot:
        """An immutable read of one series' distribution."""
        key = self._key(labels)
        with self._lock:
            counts = list(self._counts.get(key, [0] * (len(self.buckets) + 1)))
            total = self._sums.get(key, 0.0)
        return HistogramSnapshot(
            buckets=tuple(zip(self.buckets, counts[:-1])),
            count=counts[-1],
            total=total,
        )

    def _samples(self) -> list[tuple[str, dict[str, str], float]]:
        with self._lock:
            items = [(key, list(counts)) for key, counts in self._counts.items()]
            sums = dict(self._sums)
        names = list(self.labels)
        out: list[tuple[str, dict[str, str], float]] = []
        for key, counts in items:
            base = dict(zip(names, key))
            for index, bound in enumerate(self.buckets):
                out.append(
                    ("_bucket", {**base, "le": _format_bound(bound)}, counts[index])
                )
            out.append(("_bucket", {**base, "le": "+Inf"}, counts[-1]))
            out.append(("_sum", dict(base), sums[key]))
            out.append(("_count", dict(base), counts[-1]))
        return out


def _format_bound(bound: float) -> str:
    """Render a bucket upper bound the way Prometheus expects to read it back."""
    return repr(float(bound))


def _escape_help(text: str) -> str:
    """Escape a HELP line: backslash and newline only, per the spec."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label_value(value: str) -> str:
    """Escape a label value: backslash, double quote and newline."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_value(value: float) -> str:
    """Render a sample value, preferring the integral form when exact."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


class Registry:
    """A named collection of metrics that renders as one exposition body."""

    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _declare(self, metric: _Metric) -> _Metric:
        """Register ``metric``, or return the identical existing declaration.

        Re-declaration is idempotent so that importing this module twice under
        different paths — which happens in tests — does not double-register.
        A re-declaration with a *different* shape is a programming error and
        raises, because two call sites disagreeing about a metric's labels
        produces samples that cannot be aggregated.
        """
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is None:
                self._metrics[metric.name] = metric
                return metric
        if type(existing) is not type(metric) or existing.labels != metric.labels:
            raise ValueError(
                f"{metric.name} is already declared with a different shape: "
                f"{type(existing).__name__}{existing.labels} vs "
                f"{type(metric).__name__}{metric.labels}"
            )
        return existing

    def _force_declare_for_test(self, metric: _Metric) -> _Metric:
        """Register ``metric`` bypassing nothing but intended for tests only.

        Exists so a test can build a registry containing a *deliberately bad*
        metric and prove that the cardinality assertion catches it — a positive
        control for a check that would otherwise be indistinguishable from an
        inert one. Never call this from product code.
        """
        with self._lock:
            self._metrics[metric.name] = metric
        return metric

    def counter(
        self,
        name: str,
        documentation: str,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Counter:
        """Declare (or fetch) a counter on this registry."""
        return self._declare(Counter(name, documentation, labels))  # type: ignore[return-value]

    def gauge(
        self,
        name: str,
        documentation: str,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Gauge:
        """Declare (or fetch) a gauge on this registry."""
        return self._declare(Gauge(name, documentation, labels))  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        documentation: str,
        labels: Optional[Mapping[str, Sequence[str]]] = None,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        """Declare (or fetch) a histogram on this registry."""
        return self._declare(Histogram(name, documentation, labels, buckets))  # type: ignore[return-value]

    def collect(self) -> list[_Metric]:
        """Every declared metric, in declaration order."""
        with self._lock:
            return list(self._metrics.values())

    def render(self) -> str:
        """Render every metric in the Prometheus text exposition format.

        A metric with no observations yet still emits its ``# HELP`` and
        ``# TYPE`` lines, so a scraper sees the series exist before the first
        event rather than after it — the difference between "no writes are
        failing" and "no data".
        """
        lines: list[str] = []
        for metric in self.collect():
            lines.append(f"# HELP {metric.name} {_escape_help(metric.documentation)}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            for suffix, labels, value in metric._samples():
                rendered = ",".join(
                    f'{k}="{_escape_label_value(v)}"' for k, v in labels.items()
                )
                selector = f"{{{rendered}}}" if rendered else ""
                lines.append(f"{metric.name}{suffix}{selector} {_format_value(value)}")
        return "".join(f"{line}\n" for line in lines)

    def reset(self) -> None:
        """Drop every declared metric. Test-support only."""
        with self._lock:
            self._metrics.clear()


#: The process-wide registry. Everything below is declared on it at import.
REGISTRY = Registry()


def render() -> str:
    """Render the process-wide registry. What ``/metrics`` returns."""
    return REGISTRY.render()


def reset_for_test() -> None:
    """Zero every declared series without dropping the declarations.

    Tests assert on deltas, and a process-wide registry makes every test see
    every other test's increments. This clears the *values* and keeps the
    *shape*, so a test never has to re-import the module to get a clean slate.
    """
    for metric in REGISTRY.collect():
        with metric._lock:
            for attr in ("_values", "_counts", "_sums"):
                store = getattr(metric, attr, None)
                if store is not None:
                    store.clear()


# ═══════════════════════════════════════════════════════════════════════
# The declared surface. Every metric in the product is below this line.
#
# Label discipline, restated because it is easy to lose: a label is a
# *category with a closed set of values*. Outcomes, scopes, tool names.
# Never an identifier, never anything a caller supplies as free text.
# ═══════════════════════════════════════════════════════════════════════

#: The outcome vocabulary shared by the write and recall paths. ``quota`` is
#: separate from ``error`` because a plan limit being hit is a *product* signal
#: (someone should upgrade) while an error is an *operational* one (someone
#: should be paged), and an alert rule must be able to tell them apart.
OUTCOMES: tuple[str, ...] = ("ok", "quota", "error")

#: Memory visibility scopes, mirroring the product's scope levels.
SCOPES: tuple[str, ...] = ("private", "project", "org", "unknown")

#: Every MCP tool name. Bounded by the tool surface, which is a code artifact,
#: not caller input — an unknown name is recorded as ``other`` rather than
#: minting a series, so a new tool cannot silently widen the label set.
TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "remember",
    "note",
    "decide",
    "plan",
    "observe",
    "explore",
    "forget",
    "compress",
    "share",
    "space",
    "init",
    "klio_retrieve",
    "artifact_put",
    "artifact_get",
    "project_create",
    "project_grant",
    "project_list",
    "project_members",
    "other",
)


def tool_label(name: Optional[str]) -> str:
    """Coerce a tool name to a declared label value.

    The guard that keeps :data:`TOOL_NAMES` honest: a name this module has not
    heard of becomes ``other`` instead of raising or creating a new series.
    Instrumentation must never be the thing that breaks a tool call.
    """
    return name if name in TOOL_NAMES else "other"


# ── writes and recalls ────────────────────────────────────────────────

MEMORY_WRITES = REGISTRY.counter(
    "klio_memory_writes_total",
    "Memory writes, by outcome.",
    labels={"outcome": OUTCOMES},
)

MEMORY_WRITE_SECONDS = REGISTRY.histogram(
    "klio_memory_write_seconds",
    "End-to-end memory write latency, by outcome.",
    labels={"outcome": OUTCOMES},
)

MEMORY_RECALLS = REGISTRY.counter(
    "klio_memory_recalls_total",
    "Memory recalls, by outcome.",
    labels={"outcome": OUTCOMES},
)

MEMORY_RECALL_SECONDS = REGISTRY.histogram(
    "klio_memory_recall_seconds",
    "End-to-end memory recall latency, by outcome.",
    labels={"outcome": OUTCOMES},
)

MEMORY_RECALL_RESULTS = REGISTRY.counter(
    "klio_memory_recall_results_total",
    "Rows returned by recall, by scope. A recall that returns nothing is the "
    "failure mode users actually report, and it is invisible in an error rate.",
    labels={"scope": SCOPES},
)

TOOL_CALLS = REGISTRY.counter(
    "klio_tool_calls_total",
    "MCP tool invocations, by tool and outcome.",
    labels={"tool": TOOL_NAMES, "outcome": OUTCOMES},
)

TOOL_SECONDS = REGISTRY.histogram(
    "klio_tool_seconds",
    "MCP tool latency, by tool.",
    labels={"tool": TOOL_NAMES},
)

# ── embedding ─────────────────────────────────────────────────────────

#: Embedding outcomes. ``timeout`` is its own value rather than folded into
#: ``error`` because they call for different responses: a timeout means the
#: backend is slow, an error means it is wrong.
EMBED_OUTCOMES: tuple[str, ...] = ("ok", "timeout", "error")

EMBEDDING_CALLS = REGISTRY.counter(
    "klio_embedding_calls_total",
    "Embedding backend calls, by outcome.",
    labels={"outcome": EMBED_OUTCOMES},
)

EMBEDDING_SECONDS = REGISTRY.histogram(
    "klio_embedding_seconds",
    "Embedding backend call latency, excluding time queued behind the concurrency cap.",
)

EMBEDDING_MEMO = REGISTRY.counter(
    "klio_embedding_memo_total",
    "Embedding memo lookups, by result. A single write embeds the same string "
    "twice, so a hit rate far below 50% means the memo is not working.",
    labels={"result": ("hit", "miss")},
)

EMBEDDING_CONCURRENCY_SATURATED = REGISTRY.counter(
    "klio_embedding_concurrency_saturated_total",
    "Embedding calls that had to queue because the concurrency cap was full. "
    "The saturation signal for the cap: nonzero and rising means the cap, not "
    "the backend, is the latency.",
)

EMBEDDING_IN_FLIGHT = REGISTRY.gauge(
    "klio_embedding_in_flight",
    "Embedding calls currently in flight, against the concurrency cap.",
)

# ── database pool ─────────────────────────────────────────────────────

DB_POOL_IN_USE = REGISTRY.gauge(
    "klio_db_pool_checkouts_in_use",
    "Pooled database connections currently checked out.",
)

DB_POOL_WAIT_SECONDS = REGISTRY.histogram(
    "klio_db_pool_checkout_wait_seconds",
    "Time spent waiting for a pooled database connection.",
)

DB_POOL_TIMEOUTS = REGISTRY.counter(
    "klio_db_pool_timeouts_total",
    "Pool checkouts that timed out — the pool-exhaustion signal.",
)

DB_CONNECT_FAILURES = REGISTRY.counter(
    "klio_db_connect_failures_total",
    "Connections the database refused or that could not be opened. This is "
    "what a pooler rejecting a startup parameter looks like from in here.",
)

# ── curator ───────────────────────────────────────────────────────────

CURATOR_TICKS = REGISTRY.counter(
    "klio_curator_ticks_total",
    "Curator passes, by outcome.",
    labels={"outcome": ("ok", "error")},
)

CURATOR_TICK_SECONDS = REGISTRY.histogram(
    "klio_curator_tick_seconds",
    "Curator pass duration.",
)

CURATOR_ROWS_READ = REGISTRY.counter(
    "klio_curator_rows_read_total",
    "Rows the curator read for distillation.",
)

CURATOR_ROWS_DISTILLED = REGISTRY.counter(
    "klio_curator_rows_distilled_total",
    "Rows the curator actually distilled. The gap against rows_read is the "
    "skip rate, which is the defect this instrumentation exists to surface.",
)

CURATOR_ENTITIES = REGISTRY.counter(
    "klio_curator_entities_total",
    "Entities the curator touched, by disposition.",
    labels={"disposition": ("created", "referenced")},
)

CURATOR_CURSOR = REGISTRY.counter(
    "klio_curator_cursor_total",
    "Curator cursor decisions. A cursor that is held rather than advanced, "
    "pass after pass, is a stalled curator that no error rate reports.",
    labels={"action": ("advanced", "held")},
)

CURATOR_BUDGET_EXHAUSTED = REGISTRY.counter(
    "klio_curator_budget_exhausted_total",
    "Curator passes that stopped early on a budget, by which budget.",
    labels={"budget": ("entity", "reconcile", "row", "time")},
)

# ── capture ───────────────────────────────────────────────────────────

CAPTURE_EVENTS = REGISTRY.counter(
    "klio_capture_events_total",
    "Capture events, by outcome.",
    labels={"outcome": ("accepted", "rejected", "error")},
)

CAPTURE_REDACTIONS = REGISTRY.counter(
    "klio_capture_redactions_total",
    "Capture payloads in which the redactor found and removed something.",
)

# ── readiness ─────────────────────────────────────────────────────────

READINESS_CHECKS = REGISTRY.counter(
    "klio_readiness_checks_total",
    "Readiness probes, by result. A run of ``not_ready`` with a healthy "
    "liveness probe is the shape of a dependency outage.",
    labels={"result": ("ready", "not_ready")},
)
