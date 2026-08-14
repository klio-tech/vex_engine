"""Shared session memory module.

Provides embedding generation, pgvector read/write, semantic search,
deduplication, fact extraction, and summary generation for agent session
memory.  All DB access uses raw SQL via ``sqlalchemy.text()`` and pgvector's
cosine distance operator (``<=>``) for similarity queries.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Optional

import anyio
import litellm
from litellm import acompletion, aembedding
from sqlalchemy import text

from shared import lexical, metrics, stale
from shared.env_config import parse_float, parse_int
from shared.redaction import redact

# Project ACLs. Safe at module scope: ``projects`` imports nothing from this
# module (recall inlines its membership subquery rather than calling back
# here), so there is no cycle to break with a deferred import.
from shared import projects

logger = logging.getLogger("agentguard.shared.memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _parse_dedup_threshold() -> float:
    """Read, validate, and clamp ``KLIO_DEDUP_THRESHOLD`` from the environment.

    A malformed (non-numeric) or out-of-range value must never crash module
    import — an operator typo in one env var would otherwise take down every
    process that imports this module. Instead we log a warning and fall back to
    the ``0.95`` default. The value is also clamped to the valid cosine-similarity
    range ``[0.0, 1.0]`` (a threshold outside it is meaningless: below 0 collapses
    everything, above 1 disables dedup entirely).
    """
    raw = os.environ.get("KLIO_DEDUP_THRESHOLD", "0.95")
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid KLIO_DEDUP_THRESHOLD=%r; using 0.95", raw)
        return 0.95
    if not 0.0 <= value <= 1.0:
        logger.warning("KLIO_DEDUP_THRESHOLD=%s out of [0,1]; using 0.95", value)
        return 0.95
    return value


DEDUP_THRESHOLD: float = _parse_dedup_threshold()
"""Cosine similarity threshold above which two memories are treated as duplicates.

Default ``0.95``, overridable at process start via the ``KLIO_DEDUP_THRESHOLD``
env var. Raised from the historical ``0.92`` because same-topic technical text
(code, JSON, config, prose on one subject) embeds very tightly: at 0.92
genuinely distinct notes cleared the bar and were collapsed into a single row
by :func:`find_similar_memory`. 0.95 keeps near-identical dedup while letting
distinct same-topic notes coexist. Read from the environment so it can be tuned
per deployment without a code change."""

DEFAULT_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
"""Default model used for embedding generation when env var is unset."""

DEFAULT_LLM_MODEL: str = "claude-haiku-4-5-20251001"
"""Fallback model for LLM calls when VERIFICATION_MODEL is unset."""

DEFAULT_LLM_TIMEOUT_S: float = 10.0
"""Default timeout (seconds) for LLM completion calls."""

MAX_EMBED_CHARS: int = 1800
"""Head-truncation budget for embedding input (first attempt).

The default embedding model caps input at 512 tokens; at ~4 chars/token a
1800-char head stays under that (≈450 tokens) for typical text. But this is a
*char* proxy for a *token* cap: dense code/JSON/base64 packs >1 token/char, so
a 1800-char head can still exceed 512 tokens and make the backend raise. That
residual risk is handled by :func:`embed_text`'s progressive-truncation retry
(see :data:`EMBED_MIN_CHARS`), so a long/dense note is never lost.

Truncation applies ONLY to the vector input — stored ``content`` is never
modified. This also prevents oversized content from returning HTTP 400 and
looping the embedder (which previously flooded logs at the host's rate limit)."""

GRAPH_EXPANSION_CAP: int = 5
"""Max rows recall's stage-B graph expansion may ADD beyond the vector stage.

A hard payload bound: a hub memory with hundreds of edges must not turn one
recall into a graph dump. The expansion itself is depth-1 and ladder-scoped in
SQL inside :func:`shared.graph_store.neighbors`; this cap only bounds volume.
"""

VIA_GRAPH: str = "graph"
"""Value of :attr:`MemoryRecord.via` on a row that arrived by graph expansion."""

VIA_LEXICAL: str = "lexical"
"""Value of :attr:`MemoryRecord.via` on a row only the lexical arm returned.

Same reasoning as :data:`VIA_GRAPH`. A lexical-only row has no cosine score to
report, and inventing one would misrepresent why it is present — so its
``relevance`` is ``None`` and this positive marker is what tells a caller the
row came from an exact-token match rather than a semantic one. A row BOTH arms
returned keeps its cosine ``relevance`` and no marker: it was a semantic match
that lexical merely agreed with.
"""

# Stage-C blended re-rank weights (spec: recall learns to walk). Similarity
# dominates; recency and edge-connectedness are equal minority signals. The
# pre-graph SQL rank used 0.85/0.15 with no connectedness term — the 0.15 the
# graph gets comes out of similarity's share, not recency's.
_RERANK_SIMILARITY_WEIGHT = 0.70
_RERANK_RECENCY_WEIGHT = 0.15
_RERANK_CONNECTEDNESS_WEIGHT = 0.15
_RECENCY_WINDOW_SECONDS = 30 * 86400

EMBED_MIN_CHARS: int = 256
"""Smallest truncation floor for :func:`embed_text`'s progressive retry.

When an embedding-backend call fails, :func:`embed_text` halves the input and
retries, down to this floor. A failure at (or below) the floor is treated as
the embedding service being genuinely unavailable → ``None``; it is NOT the
path for "the content was long". Consequently a short input that fails returns
``None`` on a single attempt (no needless retries), while a long/dense input is
shrunk repeatedly until it fits under the model's token cap."""

DEFAULT_EMBED_TIMEOUT_S: float = 15.0
"""Default per-attempt bound on one ``litellm.aembedding`` call, in seconds.

Overridable at run time via ``KLIO_EMBED_TIMEOUT_S``. Before this bound the
call was awaited with nothing at all: a hung embedding backend held the
awaiting coroutine forever, and — because the caller held a pooled DB
connection across that await — took the connection pool down with it.

15 s is generous for an embedding of at most :data:`MAX_EMBED_CHARS`
characters (a healthy gateway answers in tens of milliseconds) while still
being an order of magnitude below the 30 s pool checkout queue it used to
exhaust."""

DEFAULT_EMBED_MAX_CONCURRENCY: int = 32
"""Default cap on embedding calls this process may have in flight at once.

Overridable via ``KLIO_EMBED_MAX_CONCURRENCY``. This bounds fan-out at the
embedding *backend*, which nothing previously did: the DB pool incidentally
capped it at 15 only because every embed was holding a connection, and fixing
that (correctly) removed the accidental limit. Without an explicit cap, a burst
of writes becomes an unbounded burst at the gateway."""

EMBED_MEMO_TTL_S: float = 60.0
"""How long a computed embedding stays reusable, in seconds.

Deliberately short. This is a *within-request collapse*, not a cache: a single
write embeds the same string twice — once in :func:`find_similar_memory` to
look for a duplicate, once in :func:`write_memory` to store it — and this memo
exists to make that one call instead of two. A TTL of seconds is enough for
that and short enough that the process is not carrying a content cache around.

It lives at module scope rather than in a request-scoped context because the
two embeds happen inside :func:`shared.brain.write_scoped_memory`, which calls
the two primitives independently; a memo the caller has to open would only
help callers that had been updated to open it."""

EMBED_MEMO_MAX_ENTRIES: int = 256
"""Hard bound on memo size, evicting oldest-first.

At the model's 1024-dimension output this is a few megabytes — a bound, not a
budget. It exists so a write-heavy process cannot turn the memo into a leak."""

# Providers that support response_format via gateway
_JSON_MODE_PREFIXES = ("openai/", "gpt-")

# Credential-reproduction guard injected into every synthesis prompt so that
# the LLM is explicitly instructed not to echo secrets even in prose form.
_NO_CREDENTIALS_RULE = (
    "Never reproduce credentials, API keys, tokens, passwords, or other "
    "secrets — not even partially. If the content contains one, refer to it "
    "generically (e.g. 'a Stripe test key was provided'). "
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """In-memory representation of a ``session_memories`` row."""

    id: uuid.UUID
    session_id: str
    org_id: str
    agent_id: str
    memory_type: str
    content: str
    execution_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    confidence: Optional[float] = None
    sequence_number: Optional[int] = None
    scope: str = "session"
    status: str = "active"
    superseded_by: Optional[uuid.UUID] = None
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    space_id: Optional[str] = None
    project_id: Optional[str] = None
    relevance: Optional[float] = None
    user_id: Optional[str] = None
    provenance: str = "EXTRACTED"
    via: Optional[str] = None
    """How this record reached the caller: ``"graph"`` for a row that rode in
    on :data:`VIA_GRAPH` expansion, ``None`` for everything else.

    A POSITIVE marker on purpose. Expanded rows used to be identifiable only
    by ``relevance is None``, and structural compression strips null-valued
    keys row by row — so on the wire the expanded row simply had no
    ``relevance`` key and nothing at all separated a semantic match from a
    graph walk. An absent key is not a signal; a value is."""
    revision: Optional[str] = None
    """The row's current CAS token (migration 043). ``None`` for records built
    without selecting the column (dedup lookups, verification's session-scoped
    reads) — those callers never round-trip a ``base_revision``, so its
    absence costs them nothing. A caller that DOES read this back and pass it
    to :func:`supersede_with_revision` gets the optimistic-concurrency
    guarantee; one that ignores it keeps writing exactly as before."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def release_connection(db_session: Any) -> None:
    """Return *db_session*'s pooled connection to the pool before an ``await``.

    SQLAlchemy 2.0 autobegins a transaction on the first ``execute()`` and
    holds the checked-out connection until commit/rollback/close. Every write
    path in this engine opens with synchronous DB work (a quota check, project
    resolution, space resolution) and *then* awaits an embedding, so each
    in-flight write pinned one pooled connection for the whole duration of a
    network call to a third party. Against the production pool
    (``pool_size=5, max_overflow=10`` -> 15 checkouts) with a 1 s embed, the
    16th concurrent write queued and raised ``TimeoutError``; at concurrency 40,
    25 of 40 failed. With the connection released, 200 concurrent writes
    completed in 2.24 s.

    ``commit`` rather than ``rollback`` because at every call site the pending
    state is work the caller intends to keep — a resolved-or-created space, a
    metered usage row — and discarding it to free a connection would trade an
    availability bug for a correctness one. Committing a read-only transaction
    is free.

    A caller that exposes no ``commit`` (a raw DBAPI connection, a test fake)
    is left alone: this is an optimisation of the pooling behaviour, not a
    contract change, and it must never be the reason a call site starts
    failing.

    A failed commit is NOT swallowed. It means the transaction is already
    broken, and continuing on to embed and insert would only surface the same
    failure later with less context.
    """
    commit = getattr(db_session, "commit", None)
    if commit is None:
        return
    commit()


def _embed_model() -> str:
    """The embedding model this process is configured to use."""
    return os.environ.get("MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _embed_timeout_s() -> float:
    """Per-attempt embedding timeout, read at call time.

    Read here rather than at import so an operator can change it without a
    redeploy, and guarded by :func:`shared.env_config.parse_float` so a typo
    (``KLIO_EMBED_TIMEOUT_S='10s'``) degrades to the default instead of raising
    out of every write — the same failure ``VERIFICATION_TIMEOUT_S`` produced.
    """
    return parse_float(
        os.environ.get("KLIO_EMBED_TIMEOUT_S"),
        default=DEFAULT_EMBED_TIMEOUT_S,
        name="KLIO_EMBED_TIMEOUT_S",
        minimum=0.01,
        maximum=600.0,
    )


#: In-flight embedding limiters, one per event loop.
#:
#: An ``asyncio.Semaphore`` belongs to the loop that first suspends on it, so a
#: module-level singleton would leak waiters across loops (every process that
#: runs more than one loop — notably the test suite — would eventually block on
#: a dead one). Keyed by loop, and dropped when the loop closes.
_embed_limiters: "dict[Any, asyncio.Semaphore]" = {}
_embed_limiter_lock = threading.Lock()


def _embed_limiter() -> "asyncio.Semaphore":
    """The concurrency limiter for the running event loop."""
    loop = asyncio.get_running_loop()
    with _embed_limiter_lock:
        limiter = _embed_limiters.get(loop)
        if limiter is None:
            limit = parse_int(
                os.environ.get("KLIO_EMBED_MAX_CONCURRENCY"),
                default=DEFAULT_EMBED_MAX_CONCURRENCY,
                name="KLIO_EMBED_MAX_CONCURRENCY",
                minimum=1,
                maximum=4096,
            )
            limiter = asyncio.Semaphore(limit)
            _embed_limiters[loop] = limiter
        # Closed loops can never be used again; do not hold their limiters.
        for dead in [loop_ for loop_ in _embed_limiters if loop_.is_closed()]:
            del _embed_limiters[dead]
        return limiter


def reset_embed_concurrency_limiter() -> None:
    """Drop cached limiters so a changed cap takes effect. For tests."""
    with _embed_limiter_lock:
        _embed_limiters.clear()


#: content-key -> (vector, monotonic timestamp). See :data:`EMBED_MEMO_TTL_S`.
_embed_memo: "OrderedDict[tuple[str, str], tuple[list[float], float]]" = OrderedDict()
_embed_memo_lock = threading.Lock()


def _memo_get(key: tuple[str, str]) -> Optional[list[float]]:
    """The memoized vector for *key*, or ``None`` if absent or expired."""
    now = time.monotonic()
    with _embed_memo_lock:
        entry = _embed_memo.get(key)
        if entry is None:
            return None
        vector, stored_at = entry
        if now - stored_at >= EMBED_MEMO_TTL_S:
            del _embed_memo[key]
            return None
        _embed_memo.move_to_end(key)
        return vector


def _memo_put(key: tuple[str, str], vector: list[float]) -> None:
    """Memoize *vector*, evicting oldest-first at :data:`EMBED_MEMO_MAX_ENTRIES`."""
    with _embed_memo_lock:
        _embed_memo[key] = (vector, time.monotonic())
        _embed_memo.move_to_end(key)
        while len(_embed_memo) > EMBED_MEMO_MAX_ENTRIES:
            _embed_memo.popitem(last=False)


def clear_embedding_memo() -> None:
    """Empty the embedding memo. For tests and for operational reset."""
    with _embed_memo_lock:
        _embed_memo.clear()


def embedding_memo_size() -> int:
    """Current memo occupancy. For tests and metrics."""
    with _embed_memo_lock:
        return len(_embed_memo)


async def _aembedding_call(text_input: str) -> list[float]:
    """Perform a single embedding-backend call via ``litellm.aembedding()``.

    Reads the model name from the ``MEMORY_EMBEDDING_MODEL`` env var, falling
    back to :data:`DEFAULT_EMBEDDING_MODEL`, and optional gateway credentials
    from the ``LITELLM_*`` / ``OPENAI_*`` env vars.

    Isolated from :func:`embed_text` so the latter can retry it with
    progressively smaller input without duplicating the litellm wiring. Raises
    whatever the backend raises (e.g. an over-token-budget HTTP 400), and
    :class:`asyncio.TimeoutError` when the call exceeds
    :func:`_embed_timeout_s`.

    Two bounds are applied here rather than in :func:`embed_text` so that every
    attempt of the progressive-truncation retry is bounded, not just the first:

    * a **timeout**, so a hung backend cannot hold the caller (and, before the
      release below, the caller's pooled DB connection) indefinitely;
    * a **concurrency cap**, so a burst of writes cannot become an unbounded
      burst at the gateway. The cap is acquired *outside* the timeout, so time
      spent queueing behind other embeds is not charged against the backend's
      own budget — otherwise a busy process would report timeouts for calls the
      backend never received.

    Returns:
        The embedding vector as a list of floats.
    """
    model = _embed_model()
    api_base = os.environ.get("LITELLM_API_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    kwargs: dict[str, Any] = {"model": model, "input": [text_input]}
    if api_base:
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "openai"
    if api_key:
        kwargs["api_key"] = api_key

    limiter = _embed_limiter()
    # Read the cap's state *before* awaiting it. A limiter that is already at
    # zero means this call is about to queue, and that is the saturation signal
    # — the difference between "the backend is slow" and "we are waiting behind
    # our own limit", which look identical in a latency histogram alone.
    if limiter.locked():
        metrics.record(metrics.EMBEDDING_CONCURRENCY_SATURATED.inc)

    async with limiter:
        started = time.monotonic()
        metrics.record(metrics.EMBEDDING_IN_FLIGHT.inc)
        try:
            response = await asyncio.wait_for(
                aembedding(**kwargs), timeout=_embed_timeout_s()
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Counted apart from ``error`` deliberately: a timeout means the
            # backend is slow, an error means it is wrong, and the two call
            # for different responses.
            metrics.record(metrics.EMBEDDING_CALLS.inc, outcome="timeout")
            raise
        except Exception:
            metrics.record(metrics.EMBEDDING_CALLS.inc, outcome="error")
            raise
        else:
            metrics.record(metrics.EMBEDDING_CALLS.inc, outcome="ok")
        finally:
            # Both in a ``finally``: the latency of a failure is still latency,
            # and a gauge that is not released on the error path ratchets up
            # until it reads as permanent saturation.
            metrics.record(metrics.EMBEDDING_IN_FLIGHT.dec)
            metrics.record(
                metrics.EMBEDDING_SECONDS.observe, time.monotonic() - started
            )

    item = response.data[0]
    return item["embedding"] if isinstance(item, dict) else item.embedding


async def embed_text(text_input: str) -> Optional[list[float]]:
    """Generate an embedding vector via ``litellm.aembedding()``.

    Reads the model name from the ``MEMORY_EMBEDDING_MODEL`` env var, falling
    back to :data:`DEFAULT_EMBEDDING_MODEL`.

    Robustness — progressive truncation so a long/dense note never fails a
    write: :data:`MAX_EMBED_CHARS` is only a *char* proxy for the model's 512
    *token* cap, and dense code/JSON packs >1 token/char, so a char-truncated
    input can still exceed the cap and make the backend raise. Rather than lose
    the write, ``embed_text`` retries with the input halved on each backend
    error, down to the :data:`EMBED_MIN_CHARS` floor. ``None`` is returned ONLY
    when even the floor-sized attempt fails (the embedding service is genuinely
    unavailable) — never merely because the content was long.

    Truncation applies ONLY to the vector input; the caller's stored ``content``
    is never modified by this function.

    Availability — three bounds, none of which existed before:

    * every attempt is **timed out** (:func:`_embed_timeout_s`), so a hung
      backend cannot hold the caller forever;
    * in-flight calls are **capped** (:data:`DEFAULT_EMBED_MAX_CONCURRENCY`);
    * an identical ``(model, content)`` pair is **memoized** for
      :data:`EMBED_MEMO_TTL_S`, because a single write embeds the same string
      twice — once to look for a duplicate, once to store it.

    A timeout degrades to ``None`` exactly like any other backend error, and is
    explicitly NOT routed into the progressive-truncation retry: shrinking the
    input does not un-hang a backend, and retrying would turn one stall into
    four.

    Returns:
        A list of floats representing the embedding, or ``None`` only when the
        embedding backend fails at every attempted size.
    """
    # The memo is keyed on the ORIGINAL input, not the truncated candidate: two
    # callers passing the same content must collapse to one call regardless of
    # which truncation path each would have taken (they take the same one).
    memo_key = (_embed_model(), text_input)
    memoized = _memo_get(memo_key)
    if memoized is not None:
        # A single write embeds the same string twice — once to probe for a
        # duplicate, once to store it — so the steady-state hit rate should sit
        # near 50%. A rate far below that means the memo has stopped working
        # and every write is paying the backend twice again.
        metrics.record(metrics.EMBEDDING_MEMO.inc, result="hit")
        return memoized
    metrics.record(metrics.EMBEDDING_MEMO.inc, result="miss")

    # First attempt: head-truncate to the char budget (unchanged behavior).
    if text_input and len(text_input) > MAX_EMBED_CHARS:
        logger.debug(
            "embed_text: truncating input from %d to %d chars",
            len(text_input),
            MAX_EMBED_CHARS,
        )
        candidate = text_input[:MAX_EMBED_CHARS]
    else:
        candidate = text_input

    while True:
        try:
            vector = await _aembedding_call(candidate)
            # Only successes are memoized. Memoizing a failure would pin a
            # transient gateway blip to that content for the whole TTL, so a
            # retry that would have worked returns None instead.
            _memo_put(memo_key, vector)
            return vector
        except asyncio.TimeoutError:
            # A stall is not an over-length rejection. Shrinking the input does
            # not un-hang a backend, so this must NOT fall into the truncation
            # retry below: that would turn one stall into four calls at an
            # already-failing backend, and would silently store a TRUNCATED
            # vector for full content if the stall cleared mid-retry.
            logger.warning(
                "embed_text: embedding timed out after %.2fs; returning None",
                _embed_timeout_s(),
            )
            return None
        except (litellm.ContextWindowExceededError, litellm.BadRequestError):
            # Over-length-shaped rejection: shrinking the vector input can make
            # the SAME backend accept it, so retry with harder truncation rather
            # than lose the write.
            current_len = len(candidate) if candidate else 0
            if current_len <= EMBED_MIN_CHARS:
                # At/below the floor and still rejected as over-length: the input
                # cannot be shrunk any further usefully. Preserve the historical
                # "cannot embed → None" contract rather than loop forever.
                logger.warning(
                    "embed_text: over-length even at the %d-char floor; returning None",
                    current_len,
                    exc_info=True,
                )
                return None
            new_len = max(EMBED_MIN_CHARS, current_len // 2)
            logger.warning(
                "embed_text: over-length at %d chars; "
                "retrying with harder truncation to %d chars",
                current_len,
                new_len,
            )
            candidate = candidate[:new_len]
        except Exception:
            # Any non-over-length failure (backend down, timeout, transient
            # gateway error) is NOT fixed by shrinking: truncating here would
            # either fan out up to four calls at an already-failing backend or,
            # worse, silently embed TRUNCATED content once a transient error
            # cleared. Return None immediately on the original-size attempt,
            # preserving the "backend error → None" contract.
            logger.warning(
                "embed_text: embedding backend error; returning None",
                exc_info=True,
            )
            return None


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _extract_json(raw_text: str) -> Optional[dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown fences."""
    # Direct parse
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # First { ... } block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


async def call_llm(
    prompt: str,
    system: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Call an LLM via ``litellm.acompletion()`` and parse the JSON response.

    Configuration is read from environment variables:

    * ``VERIFICATION_MODEL`` – model name (default ``claude-haiku-4-5-20251001``)
    * ``LITELLM_API_URL`` – optional API base URL
    * ``LITELLM_API_KEY`` – optional API key

    Returns:
        Parsed JSON dict, or ``None`` on timeout / error.
    """
    model = os.environ.get("VERIFICATION_MODEL", DEFAULT_LLM_MODEL)
    # Guarded, because the bare ``float()`` that used to be here sat ABOVE the
    # ``try`` below whose whole job is to make LLM failure degrade to ``None``.
    # Executed: ``VERIFICATION_TIMEOUT_S='10s'`` raised
    # ``ValueError: could not convert string to float: '10s'`` out of every
    # ``call_llm``. The identical defect exists at
    # ``verification-engine/engine/llm_client.py``; both are fixed.
    timeout_s = parse_float(
        os.environ.get("VERIFICATION_TIMEOUT_S"),
        default=DEFAULT_LLM_TIMEOUT_S,
        name="VERIFICATION_TIMEOUT_S",
        minimum=0.1,
        maximum=600.0,
    )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    api_base = os.environ.get("LITELLM_API_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout_s,
        "temperature": 0.0,
    }

    if api_base:
        kwargs["custom_llm_provider"] = "openai"
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    try:
        response = await acompletion(**kwargs)
        content = response.choices[0].message.content
        result = _extract_json(content)
        if result is None:
            logger.warning("LLM returned non-JSON response: %s", content[:200])
        return result
    except Exception:
        logger.warning("LLM call failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Summary & fact extraction
# ---------------------------------------------------------------------------

#: Per-field cap on caller-supplied text interpolated into an LLM prompt.
#: Matches ``MAX_PROMPT_INPUT_CHARS`` in ``sync-gateway`` and ``async-worker``,
#: which bound the same values at their own call sites. Keeping the number
#: identical means the two layers agree rather than compounding.
MAX_PROMPT_INPUT_CHARS = 20_000


def _bounded_prompt_input(value: Any) -> str:
    """``value`` as text, capped at :data:`MAX_PROMPT_INPUT_CHARS`.

    ``output`` and ``input_data`` are whatever an agent turn produced — an
    unbounded blob as far as this module is concerned. Interpolated raw, a
    40 MB tool result becomes a 40 MB request to the vendor: a cost and latency
    fault that surfaces as a timeout indistinguishable from an LLM outage.

    The bound lives here rather than only at the call sites so it holds for
    every caller, including the ones that have not been written yet. The
    call-site caps in ``sync-gateway``/``async-worker`` stay — truncating twice
    at the same limit is a no-op, and the earlier cut also bounds what those
    services carry in memory.

    The marker is visible in the prompt on purpose: a model that receives a cut
    transcript should be able to tell that it was cut.
    """
    as_text = value if isinstance(value, str) else str(value)
    if len(as_text) <= MAX_PROMPT_INPUT_CHARS:
        return as_text
    return as_text[:MAX_PROMPT_INPUT_CHARS] + "…[truncated]"


async def generate_summary(
    output: Any,
    task: Optional[str] = None,
    input_data: Optional[Any] = None,
) -> Optional[str]:
    """Generate a compact 1-2 sentence turn summary via :func:`call_llm`.

    Returns:
        The summary string, or ``None`` if the LLM call fails.
    """
    parts = [f"Output: {_bounded_prompt_input(output)}"]
    if task:
        parts.append(f"Task: {_bounded_prompt_input(task)}")
    if input_data:
        parts.append(f"Input: {_bounded_prompt_input(input_data)}")

    prompt = (
        "Summarize the following agent turn in 1-2 concise sentences. "
        "Focus on what the agent did and the outcome. "
        + _NO_CREDENTIALS_RULE
        + "\n\n"
        + "\n".join(parts)
        + '\n\nRespond with JSON: {"summary": "<your summary>"}'
    )

    result = await call_llm(prompt)
    if result is None:
        return None

    return result.get("summary")


async def extract_facts(
    output: Any,
    task: Optional[str] = None,
    input_data: Optional[Any] = None,
) -> list[str]:
    """Extract discrete verifiable facts from an agent turn via :func:`call_llm`.

    Returns:
        A list of fact strings, or an empty list if the LLM call fails.
    """
    parts = [f"Output: {_bounded_prompt_input(output)}"]
    if task:
        parts.append(f"Task: {_bounded_prompt_input(task)}")
    if input_data:
        parts.append(f"Input: {_bounded_prompt_input(input_data)}")

    prompt = (
        "Extract discrete, verifiable facts from the following agent turn. "
        "Each fact should be a short, self-contained statement. "
        + _NO_CREDENTIALS_RULE
        + "\n\n"
        + "\n".join(parts)
        + '\n\nRespond with JSON: {"facts": ["fact1", "fact2", ...]}'
    )

    result = await call_llm(prompt)
    if result is None:
        return []

    facts = result.get("facts", [])
    if not isinstance(facts, list):
        return []
    return facts


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: tuple) -> MemoryRecord:
    """Convert a raw DB row tuple to a :class:`MemoryRecord`.

    Expected column order (first 16 columns, required):
        id, session_id, org_id, agent_id, execution_id, memory_type,
        content, embedding, confidence, sequence_number, scope, status,
        superseded_by, metadata, created_at, updated_at

    Optional trailing columns (additive, read only if present so existing
    16-column callers are unaffected):
        space_id (index 16), project_id (index 17), user_id (index 18),
        provenance (index 19), revision (index 20)

    ``user_id`` and ``provenance`` are read positionally like the two before
    them, and every caller that selects them appends them in this order. A
    caller that selects neither still gets a valid record: ``user_id`` stays
    ``None`` (the pre-migration-031 state for 37,605 rows) and ``provenance``
    falls back to the column's own default, ``EXTRACTED``. ``revision``
    (migration 043) follows the identical convention: absent → ``None``,
    present → stringified so it round-trips as JSON without a caller having to
    know it started life as a ``UUID``.
    """
    return MemoryRecord(
        id=row[0],
        session_id=row[1],
        org_id=row[2],
        agent_id=row[3],
        execution_id=row[4],
        memory_type=row[5],
        content=row[6],
        embedding=row[7],
        confidence=row[8],
        sequence_number=row[9],
        scope=row[10],
        status=row[11],
        superseded_by=row[12],
        metadata=row[13] if row[13] is not None else {},
        created_at=row[14],
        updated_at=row[15],
        space_id=row[16] if len(row) > 16 else None,
        project_id=row[17] if len(row) > 17 else None,
        user_id=row[18] if len(row) > 18 else None,
        provenance=(row[19] if len(row) > 19 and row[19] is not None else "EXTRACTED"),
        revision=(
            str(row[20]) if len(row) > 20 and row[20] is not None else None
        ),
    )


@metrics.tracked(metrics.MEMORY_WRITES, metrics.MEMORY_WRITE_SECONDS)
async def write_memory(
    db_session: Any,
    session_id: str,
    org_id: str,
    agent_id: str,
    execution_id: str,
    memory_type: str,
    content: str,
    confidence: Optional[float] = None,
    sequence_number: Optional[int] = None,
    metadata: Optional[dict] = None,
    recall_hidden: bool = False,
    user_id: Optional[str] = None,
    provenance: str = "EXTRACTED",
) -> Optional[MemoryRecord]:
    """Generate an embedding and INSERT a new row into ``session_memories``.

    Args:
        db_session: A synchronous SQLAlchemy session/connection.
        session_id: The session this memory belongs to.
        org_id: Owning org id (tenant scope).
        agent_id: Authoring agent principal.
        execution_id: Verification execution id (nullable — pass ``None`` for
            brain writes that are not tied to a specific execution).
        memory_type: The memory type string (e.g. ``observation``, ``fact``).
        content: The memory text to store.
        confidence: Optional confidence score.
        sequence_number: Optional ordering sequence within a session.
        metadata: Optional metadata dict persisted on the row.
        recall_hidden: When ``True`` the row is stored but excluded from agent
            recall results (``retrieve_memories_scoped`` and the recency
            fallback). Raw ``observation`` rows written by passive hooks are
            stored recall-hidden so they act as curator fuel without polluting
            the recall surface. Deliberate writes (``note``, ``plan``,
            ``decision``, ``memory``) keep the default ``False``.
        user_id: The human this write is attributed to, or ``None`` when the
            request could not be attributed. On a ``private`` row this is the
            *owner* and the only thing that makes the row readable; on a shared
            row it is *authorship* only and grants nobody anything. The row is
            inserted at ``scope='session'`` here regardless (see
            :func:`shared.brain.write_scoped_memory` for the promotion), so this
            function never creates an owner-less private row.
        provenance: One of ``EXTRACTED`` / ``INFERRED`` / ``AMBIGUOUS``.
            Defaults to ``EXTRACTED`` — this function records what was said or
            done. Only the curator, which deduces, passes ``INFERRED``.

    Returns:
        A :class:`MemoryRecord` on success, or ``None`` if embedding
        generation fails.
    """
    # Availability: return the pooled connection BEFORE awaiting the
    # embedding. See :func:`release_connection` for the measurements.
    release_connection(db_session)
    embedding = await embed_text(content)
    if embedding is None:
        return None

    memory_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    meta = metadata or {}

    def _insert() -> None:
        """The blocking half: one INSERT and a commit.

        Extracted so it can be handed to ``anyio.to_thread.run_sync``. It
        cannot simply be awaited off-loop in place because ``write_memory`` is
        ``async def`` and ``to_thread`` takes a *sync* callable — the body has
        to be a function before it can leave the event loop.
        """
        db_session.execute(
            text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, confidence,
                sequence_number, scope, status, metadata,
                created_at, updated_at, recall_hidden,
                user_id, provenance
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, :execution_id,
                :memory_type, :content, :embedding, :confidence,
                :sequence_number, 'session', 'active', :metadata,
                :created_at, :updated_at, :recall_hidden,
                :user_id, :provenance
            )
        """),
            {
                "id": str(memory_id),
                "session_id": session_id,
                "org_id": org_id,
                "agent_id": agent_id,
                "execution_id": execution_id,
                "memory_type": memory_type,
                "content": content,
                "embedding": str(embedding),
                "confidence": confidence,
                "sequence_number": sequence_number,
                "metadata": json.dumps(meta),
                "created_at": now,
                "updated_at": now,
                "recall_hidden": recall_hidden,
                "user_id": user_id,
                "provenance": provenance,
            },
        )
        db_session.commit()

    # Threading contract (``shared.projects``): psycopg2 is blocking, so it
    # runs off the event loop. Inline, a single ``remember`` held the loop for
    # every one of its statements and throughput plateaued at ~12 rps no matter
    # how much concurrency was offered; offloaded, the same workload reached
    # 126 rps with worst-case loop lag falling from 4092ms to 1.4ms.
    await anyio.to_thread.run_sync(_insert)

    return MemoryRecord(
        id=memory_id,
        session_id=session_id,
        org_id=org_id,
        agent_id=agent_id,
        execution_id=execution_id,
        memory_type=memory_type,
        content=content,
        embedding=embedding,
        confidence=confidence,
        sequence_number=sequence_number,
        metadata=meta,
        created_at=now,
        updated_at=now,
        user_id=user_id,
        provenance=provenance,
    )


@metrics.tracked(metrics.MEMORY_RECALLS, metrics.MEMORY_RECALL_SECONDS)
async def retrieve_memories(
    db_session: Any,
    session_id: str,
    org_id: str,
    query_text: str,
    limit: int = 10,
    include_summaries: int = 5,
) -> list[MemoryRecord]:
    """Retrieve relevant memories via pgvector cosine similarity search.

    Performs two queries:

    1. Semantic search across the session's active session-scoped memories,
       ordered by cosine distance (``embedding <=> :embedding``).
    2. Fetches the last *include_summaries* summaries by
       ``sequence_number`` for recency context.

    Results are deduplicated by ``id`` and returned as a combined list.

    Both queries carry ``scope = 'session' AND recall_hidden = FALSE``, and
    both need them. ``session_id`` is **not** a trust boundary: the
    sync-gateway verification path passes ``event.session_id`` straight off
    the wire, and the reserved ``klio-brain:<org_id>:<agent_id>`` buckets are
    derived from a client-declared ``X-Vex-Agent`` header. Any authenticated
    caller in an org can therefore name any bucket in that org. Without the
    scope predicate a named bucket returns another user's ``scope='private'``
    rows, and without the ``recall_hidden`` predicate it returns curator fuel
    that is deliberately withheld from every recall surface — and whatever
    this function returns is forwarded to the consistency LLM.

    Session-scoped reads are the *only* thing this function serves. Reads at
    ``private``/``agent``/``project``/``org`` scope belong to
    :func:`retrieve_memories_scoped`, which decides entitlement from the
    authenticated principal rather than from a client-supplied string.

    Returns:
        A list of :class:`MemoryRecord` instances, or an empty list if
        embedding generation fails.
    """
    # Availability: return the pooled connection BEFORE awaiting the
    # embedding. See :func:`release_connection` for the measurements.
    release_connection(db_session)
    embedding = await embed_text(query_text)
    if embedding is None:
        return []

    embedding_str = str(embedding)

    # Semantic similarity search
    semantic_result = db_session.execute(
        text("""
            SELECT id, session_id, org_id, agent_id, execution_id,
                   memory_type, content, embedding, confidence,
                   sequence_number, scope, status, superseded_by,
                   metadata, created_at, updated_at
            FROM session_memories
            WHERE session_id = :session_id
              AND org_id = :org_id
              AND status = 'active'
              AND scope = 'session'
              AND recall_hidden = FALSE
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """),
        {
            "session_id": session_id,
            "org_id": org_id,
            "embedding": embedding_str,
            "limit": limit,
        },
    )
    semantic_rows = semantic_result.fetchall()

    # Recent summaries by sequence_number
    summary_result = db_session.execute(
        text("""
            SELECT id, session_id, org_id, agent_id, execution_id,
                   memory_type, content, embedding, confidence,
                   sequence_number, scope, status, superseded_by,
                   metadata, created_at, updated_at
            FROM session_memories
            WHERE session_id = :session_id
              AND org_id = :org_id
              AND memory_type = 'summary'
              AND status = 'active'
              AND scope = 'session'
              AND recall_hidden = FALSE
            ORDER BY sequence_number DESC
            LIMIT :limit
        """),
        {
            "session_id": session_id,
            "org_id": org_id,
            "limit": include_summaries,
        },
    )
    summary_rows = summary_result.fetchall()

    # Deduplicate by id
    seen_ids: set[uuid.UUID] = set()
    records: list[MemoryRecord] = []

    for row in semantic_rows:
        record = _row_to_record(row)
        if record.id not in seen_ids:
            seen_ids.add(record.id)
            records.append(record)

    for row in summary_rows:
        record = _row_to_record(row)
        if record.id not in seen_ids:
            seen_ids.add(record.id)
            records.append(record)

    return records


async def find_similar_memory(
    db_session: Any,
    session_id: str,
    org_id: str,
    content: str,
    threshold: float = DEDUP_THRESHOLD,
    *,
    scope: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[MemoryRecord]:
    """Find the closest existing memory and return it if above *threshold*.

    Similarity is computed as ``1 - (embedding <=> :embedding)`` (cosine
    similarity from cosine distance).

    Dedup window (which existing rows a write is compared against):

        * Default (``scope=None``) — session-scoped: rows sharing
          ``session_id`` within the org. This is the historical behavior used
          by the verification and org-brain write paths; unchanged.
        * ``scope='agent'`` (with ``agent_id``) — agent-private: rows where
          ``scope='agent' AND agent_id=:agent_id`` within the org, IGNORING
          ``session_id``. This makes an agent-private write dedup only against
          the *same agent's* private rows — never against the org brain, and
          never against another agent's private rows. (Required so per-agent
          isolation holds at write time, not just recall time.)
        * ``scope='private'`` (with ``user_id``) — user-private: rows where
          ``scope='private' AND user_id=:user_id`` within the org, IGNORING
          ``session_id``.

          This window is a **privacy** control, not just a correctness one. A
          dedup hit returns the matching row's id and memory_type to the
          caller, so deduping a private write against the whole org would
          disclose that another person already stored near-identical text, and
          hand back their row id. The predicate is therefore the same
          ``user_id = :user_id`` that guards recall. ``user_id`` is mandatory
          here: without an owner there is no window that can be safely
          searched, so the call raises rather than silently widening.

    Raises:
        ValueError: If ``scope='private'`` is requested without a ``user_id``.

    Returns:
        The matching :class:`MemoryRecord` if similarity >= *threshold*,
        otherwise ``None``.
    """
    # Validate BEFORE the (expensive) embed call, and before any query could
    # be built without the owner predicate.
    if scope == "private" and not user_id:
        raise ValueError("user_id is required when scope='private'")

    # Availability: return the pooled connection BEFORE awaiting the
    # embedding. See :func:`release_connection` for the measurements.
    release_connection(db_session)
    embedding = await embed_text(content)
    if embedding is None:
        return None

    embedding_str = str(embedding)

    if scope == "private":
        # User-private dedup: compare only against this user's own private
        # rows (org-wide, not session-scoped). See the docstring for why this
        # predicate is a privacy control.
        where_clause = (
            "org_id = :org_id\n"
            "              AND scope = 'private'\n"
            "              AND user_id = :user_id\n"
            "              AND status = 'active'"
        )
        params: dict[str, Any] = {
            "org_id": org_id,
            "user_id": user_id,
            "embedding": embedding_str,
        }
    elif scope == "agent":
        # Agent-private dedup: compare only against this agent's own private
        # rows (org-wide, not session-scoped). agent_id must be supplied.
        where_clause = (
            "org_id = :org_id\n"
            "              AND scope = 'agent'\n"
            "              AND agent_id = :agent_id\n"
            "              AND status = 'active'"
        )
        params = {
            "org_id": org_id,
            "agent_id": agent_id,
            "embedding": embedding_str,
        }
    else:
        # Default session-scoped dedup (historical behavior; unchanged).
        where_clause = (
            "session_id = :session_id\n"
            "              AND org_id = :org_id\n"
            "              AND status = 'active'"
        )
        params = {
            "session_id": session_id,
            "org_id": org_id,
            "embedding": embedding_str,
        }

    def _nearest():
        """The blocking half: one pgvector nearest-neighbour scan.

        Extracted for ``anyio.to_thread.run_sync`` — see the note in
        :func:`write_memory`. ``to_thread`` takes a sync callable, so the body
        has to be a function before it can leave the event loop.
        """
        return db_session.execute(
            text(f"""
            SELECT id, session_id, org_id, agent_id, execution_id,
                   memory_type, content, embedding, confidence,
                   sequence_number, scope, status, superseded_by,
                   metadata, created_at, updated_at,
                   1 - (embedding <=> :embedding) AS similarity
            FROM session_memories
            WHERE {where_clause}
            ORDER BY embedding <=> :embedding
            LIMIT 1
        """),
            params,
        ).fetchone()

    # Threading contract (``shared.projects``): a vector scan is blocking
    # psycopg2 I/O and must not run on the event loop.
    row = await anyio.to_thread.run_sync(_nearest)
    if row is None:
        return None

    # The similarity score is the last column (index 16)
    similarity = row[-1]
    if similarity >= threshold:
        return _row_to_record(row[:16])

    return None


def supersede_memory(
    db_session: Any,
    old_memory_id: uuid.UUID,
    new_memory_id: uuid.UUID,
    *,
    org_id: str,
) -> None:
    """Mark an old memory as superseded by a newer one, within one tenant.

    Sets ``status = 'superseded'`` and ``superseded_by = new_memory_id``
    on the old memory row.

    ``org_id`` is mandatory. This was the one write in the memory layer whose
    WHERE named only an ``id``, which made it the only one whose correctness
    depended entirely on where its caller got that id. Both current callers
    (the curator's reconcile pass and the async worker's fact writer) derive
    theirs from org-scoped queries, so nothing was reachable through it — but
    "every caller so far has been careful" is not a boundary, and a keyword
    with no default means the next caller cannot omit it by accident.

    A mismatched ``org_id`` updates nothing and raises nothing: the row is not
    this tenant's, which is the same non-event as an id that does not exist.

    After the status flip commits, retroactively marks any past recall
    serves of the old memory stale (:func:`shared.stale.mark_stale_serves`)
    — but only the ones served at or after the moment the superseding
    belief actually came into being. Serves before it were correct at the
    time. That moment is the *new* row's real ``created_at``, fetched with a
    single org-predicated primary-key lookup (we already have its id — this
    is not the "re-query to find the row" the brief warned against, it's an
    indexed read of one column). Using ``now()`` at call time instead would
    systematically run late: the worker inserts-and-commits the new row
    before calling this, and the curator embeds, vector-searches, and runs a
    per-candidate LLM judge first — seconds of the exact curator-lag window
    this counter exists to catch would go under-marked. The lookup is
    best-effort: if it fails or the row is somehow gone, this falls back to
    ``now()`` rather than block or fail the supersession.
    """
    flip_at = datetime.now(timezone.utc)
    db_session.execute(
        text("""
            UPDATE session_memories
            SET status = 'superseded',
                superseded_by = :new_memory_id,
                updated_at = :flip_at
            WHERE id = :old_memory_id
              AND org_id = :org_id
        """),
        {
            "old_memory_id": str(old_memory_id),
            "new_memory_id": str(new_memory_id),
            "org_id": org_id,
            "flip_at": flip_at,
        },
    )
    db_session.commit()

    superseding_created_at = flip_at
    try:
        row = db_session.execute(
            text("""
                SELECT created_at FROM session_memories
                WHERE id = :new_memory_id
                  AND org_id = :org_id
            """),
            {"new_memory_id": str(new_memory_id), "org_id": org_id},
        ).fetchone()
        if row is not None and row[0] is not None:
            superseding_created_at = row[0]
    except Exception:
        try:
            db_session.rollback()
        except Exception:
            pass
        logger.debug(
            "supersede_memory: could not read new row's created_at "
            "(new=%s org=%s); falling back to flip time",
            new_memory_id, org_id, exc_info=True,
        )

    stale.mark_stale_serves(
        db_session,
        org_id=org_id,
        old_memory_id=str(old_memory_id),
        superseding_created_at=superseding_created_at,
    )


# ---------------------------------------------------------------------------
# Optimistic concurrency (CAS writes) — Build 3, Task 8
# ---------------------------------------------------------------------------


def record_write_conflict() -> None:
    """Record one CAS write rejected because its ``base_revision`` was stale.

    Follows the ``record_*`` instrumentation-helper pattern in
    ``app.curator`` (:func:`app.curator.record_tick` et al.): one line per
    call site, and the metrics fault itself can never propagate — it goes
    through :func:`shared.metrics.record`, which swallows.
    """
    metrics.record(metrics.WRITE_CONFLICTS.inc)


def _embed_text_sync(text_input: str) -> Optional[list[float]]:
    """Synchronous embedding, for the one write path that cannot ``await``.

    :func:`supersede_with_revision` is the server side of Mukta's
    optimistic-concurrency loop, and its signature is deliberately a plain
    ``def``: the guard is a single atomic ``UPDATE``, and the function has to
    be callable from a worker thread handed off via
    ``anyio.to_thread.run_sync`` (the house pattern every blocking DB call in
    this codebase already uses), where there is no running event loop to
    ``await`` against. :func:`embed_text` is unusable here for exactly that
    reason — it is ``async def`` and built on ``litellm.aembedding``.

    This mirrors :func:`embed_text`'s model/api_base/api_key resolution and
    its first-attempt char-budget truncation, but deliberately skips its
    retry/timeout/concurrency-cap/memoization machinery: CAS supersedes run at
    explicit-correction volume, not steady-state write volume, and each call
    already executes inside its own dedicated worker thread, so a slow embed
    here blocks only that one CAS write rather than the event loop every other
    write path shares.

    Returns:
        The embedding vector, or ``None`` if the backend call fails.
    """
    if text_input and len(text_input) > MAX_EMBED_CHARS:
        candidate = text_input[:MAX_EMBED_CHARS]
    else:
        candidate = text_input

    model = _embed_model()
    api_base = os.environ.get("LITELLM_API_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    kwargs: dict[str, Any] = {"model": model, "input": [candidate]}
    if api_base:
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "openai"
    if api_key:
        kwargs["api_key"] = api_key

    try:
        response = litellm.embedding(**kwargs)
    except Exception:
        logger.warning(
            "CAS supersede: synchronous embedding call failed", exc_info=True
        )
        return None

    item = response.data[0]
    return item["embedding"] if isinstance(item, dict) else item.embedding


#: Hard ceiling on how many hops :func:`_current_belief_in_chain` will walk.
#: A legitimate ``superseded_by`` chain is a handful of hops at most; this
#: exists only so a corrupted or (impossible, but never trust that) cyclic
#: chain fails fast instead of making the database walk it forever.
_CAS_CHAIN_MAX_DEPTH = 50


def _authorize_scoped_action(
    db_session: Any,
    row: Any,
    *,
    org_id: str,
    user_id: Optional[str],
    agent_id: Optional[str],
) -> Optional[str]:
    """The shared scope-authorization ladder for acting on an existing row.

    ``row`` must expose ``scope`` / ``agent_id`` / ``user_id`` / ``project_id``
    via attribute access (a plain ``SELECT`` of those four columns). Returns
    ``None`` when the caller may act on the row, otherwise the refusal code:

    * ``"not_found"`` — ``scope='private'`` owned by someone else, or
      ``scope='project'`` in a project the caller is not a member of.
      Indistinguishable from an id that was never issued: confirming the row
      EXISTS would itself be the disclosure a private/project scope promises
      not to make.
    * ``"forbidden"`` — ``scope='agent'`` owned by a different agent. The
      row's existence is not in question here, only the caller's standing.

    Extracted from :func:`retract_memory`'s original inline ladder (both now
    call this) so the two authorization surfaces — retracting a memory and
    CAS-superseding one — cannot silently drift apart. Same four checks, same
    order, same refusal shape as before the extraction; see
    :func:`retract_memory`'s docstring for why each answers the way it does.
    """
    if row.scope == "private" and (not user_id or row.user_id != user_id):
        return "not_found"
    if row.scope == "project" and not (
        row.project_id
        and projects.is_member(
            db_session,
            org_id=org_id,
            project_id=str(row.project_id),
            user_id=user_id,
        )
    ):
        return "not_found"
    if row.scope == "agent" and row.agent_id != agent_id:
        return "forbidden"
    return None


def _current_belief_in_chain(
    db_session: Any,
    *,
    org_id: str,
    memory_id: str,
    user_id: Optional[str],
    agent_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Walk ``memory_id``'s ``superseded_by`` chain forward to the live belief.

    A superseded row's OWN ``revision`` never changes again once it stops
    being current — the CAS guard bumps it once, at the moment it is
    dethroned, and nothing touches it after. So a caller holding a stale copy
    of a row that has SINCE been superseded can still present a
    ``base_revision`` that matches the dead row's current value; the guard
    itself is what refuses that (``AND status = 'active' AND superseded_by IS
    NULL``). This function is the other half of that fix: once the guard has
    missed, the honest answer to "what is current" is whatever sits at the
    END of the supersede chain, not the (possibly long-dead) row named by
    ``memory_id``. A recursive CTE walks the chain forward, org-predicated at
    every hop, capped at :data:`_CAS_CHAIN_MAX_DEPTH` so a corrupted chain
    cannot spin the database instead of failing fast.

    N1: the chain can cross scope. An org-scoped row can be superseded by a
    ``private`` or ``project``-scoped successor (the caller supplies the new
    row's scope on every CAS write, and nothing forces it to match the row
    being replaced), so the terminal row is NOT covered by the authorization
    check :func:`supersede_with_revision` already ran against the row NAMED
    by ``memory_id`` — that check happened one hop back. Returning the
    terminal row's content unconditionally would hand a non-member the
    content, ``user_id`` and live ``revision`` of a row they cannot read,
    through a conflict payload rather than a `recall` — the exact class of
    leak H1 closed one layer up. So the SAME ladder
    (:func:`_authorize_scoped_action`) runs again here, against the terminal
    row specifically. A refusal here returns ``None`` — not a distinct error
    code, since :func:`supersede_with_revision` already established the
    caller may act on ``memory_id`` itself; ``current: None`` just means "the
    live belief is not something you're entitled to see," and the caller's
    own retry with a corrected ``base_revision`` will re-run the WHOLE ladder
    from scratch against whatever they name next.

    Returns ``None`` if ``memory_id`` no longer resolves in this org at all
    (retracted out from under a concurrent embed, or a caller who raced past
    the earlier existence check — vanishingly rare, and this is the same
    non-event an unknown id already is everywhere else in this module) OR if
    the terminal row exists but the caller may not see it (N1, above).
    """
    row = (
        db_session.execute(
            text(
                """
                WITH RECURSIVE chain AS (
                    SELECT id, content, revision, user_id, agent_id, scope,
                           project_id, superseded_by, 0 AS depth
                    FROM session_memories
                    WHERE id = :memory_id AND org_id = :org_id
                    UNION ALL
                    SELECT m.id, m.content, m.revision, m.user_id,
                           m.agent_id, m.scope, m.project_id,
                           m.superseded_by, c.depth + 1
                    FROM session_memories m
                    JOIN chain c ON m.id = c.superseded_by
                    WHERE m.org_id = :org_id AND c.depth < :max_depth
                )
                SELECT id, content, revision, user_id, agent_id, scope,
                       project_id
                FROM chain
                WHERE superseded_by IS NULL
                ORDER BY depth DESC
                LIMIT 1
                """
            ),
            {
                "memory_id": memory_id,
                "org_id": org_id,
                "max_depth": _CAS_CHAIN_MAX_DEPTH,
            },
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    refusal = _authorize_scoped_action(
        db_session, row, org_id=org_id, user_id=user_id, agent_id=agent_id
    )
    if refusal is not None:
        return None
    return _normalize_current_row(row)


def _prepare_cas_write(
    new_content: str,
    *,
    metadata: Optional[dict],
    scope: str,
    project_id: Optional[str],
    user_id: Optional[str],
) -> tuple[str, dict, Optional[list[float]], bool, list[str]]:
    """Redact and embed the re-asserted belief — BEFORE the CAS guard runs.

    H3: the guard is a single ``UPDATE`` that takes a row-level lock and
    holds a pooled connection until commit. A network call (the embedding
    request) sitting between that ``UPDATE`` and its commit starves the pool
    under concurrency and is worse than merely slow against Neon's pooled
    (PgBouncer transaction-mode) endpoint, where a connection held open
    across an unrelated round trip blocks every other statement sharing that
    backend. So everything that can happen WITHOUT touching the guard's
    transaction — scope validation, redaction, embedding — happens here,
    called strictly before :func:`supersede_with_revision` issues the guard.
    No network call is ever between the guard and its commit.

    Redaction mirrors :func:`shared.brain.write_scoped_memory`'s fail-closed
    contract: a redactor crash rejects the write rather than storing
    unscreened content. ``shared.brain.screen_metadata`` is imported lazily
    (not at module scope) because ``shared.brain`` imports from THIS module —
    a top-level import here would be circular, the same reason
    ``_expand_via_graph`` imports ``shared.graph_store`` lazily.

    Returns:
        ``(content, metadata, embedding, redacted, redaction_kinds)``.
        ``embedding`` is ``None`` only when the backend call failed — the
        caller decides how to respond, this function does not raise for it.

    Raises:
        ValueError: unsupported ``scope``, a ``scope='private'`` write with
            no ``user_id``, or a ``scope='project'`` write with no
            ``project_id`` — the same validation
            :func:`shared.brain.write_scoped_memory` applies.
        Exception: whatever the redactor / metadata screener raise, re-raised
            after logging (fail closed — unscreened content is never stored).
    """
    if scope not in ("org", "project", "private", "agent"):
        raise ValueError(
            f"unsupported scope: {scope!r} (supersede_with_revision accepts "
            "'org', 'project', 'private' or 'agent')"
        )
    if scope == "private" and not user_id:
        raise ValueError("user_id is required when scope='private'")
    if scope == "project" and not project_id:
        raise ValueError("project_id is required when scope='project'")

    from shared.brain import screen_metadata

    content = new_content
    meta = dict(metadata or {})
    redacted = False
    redaction_kinds: list[str] = []
    if os.environ.get("MEMORY_REDACTION_DISABLED") != "1":
        try:
            redaction = redact(content)
            meta, metadata_kinds = screen_metadata(meta)
        except Exception:
            logger.exception(
                "CAS supersede: redaction failed; rejecting write (fail closed)"
            )
            raise
        if redaction.redacted or metadata_kinds:
            if redaction.redacted:
                content = redaction.clean_text
            redaction_kinds = sorted(
                {f.kind for f in redaction.findings} | set(metadata_kinds)
            )
            redacted = True
            meta = dict(meta or {})
            meta["redactions"] = redaction_kinds

    embedding = _embed_text_sync(content)
    return content, meta, embedding, redacted, redaction_kinds


def _apply_supersession_for_cas(
    db_session: Any,
    *,
    org_id: str,
    old_id: str,
    content: str,
    embedding: list[float],
    metadata: dict,
    agent_id: str,
    memory_type: str,
    session_id: str,
    scope: str = "org",
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    recall_hidden: bool = False,
    user_id: Optional[str] = None,
    provenance: str = "EXTRACTED",
) -> dict[str, Any]:
    """Insert the pre-redacted, pre-embedded belief and flip the old row.

    Called ONLY after :func:`supersede_with_revision`'s guard ``UPDATE`` has
    already confirmed (by rowcount) that ``base_revision`` was current, on
    content that :func:`_prepare_cas_write` already redacted and embedded
    BEFORE the guard ran (H3 — see that function). Everything this function
    itself does is a local DB statement — no network call, so nothing here
    can hold the guard's row lock open across a round trip to a third party.

    This function's own job is narrow: insert the new row at its final
    scope, then delegate the status flip and stale-serve marking to
    :func:`supersede_memory` VERBATIM — changing none of its semantics. That
    is also why the old row's status flip is not duplicated here: it happens
    exactly once, inside ``supersede_memory``.

    Deliberately NOT a thin wrapper around :func:`shared.brain.
    write_scoped_memory`, which is ``async def`` (embeds via ``embed_text``)
    for the same reason :func:`_embed_text_sync` exists. The insert below
    writes the row directly at its requested scope (rather than
    write-then-promote) because there is no session-scoped intermediate state
    to preserve here — the row is born already superseding something.

    Returns:
        ``{"new_id": ..., "new_revision": ...}`` — the freshly-inserted row's
        id and its own CAS token (read via ``INSERT ... RETURNING``, not a
        follow-up ``SELECT``), both stringified.
    """
    new_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    inserted = db_session.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, confidence,
                sequence_number, scope, status, metadata,
                created_at, updated_at, recall_hidden,
                user_id, provenance, space_id, project_id
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                :memory_type, :content, :embedding, NULL,
                NULL, :scope, 'active', :metadata,
                :created_at, :updated_at, :recall_hidden,
                :user_id, :provenance, :space_id, :project_id
            )
            RETURNING revision
            """
        ),
        {
            "id": str(new_id),
            "session_id": session_id,
            "org_id": org_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": str(embedding),
            "scope": scope,
            "metadata": json.dumps(metadata),
            "created_at": now,
            "updated_at": now,
            "recall_hidden": recall_hidden,
            "user_id": user_id,
            "provenance": provenance,
            "space_id": space_id,
            "project_id": project_id,
        },
    )
    returned = inserted.fetchone()
    new_revision = str(returned[0]) if returned is not None else None

    # The status flip, superseded_by link, and stale-serve marking — verbatim,
    # unchanged. supersede_memory commits.
    supersede_memory(db_session, uuid.UUID(str(old_id)), new_id, org_id=org_id)

    return {"new_id": str(new_id), "new_revision": new_revision}


def _normalize_current_row(current: Any) -> Optional[dict[str, Any]]:
    """Stringify a raw ``current`` row for the conflict payload.

    Real Postgres hands back ``uuid.UUID`` objects for ``id``/``revision``/
    ``user_id`` (an ``sqlalchemy.RowMapping``, not JSON-serializable as-is);
    the unit tests in ``test_cas_write.py`` hand back a plain ``dict`` of
    strings already. ``str()`` is a no-op on a string and the standard
    stringification on a UUID, so one code path covers both without a type
    check — the same trick ``id``/``space_id``/``project_id`` use everywhere
    else a row crosses into a JSON response (see ``_serialize_record``).
    """
    if current is None:
        return None
    return {
        "id": str(current["id"]),
        "content": current["content"],
        "revision": str(current["revision"]),
        "user_id": (
            str(current["user_id"]) if current["user_id"] is not None else None
        ),
    }


def supersede_with_revision(
    db: Any,
    *,
    org_id: str,
    memory_id: str,
    base_revision: str,
    new_content: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **write_kwargs: Any,
) -> dict[str, Any]:
    """Mukta's optimistic-concurrency loop, server side.

    Four phases, strictly in this order:

    1. **Validate** (N3) — an unparseable ``memory_id`` OR ``base_revision``
       (both UUID-shaped columns) is refused before any query, mirroring
       :func:`retract_memory`'s ``invalid_id``. Catching a bad
       ``base_revision`` here also prevents it reaching the guard's
       ``revision = :base_revision`` comparison against a UUID column, which
       would otherwise raise an uncaught driver error instead of a clean
       refusal.
    2. **Authorize** (H1) — the SAME scope ladder :func:`retract_memory` uses
       (:func:`_authorize_scoped_action`), run against ``memory_id`` before
       ANYTHING else happens. Without this, any org principal could name a
       private row's id, read its content+revision back out of the conflict
       payload, and then use that revision to actually supersede (destroy)
       it — a read primitive and a write primitive hiding inside what looks
       like a single guarded write. An unauthorized or nonexistent id gets
       the exact same refusal :func:`retract_memory` gives: ``not_found`` for
       a private/project row the caller cannot see (never ``forbidden`` —
       that would itself confirm the row exists), ``forbidden`` only for the
       agent-ownership mismatch, where existence was never secret. This
       phase's own read is closed out with a rollback (N2) before phase 3 —
       nothing was written, and holding that transaction (hence the pooled
       connection) open across phase 3's network call is exactly the
       pin-a-backend-across-an-embed problem phase 3 exists to avoid; see
       phase 3 for why that matters under a transaction-mode pooler.
    3. **Prepare** (H3) — redact and embed ``new_content``
       (:func:`_prepare_cas_write`) BEFORE the guard runs, on a connection
       that phase 2 already released (no open transaction, per N2 above).
       The embed is a network call; the guard is a row lock. Doing the
       network call first, with nothing else pinning the connection, means
       no network call ever sits inside an open transaction on this
       connection — which is what keeps a slow embed from holding a pooled
       connection checked out (fatal against Neon's PgBouncer
       transaction-mode endpoint, where a connection held across an
       unrelated round trip blocks every other statement sharing that
       backend) for longer than the embed itself needs.
    4. **Guard, then apply** (H2) — the atomic ``UPDATE`` bumps ``revision``
       ONLY if ``base_revision`` is still current AND the row is still
       ``status = 'active'`` AND ``superseded_by IS NULL``. The status/
       superseded_by predicates close a hole the revision check alone does
       not: a superseded row's OWN ``revision`` never changes again once it
       stops being current, so a caller holding a stale copy of an
       ALREADY-superseded row could otherwise still pass the guard. Zero rows
       means someone moved the belief (or it is dead) — the conflict payload
       follows the ``superseded_by`` chain to the actual live belief
       (:func:`_current_belief_in_chain`), which re-runs the SAME
       authorization ladder against whatever sits at the end of that chain
       (N1: the chain can cross scope, so the terminal row is not covered by
       phase 2's check on ``memory_id`` — see that function), rather than
       echoing back the dead row's own (frozen) content or an unauthorized
       terminal row's. A match proceeds through the existing supersession
       machinery (:func:`_apply_supersession_for_cas`) unchanged.

    Under real Postgres phase 4 is race-safe with no extra locking on the
    caller's part: the ``UPDATE ... WHERE revision = :base_revision`` takes a
    row-level lock, so a second writer racing on the SAME row blocks until the
    first commits or rolls back, then re-evaluates its own ``WHERE`` against
    the now-current ``revision`` — which the first writer just changed. Only
    one of two concurrent callers with the same stale ``base_revision`` can
    ever see ``rowcount == 1``.

    Deliberately a plain ``def``, not ``async def``: callers invoke this from
    a worker thread (``anyio.to_thread.run_sync``), matching the house
    pattern for every other blocking DB write in this module
    (:func:`supersede_memory`, :func:`retract_memory`). See
    :func:`_embed_text_sync` for why phase 3 cannot use the async embedding
    helper.

    Args:
        db: A synchronous SQLAlchemy session/connection.
        org_id: Owning org id (tenant scope). Always enforced.
        memory_id: The row being re-asserted.
        base_revision: The ``revision`` the caller last read. Stale → conflict.
        new_content: The re-asserted belief's text.
        user_id: The caller's resolved identity — the SAME thing
            :func:`retract_memory` takes, used identically: it is the owner
            check on a ``private`` row and the membership check on a
            ``project`` row, and it is forwarded to the new row as authorship.
        agent_id: The caller's agent principal — the ownership check on an
            ``agent``-scoped row, and forwarded to the new row as author.
        **write_kwargs: Forwarded to :func:`_apply_supersession_for_cas` on
            the success path (``memory_type``, ``session_id``, and the
            optional ``scope`` / ``space_id`` / ``project_id`` / ``metadata``
            / ``recall_hidden`` / ``provenance``). ``metadata`` here is the
            RAW value — the redacted one built in phase 3 is what actually
            gets forwarded.

    Returns:
        Refused before any write is attempted (``stored`` is always
        ``False``): ``{"conflict": False, "stored": False, "error":
        "invalid_id" | "not_found" | "forbidden" | "redaction_failed" |
        "embedding_failed"}``.

        On conflict: ``{"conflict": True, "stored": False, "current": {"id",
        "content", "revision", "user_id"} | None}`` — ``current`` is ``None``
        only if the row vanished from this org entirely between the
        authorization check and the guard (a race with a delete, not a normal
        outcome).

        On success: ``{"conflict": False, "stored": True, "redacted": bool,
        "redaction_kinds": [...], "new_id": ..., "new_revision": ...}`` — the
        same ``redacted``/``redaction_kinds`` shape
        :func:`shared.brain.write_scoped_memory` returns on its success path.
    """
    # N3: both ids are UUID-shaped columns. A malformed memory_id fails the
    # existence lookup below anyway, but a malformed base_revision would
    # reach the guard's ``revision = :base_revision`` comparison against a
    # UUID column and raise an uncaught driver error instead of a clean
    # refusal — validate both up front, before any DB round trip or spend.
    try:
        parsed_id = uuid.UUID(str(memory_id))
    except (ValueError, AttributeError, TypeError):
        return {"conflict": False, "stored": False, "error": "invalid_id"}
    memory_id = str(parsed_id)
    try:
        uuid.UUID(str(base_revision))
    except (ValueError, AttributeError, TypeError):
        return {"conflict": False, "stored": False, "error": "invalid_id"}

    # ---- phase 2: authorize (H1) ------------------------------------------
    row = db.execute(
        text(
            """
            SELECT scope, agent_id, user_id, project_id
            FROM session_memories
            WHERE id = :memory_id AND org_id = :org_id
            LIMIT 1
            """
        ),
        {"memory_id": memory_id, "org_id": org_id},
    ).fetchone()
    if row is None:
        return {"conflict": False, "stored": False, "error": "not_found"}
    refusal = _authorize_scoped_action(
        db, row, org_id=org_id, user_id=user_id, agent_id=agent_id
    )
    if refusal is not None:
        return {"conflict": False, "stored": False, "error": refusal}

    # N2: the authz SELECT above opened a transaction on this pooled
    # connection (SQLAlchemy 2.0 autobegins on first execute). Nothing was
    # written, so a plain rollback closes it out — otherwise this backend
    # stays checked out across phase 3's network call, and under a
    # transaction-mode pooler (Neon/PgBouncer) a connection held open across
    # an unrelated round trip blocks every other statement sharing it. The
    # guard below re-reads status/superseded_by/revision itself, so nothing
    # after this point depends on the rolled-back transaction's snapshot.
    db.rollback()

    # ---- phase 3: prepare — redact + embed, BEFORE the guard (H3) --------
    try:
        content, meta, embedding, redacted, redaction_kinds = _prepare_cas_write(
            new_content,
            metadata=write_kwargs.get("metadata"),
            scope=write_kwargs.get("scope", "org"),
            project_id=write_kwargs.get("project_id"),
            user_id=user_id,
        )
    except Exception:
        metrics.record(metrics.MEMORY_WRITES.inc, outcome="error")
        return {"conflict": False, "stored": False, "error": "redaction_failed"}

    if embedding is None:
        metrics.record(metrics.MEMORY_WRITES.inc, outcome="error")
        return {"conflict": False, "stored": False, "error": "embedding_failed"}

    # ---- phase 4: guard (H2), then apply -----------------------------------
    guard = db.execute(
        text(
            """
            UPDATE session_memories
            SET revision = gen_random_uuid()
            WHERE id = :memory_id AND org_id = :org_id
              AND revision = :base_revision
              AND status = 'active'
              AND superseded_by IS NULL
            """
        ),
        {
            "memory_id": memory_id,
            "org_id": org_id,
            "base_revision": base_revision,
        },
    )
    if not guard.rowcount:
        record_write_conflict()
        current = _current_belief_in_chain(
            db, org_id=org_id, memory_id=memory_id,
            user_id=user_id, agent_id=agent_id,
        )
        return {"conflict": True, "stored": False, "current": current}

    forwarded = dict(write_kwargs)
    forwarded.pop("metadata", None)  # the redacted `meta` replaces it below
    result = _apply_supersession_for_cas(
        db,
        org_id=org_id,
        old_id=memory_id,
        content=content,
        embedding=embedding,
        metadata=meta,
        agent_id=agent_id,
        user_id=user_id,
        **forwarded,
    )
    metrics.record(metrics.MEMORY_WRITES.inc, outcome="ok")
    return {
        "conflict": False,
        "stored": True,
        "redacted": redacted,
        "redaction_kinds": redaction_kinds,
        **result,
    }


def retract_memory(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    memory_id: str,
    via: str,
    user_id: Optional[str] = None,
) -> dict:
    """Explicitly retract a memory (the ``forget`` tool's storage primitive).

    Authorization ladder (org_id always enforced):
      * unknown / cross-org id → not_found (indistinguishable by design)
      * scope='private' row owned by another user → **not_found**
      * scope='project' row in a project the caller is not a member of →
        **not_found**
      * scope='agent' row owned by another agent → forbidden
      * otherwise → status='retracted' + audit metadata; recall never
        returns non-'active' rows, so the memory disappears immediately.

    The project arm answers ``not_found`` rather than ``forbidden`` for the
    same reason the private arm does, and it must: membership is precisely what
    hides a project row's existence, so an answer that confirmed the id
    resolves to something real would disclose what the boundary exists to
    withhold. It is **strict, with no org-admin override**, matching
    ``project_grant``.

    Only ``scope='project'`` is fenced. A ``project_id`` on an ``org``-scoped
    row RANKS it, it does not fence it — ``_explicit_write_scope`` deliberately
    files a non-member's write that way, and reading membership off
    ``project_id`` alone would make those rows unforgettable by the very people
    who wrote them. A ``scope='project'`` row with a NULL ``project_id`` (no
    constraint ties the two columns) fails closed: :func:`shared.projects.is_member`
    cannot vouch for a project that is not named, so neither can this.

    Why a private row answers ``not_found`` and not ``forbidden``: ``forbidden``
    confirms the row exists. Someone who guessed or was shown a memory id would
    learn that this org holds a private memory under it, which is a disclosure
    about a scope whose whole promise is opacity from the outside. A
    non-owner's view of another person's private memory is identical to the
    view of an id that was never issued.

    ANY status is retractable — deliberately. This used to require
    ``status = 'active'``, which made a memory the curator had auto-superseded
    permanently unforgettable: the owner got the indistinguishable
    ``not_found``, believed it was gone, and the knowledge graph went on
    serving its content back to them as ``superseded[].old_content`` forever.
    The status filter never granted anything either — the four checks below
    (org, existence, private-owner, agent-owner) are the whole authorization
    ladder, and none of them reads ``status`` — so removing it widens what can
    be FORGOTTEN without widening what can be SEEN. Retracting an
    already-retracted row is a harmless idempotent no-op.

    Synchronous (inline psycopg2 I/O), matching the house pattern.
    """
    try:
        parsed = uuid.UUID(memory_id)
    except (ValueError, AttributeError, TypeError):
        return {"forgotten": False, "error": "invalid_id"}

    row = db_session.execute(
        text(
            """
            SELECT scope, agent_id, user_id, project_id FROM session_memories
            WHERE id = :id AND org_id = :org_id
            LIMIT 1
            """
        ),
        {"id": str(parsed), "org_id": org_id},
    ).fetchone()
    if row is None:
        return {"forgotten": False, "error": "not_found"}
    refusal = _authorize_scoped_action(
        db_session, row, org_id=org_id, user_id=user_id, agent_id=agent_id
    )
    if refusal is not None:
        return {"forgotten": False, "error": refusal}

    db_session.execute(
        text(
            """
            UPDATE session_memories
            SET status = 'retracted',
                metadata = COALESCE(metadata, '{}'::jsonb)
                           || jsonb_build_object(
                               'retracted_by', CAST(:agent_id AS text),
                               'retracted_at', NOW()::text,
                               'retracted_via', CAST(:via AS text)
                           ),
                updated_at = NOW()
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"id": str(parsed), "org_id": org_id, "agent_id": agent_id, "via": via},
    )
    db_session.commit()
    return {"forgotten": True, "id": str(parsed)}


def _recency_norm(created_at: Optional[datetime]) -> float:
    """Fraction of the 30-day window remaining, clamped to [0, 1].

    The Python twin of the SQL recency term in the stage-A ORDER BY, used by
    the stage-C re-rank (which has to blend rows from two queries and so
    cannot stay in one statement).
    """
    if created_at is None:
        return 0.0
    age = (datetime.now(timezone.utc) - created_at).total_seconds()
    return min(1.0, max(0.0, 1.0 - age / _RECENCY_WINDOW_SECONDS))


def _expand_via_graph(
    db_session: Any,
    records: list[MemoryRecord],
    *,
    org_id: str,
    scope: str,
    agent_id: Optional[str],
    user_id: Optional[str],
    space_id: Optional[str],
    project_id: Optional[str],
) -> list[MemoryRecord]:
    """Recall stage B+C: one-hop graph expansion, then the blended re-rank.

    Stage B calls :func:`shared.graph_store.neighbors` ONCE over the stage-A
    entry ids. Every argument that shapes visibility — scope, viewer, agent,
    space/project filters — is forwarded verbatim, so the expansion evaluates
    the SAME ladder in SQL and can never introduce a row the vector stage
    would have been forbidden to return. Volume is bounded by
    :data:`GRAPH_EXPANSION_CAP` extra rows.

    Stage C re-ranks the merged set by ``0.70·similarity + 0.15·recency +
    0.15·connectedness`` (connectedness = edge count into the entry set,
    normalized by the batch max). Expanded rows keep ``relevance=None`` — they
    did not earn a similarity score and faking one would misrepresent why
    they are present — and contribute 0 to the similarity term. They are
    instead labelled ``via=VIA_GRAPH``: a POSITIVE marker, because a null
    ``relevance`` is erased by structural compression's null-key strip and
    left nothing on the wire to tell a semantic match from a graph walk.

    Failure containment: the graph is an enhancement, never a dependency. Any
    error here logs and returns the stage-A records untouched — after rolling
    the session back. The rollback is not tidiness: a genuine SQL failure in
    the expansion aborts the transaction, and every later statement on the
    session then raises ``InFailedSqlTransaction``. Recall itself survives
    (its rows are already fetched), but the recall audit/billing event that
    follows swallows its own errors, so the run would silently write ZERO
    audit rows. Healing the session here is what keeps that from happening.
    """
    if not records:
        return records

    # Imported lazily: graph_store imports MemoryRecord/_row_to_record from
    # THIS module, so a top-level import would be circular. This function is
    # the single composition point between the two.
    from shared import graph_store

    try:
        expansion = graph_store.neighbors(
            db_session,
            org_id=org_id,
            viewer_user_id=user_id,
            memory_ids=[str(r.id) for r in records],
            scope=scope,
            agent_id=agent_id,
            space_id=space_id,
            project_id=project_id,
            limit=GRAPH_EXPANSION_CAP,
        )
    except Exception:
        logger.warning(
            "recall graph expansion failed; returning vector results alone",
            exc_info=True,
        )
        try:
            db_session.rollback()
        except Exception:
            logger.warning(
                "rollback after a failed graph expansion also failed",
                exc_info=True,
            )
        return records

    entry_ids = {str(r.id) for r in records}
    extra = [
        replace(r, via=VIA_GRAPH)
        for r in expansion["records"]
        if str(r.id) not in entry_ids
    ]
    if not extra:
        return records

    connections: dict[str, int] = expansion["connections"]
    max_connections = max(connections.values(), default=0) or 1

    def _blended(record: MemoryRecord) -> float:
        similarity = record.relevance if record.relevance is not None else 0.0
        connectedness = connections.get(str(record.id), 0) / max_connections
        return (
            _RERANK_SIMILARITY_WEIGHT * similarity
            + _RERANK_RECENCY_WEIGHT * _recency_norm(record.created_at)
            + _RERANK_CONNECTEDNESS_WEIGHT * connectedness
        )

    return sorted(records + extra, key=_blended, reverse=True)


#: The bind name the lexical arm's tsquery uses, in the shared params dict.
_LEXICAL_QUERY_PARAM = "lexical_query"


def _lexical_skip_reason(
    db_session: Any, query_text: str, weight: float
) -> Optional[str]:
    """Why the lexical arm should not run, or ``None`` to run it.

    The empty-tsquery case cannot be decided in Python: whether ``the a of
    and`` reduces to nothing is a property of the ``english`` configuration's
    stopword list, and ``websearch_to_tsquery`` is the only thing that knows
    it. So it is asked, once per recall, with a statement that touches no
    table. That is one sub-millisecond round trip on a path that has just
    awaited an embedding over the network — and it buys skipping up to two
    index scans that cannot match, plus a truthful skip reason in the metrics
    instead of a silent no-op.
    """
    if not lexical.lexical_enabled():
        return "disabled"
    if weight == 0.0:
        # Explicitly ``== 0.0`` and not ``not weight``: the operator's zero is
        # a decision, and this is the site where treating it as absence would
        # turn the kill switch back on.
        return "zero_weight"

    probe = f"SELECT {lexical.fts_query_expression(_LEXICAL_QUERY_PARAM)}::text = ''"
    try:
        empty = db_session.execute(
            text(probe), {_LEXICAL_QUERY_PARAM: query_text}
        ).scalar()
    except Exception:
        logger.warning("lexical tsquery probe failed; skipping the arm", exc_info=True)
        try:
            db_session.rollback()
        except Exception:
            logger.warning(
                "rollback after a failed tsquery probe failed", exc_info=True
            )
        return "probe_failed"

    return "empty_query" if empty else None


def _report_lexical(skip_reason: Optional[str], stats: dict, state: dict) -> None:
    """Emit the arm's four series once per recall, not once per fetch.

    The two-tier project fill calls the arm twice; counting per fetch would
    make "recalls where lexical contributed" un-interpretable against a recall
    rate.
    """
    if skip_reason is not None:
        metrics.record(metrics.LEXICAL_ARM_SKIPPED.inc, reason=skip_reason)
        return

    if stats["errors"]:
        outcome = "error"
    elif stats["rows"]:
        outcome = "rows"
    else:
        outcome = "empty"
    metrics.record(metrics.LEXICAL_ARM_QUERIES.inc, outcome=outcome)
    metrics.record(metrics.LEXICAL_ARM_SECONDS.observe, state["seconds"])
    if stats["novel"]:
        metrics.record(metrics.LEXICAL_ARM_CONTRIBUTED.inc)


@metrics.tracked(metrics.MEMORY_RECALLS, metrics.MEMORY_RECALL_SECONDS)
async def retrieve_memories_scoped(
    db_session: Any,
    org_id: str,
    query_text: str,
    *,
    scope: str = "org",
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 10,
) -> list[MemoryRecord]:
    """Brain recall across sessions, scoped to an org (and optionally an
    agent / space / project).

    ``limit`` bounds the vector stage, NOT the returned list: graph expansion
    can append up to :data:`GRAPH_EXPANSION_CAP` connected records on top of
    it. See *Returns*.

    Distinct from :func:`retrieve_memories`, which is session-scoped and used
    for verification context. This function deliberately does NOT narrow by
    ``session_id``: it spans every session in the org so the "brain" can recall
    facts learned in any prior conversation.

    Scope visibility ladder (filtered on the ``scope`` *column*, not just a
    Python switch — this is what keeps verification's ``scope='session'`` rows
    out of brain recall, since those rows live in the same table):

        * ``"org"`` (default) → the **visibility ladder**. Without a
          ``user_id`` this is exactly ``scope = 'org'``, byte-for-byte the
          historical behaviour. With a ``user_id`` it is the union of the
          three tiers the user is entitled to::

              scope = 'org'
              OR (scope = 'private' AND user_id = :user_id)
              OR (scope = 'project' AND project_id IN (
                    SELECT m.project_id FROM project_members m
                    JOIN projects p ON p.id = m.project_id
                     AND p.org_id = :org_id
                    WHERE m.user_id = :user_id))

          Note what is NOT in that predicate: the presence of ``project_id``.
          ``project_id`` is set on 32,568 pre-existing ``scope='org'`` rows
          where it is a *relevance* filter and explicitly not a boundary, and
          it is set on the new ``scope='project'`` rows where the boundary is
          the ``scope`` value. Keying access off ``project_id IS NOT NULL``
          would hide the former from the org; keying it off ``project_id IS
          NULL`` would leak the latter. The predicate is always ``scope``.
        * ``"private"`` (requires ``user_id``) →
          ``scope = 'private' AND user_id = :user_id``. Narrows to the
          caller's own memories alone: no org rows, no project rows, and —
          permanently, with no admin override — nobody else's private rows.
        * ``"agent"`` (requires ``agent_id``) →
          ``(scope = 'org' OR (scope = 'agent' AND agent_id = :agent_id))``.
          Deprecated in favour of ``private`` (``agent_id`` is client-declared
          via ``X-Vex-Agent`` and is therefore not an identity), but still
          READ so existing rows do not vanish. When a ``user_id`` is supplied
          the caller's own private and project tiers are unioned in as well,
          so an agent-scoped read is never *narrower* than the default one.
        * ``"session"`` → not supported here; raises ``ValueError``. Use
          :func:`retrieve_memories` for session-scoped reads instead of
          silently reading org-wide.
        * any other value → raises ``ValueError``.

    ``user_id`` is a SQL predicate on every arm that mentions it, exactly like
    ``org_id``. It is never a Python filter over a wider result set: a
    post-fetch filter means the wider rows were read, ranked, and one bug away
    from being returned. A caller that cannot resolve a user passes ``None``
    and sees the org tier only.

    NOTE: HNSW candidate-pool trade-off — the inner query fetches
    ``candidate_pool = limit * 4`` approximate nearest neighbours, then the
    outer query re-ranks them by the blended score and returns the top
    ``limit``.  A relevant-but-stale row that falls outside the top-4× ANN
    candidates will not be rescued by the recency bonus; this is accepted,
    matching the v1 under-fill acceptance documented below.

    NOTE: HNSW is an approximate index; with selective WHERE filters the result
    may under-fill ``limit``. Acceptable for v1.

    Hidden-row contract: rows with ``recall_hidden = TRUE`` are **always**
    excluded from results regardless of scope or other filters. These rows
    are raw passive-hook observations stored as curator fuel; only the curator
    process reads them (it does not filter on ``recall_hidden``).

    Args:
        db_session: A synchronous SQLAlchemy connection/session exposing
            ``.execute()`` (same contract as :func:`retrieve_memories`).
        org_id: Tenant whose memories to search. Always enforced — results are
            never returned from another org.
        query_text: Natural-language query; embedded and ranked by a blended
            score (``0.85 * similarity + 0.15 * recency_norm``) over a 30-day
            recency window (recency_norm clamped to [0, 1]) so that stale
            near-duplicates are demoted relative to recent relevant memories.
            Each returned :class:`MemoryRecord` has its ``relevance`` field
            populated with the raw cosine-similarity score (``1 - cosine
            distance``).  The inner query is HNSW-accelerated
            (``ORDER BY embedding <=> :embedding LIMIT :candidate_pool``);
            the outer re-ranks the bounded candidate pool by blended score.
        scope: One of ``"org"``, ``"private"`` or ``"agent"`` (see the
            visibility ladder above). ``"session"`` and any unknown value raise
            ``ValueError``.
        agent_id: Required when ``scope == "agent"``; ignored otherwise.
        user_id: The resolved caller. Required when ``scope == "private"``.
            Optional otherwise, where supplying it widens the read to include
            the caller's own private and project tiers and omitting it keeps
            the org tier alone.
        space_id: When given, match rows whose ``space_id`` equals it OR is
            ``NULL`` (unfiled memories surface in every space filter — the safe
            default so nothing is hidden by an over-narrow filter).
        project_id: Two-tier scoping. When given:
            * **Tier 1** — rows where ``project_id = :project_id`` fill the
              first ``limit`` slots.
            * **Tier 2** — unfiled rows (``project_id IS NULL``) fill any
              remaining slots.
            * Rows filed under OTHER projects are NEVER surfaced. This replaces
              the old flat ``(project_id = X OR project_id IS NULL)`` filter.
        limit: Maximum number of records the VECTOR SEARCH may return. It is
            NOT a bound on the length of the returned list — see below.

    Returns:
        A list of :class:`MemoryRecord` ordered by blended score (descending),
        or an empty list if embedding generation fails. Vector-matched records
        carry a rounded cosine-similarity score (0–1) in ``relevance``.

        **The list can be longer than ``limit``.** After the vector stage,
        graph expansion appends up to :data:`GRAPH_EXPANSION_CAP` further
        records that are connected to the vector matches, so the true bound is
        ``limit + GRAPH_EXPANSION_CAP`` — ``limit=1`` can return 6. Expanded
        records are marked ``via=VIA_GRAPH`` and have ``relevance=None``,
        which is how a caller tells them apart. This is intentional (the
        connected row is often the answer the query could not phrase), and it
        is why every caller that budgets a context window must size against
        ``limit + GRAPH_EXPANSION_CAP`` rather than ``limit``.

    Raises:
        ValueError: If ``scope == "agent"`` but ``agent_id`` is ``None``; if
            ``scope == "private"`` but ``user_id`` is ``None``; if
            ``scope == "session"`` (unsupported here); or if ``scope`` is any
            unknown value.
    """
    # Validate scope BEFORE the (expensive) embed call so bad input fails fast.
    if scope == "org":
        pass
    elif scope == "private":
        if not user_id:
            # Fail rather than fall back to org: a caller asking for "mine"
            # and silently receiving the team's brain is the wrong kind of
            # surprise, and there is no correct owner to guess.
            raise ValueError("user_id is required when scope='private'")
    elif scope == "agent":
        if agent_id is None:
            raise ValueError("agent_id is required when scope='agent'")
    elif scope == "session":
        raise ValueError(
            "scope='session' is not supported by retrieve_memories_scoped; "
            "use retrieve_memories for session-scoped reads"
        )
    else:
        raise ValueError(f"unknown scope: {scope!r}")

    # Availability: return the pooled connection BEFORE awaiting the
    # embedding. See :func:`release_connection` for the measurements.
    release_connection(db_session)
    embedding = await embed_text(query_text)
    if embedding is None:
        return []

    # Bind the query vector exactly as retrieve_memories does: the embedding
    # list is rendered via str() and bound as a plain parameter, which pgvector
    # accepts directly with the cosine-distance operator (<=>).
    embedding_str = str(embedding)

    base_clauses = [
        "org_id = :org_id",
        "status = 'active'",
        "superseded_by IS NULL",
        "recall_hidden = FALSE",
    ]
    params: dict[str, Any] = {
        "org_id": org_id,
        "embedding": embedding_str,
        "limit": limit,
        "candidate_pool": limit * 4,
    }

    # ---- the visibility ladder -------------------------------------------
    # Built as a single OR-group of scope arms. Every arm that reaches beyond
    # 'org' carries its own owner/membership predicate in SQL; there is no
    # arm that admits a row on the strength of project_id alone.
    if scope == "private":
        # Narrowed read: mine, and only mine. No org arm, no project arm.
        base_clauses.append("scope = 'private' AND user_id = :user_id")
        params["user_id"] = user_id
    else:
        arms = ["scope = 'org'"]
        if scope == "agent":  # validated above; agent_id is not None
            arms.append("(scope = 'agent' AND agent_id = :agent_id)")
            params["agent_id"] = agent_id
        if user_id:
            arms.append("(scope = 'private' AND user_id = :user_id)")
            # Membership is a subquery rather than a pre-fetched Python list
            # so the entitlement is evaluated by the same statement that
            # returns the rows — a stale or mis-passed list cannot widen it.
            # The JOIN to projects is not decoration. project_members has no
            # org_id of its own, so without it a membership row in ANOTHER
            # tenant would satisfy this arm for a row in THIS one — the outer
            # org_id filter constrains the memory, not the entitlement. An
            # adversarial test seeds exactly that shape.
            arms.append(
                "(scope = 'project' AND project_id IN ("
                "SELECT m.project_id FROM project_members m "
                "JOIN projects p ON p.id = m.project_id "
                "AND p.org_id = :org_id "
                "WHERE m.user_id = :user_id))"
            )
            params["user_id"] = user_id
        base_clauses.append("(" + " OR ".join(arms) + ")")

    if space_id is not None:
        base_clauses.append("(space_id = :space_id OR space_id IS NULL)")
        params["space_id"] = space_id

    # ANN-inner / re-rank-outer shape:
    #   Inner query — HNSW-accelerated cosine-distance ORDER BY fetches a
    #     bounded candidate pool (:candidate_pool = limit * 4).
    #   Outer query — re-ranks the candidates by blended score:
    #     0.85 * similarity + 0.15 * recency_norm, where recency_norm is the
    #     fraction of the 30-day window still remaining, clamped to [0, 1].
    # The outer SELECT * preserves the record columns plus relevance as the
    # LAST column.  blended_score is used only in the ORDER BY and is not a
    # selected column.  Relevance is read as ``row[-1]`` and the record from
    # ``row[:-1]`` rather than at fixed indices, so appending a column here
    # (user_id and provenance were appended in phase 2) does not silently
    # shift what ``relevance`` reads.
    #
    # The outer LIMIT is ``:vector_limit`` rather than ``:limit``, and the two
    # differ ONLY when the lexical arm is live. This is the one place the
    # vector statement changed, and it had to:
    #
    #   RRF at the configured weights makes a lexical-only row score
    #   0.5/(60+1) = 0.0082, while ANY vector row scores at least
    #   1/(60+limit). For every realistic limit that is larger, so a row the
    #   vector arm did not return could never displace one it did — the whole
    #   feature would be inert. Truncating the vector arm to ``limit`` BEFORE
    #   fusion is what caused that: the row containing the literal was cut
    #   before fusion could see it. Fusing the candidate pool instead and
    #   truncating AFTER lets the two signals agree: the target scores
    #   1/(60+6) + 0.5/(60+1) = 0.0233 against the best distractor's 0.0164.
    #
    # Fusing candidates and truncating afterwards is also the ordinary shape
    # of RRF; truncating each arm first is what is unusual. The single-arm
    # invariant is unaffected: with the lexical arm empty the fused order is
    # the vector order, and taking its first ``limit`` rows is byte-for-byte
    # what ``LIMIT :limit`` returned before. When the arm is off,
    # ``vector_limit`` IS ``limit`` and the statement is unchanged.
    select_sql = """
        SELECT * FROM (
            SELECT id, session_id, org_id, agent_id, execution_id,
                   memory_type, content, embedding, confidence,
                   sequence_number, scope, status, superseded_by,
                   metadata, created_at, updated_at, space_id, project_id,
                   user_id, provenance, revision,
                   (1 - (embedding <=> :embedding)) AS relevance
            FROM session_memories
            WHERE {where}
            ORDER BY embedding <=> :embedding
            LIMIT :candidate_pool
        ) candidates
        ORDER BY (0.85 * relevance
                  + 0.15 * LEAST(1, GREATEST(
                      0, 1 - EXTRACT(EPOCH FROM (NOW() - created_at))
                             / (30 * 86400)
                    ))) DESC
        LIMIT :vector_limit
    """

    # ---- the lexical arm --------------------------------------------------
    # Same shape, same {where}, same params, same candidate-pool bound. The
    # ONLY predicate this query adds is the full-text match itself: the
    # tenancy, scope, status and recall_hidden clauses are the identical
    # ``base_clauses`` list object the vector arm builds its WHERE from, so a
    # predicate cannot be present on one arm and missing on the other. That is
    # not stylistic. The Wave 1 defect was exactly this shape — a second query
    # path that filtered on session and org but not on scope or recall_hidden,
    # while the boundary held correctly everywhere else in this file.
    #
    # ``ts_rank`` is selected as the last column so the row layout matches the
    # vector arm's (record columns, then one score), but the value is NOT
    # written into ``relevance``: ts_rank is unbounded and corpus-dependent
    # while ``relevance`` is documented as a cosine similarity in [0, 1].
    # Lexical-only rows carry ``relevance=None`` and ``via=VIA_LEXICAL``,
    # exactly as graph-expanded rows do.
    lexical_sql = """
        SELECT id, session_id, org_id, agent_id, execution_id,
               memory_type, content, embedding, confidence,
               sequence_number, scope, status, superseded_by,
               metadata, created_at, updated_at, space_id, project_id,
               user_id, provenance, revision,
               ts_rank({document}, {query}) AS lexical_rank
        FROM session_memories
        WHERE {where}
          AND {document} @@ {query}
        ORDER BY ts_rank({document}, {query}) DESC, id
        LIMIT :candidate_pool
    """.replace("{document}", lexical.fts_document_expression()).replace(
        "{query}", lexical.fts_query_expression(_LEXICAL_QUERY_PARAM)
    )

    # One shared params dict, one shared bound value. The vector statement
    # never mentions :lexical_query; binding it there is harmless and keeps
    # the two arms provably parameterised from the same place.
    params[_LEXICAL_QUERY_PARAM] = query_text

    lexical_weight = lexical.lexical_weight()
    fusion_k = lexical.rrf_k()
    skip_reason = _lexical_skip_reason(db_session, query_text, lexical_weight)
    # Mutable because a lexical failure mid-recall must not be retried on the
    # tier-2 fetch: the failed statement has already aborted the transaction.
    lexical_state = {"live": skip_reason is None, "seconds": 0.0}
    stats = {"rows": 0, "novel": 0, "errors": 0}

    def _to_records(rows: list, *, via: Optional[str], scored: bool) -> list:
        out = []
        for row in rows:
            # relevance is the LAST selected column; everything before it is
            # the record. blended_score is not selected — it only drives the
            # ORDER BY.
            record = _row_to_record(row[:-1])
            relevance = row[-1] if scored else None
            record.relevance = (
                round(float(relevance), 3) if relevance is not None else None
            )
            record.via = via
            out.append(record)
        return out

    def _fetch_lexical(where_sql: str, merged: dict[str, Any]) -> list:
        """The lexical rows, or ``[]`` if the arm is off or has failed.

        Failure containment mirrors :func:`_expand_via_graph`: log, roll the
        session back so the aborted transaction cannot poison every later
        statement, and let the caller carry on with the vector rows alone.
        """
        if not lexical_state["live"]:
            return []
        started = time.monotonic()
        try:
            return db_session.execute(
                text(lexical_sql.format(where=where_sql)), merged
            ).fetchall()
        except Exception:
            logger.warning(
                "recall lexical arm failed; returning vector results alone",
                exc_info=True,
            )
            stats["errors"] += 1
            lexical_state["live"] = False
            try:
                db_session.rollback()
            except Exception:
                logger.warning(
                    "rollback after a failed lexical query also failed",
                    exc_info=True,
                )
            return []
        finally:
            lexical_state["seconds"] += time.monotonic() - started

    def _fetch(extra_clause: str, extra_params: dict[str, Any]) -> list:
        where_sql = "\n              AND ".join(base_clauses + [extra_clause])
        merged = {**params, **extra_params}
        # The vector arm reads deeper than ``limit`` only when there is a
        # second arm to fuse against; otherwise it stops exactly where it
        # always did.
        merged["vector_limit"] = (
            merged["candidate_pool"] if lexical_state["live"] else merged["limit"]
        )
        vector_rows = db_session.execute(
            text(select_sql.format(where=where_sql)), merged
        ).fetchall()

        lexical_rows = _fetch_lexical(where_sql, merged)
        if not lexical_rows:
            # Not merely an optimisation: this is the pre-change result,
            # row-for-row, whenever lexical contributes nothing. The slice is
            # what ``LIMIT :limit`` did before, applied to the same order.
            return _to_records(vector_rows[: merged["limit"]], via=None, scored=True)

        by_id = {str(row[0]): row for row in vector_rows}
        vector_ids = [str(row[0]) for row in vector_rows]
        lexical_ids = [str(row[0]) for row in lexical_rows]
        lexical_only = {row_id: row for row_id, row in zip(lexical_ids, lexical_rows)}

        stats["rows"] += 1
        if any(row_id not in by_id for row_id in lexical_ids):
            stats["novel"] += 1

        fused_ids = lexical.reciprocal_rank_fusion(
            [(lexical.VECTOR_WEIGHT, vector_ids), (lexical_weight, lexical_ids)],
            k=fusion_k,
        )[: merged["limit"]]

        records = []
        for row_id in fused_ids:
            if row_id in by_id:
                records.extend(_to_records([by_id[row_id]], via=None, scored=True))
            else:
                records.extend(
                    _to_records([lexical_only[row_id]], via=VIA_LEXICAL, scored=False)
                )
        return records

    if project_id is not None:
        # Two-tier: same-project rows fill first; remaining slots come from
        # unfiled (org-level) rows. Rows filed under OTHER projects never
        # surface. Fusion happens INSIDE _fetch, so both tiers see both arms
        # and the tier semantics are structurally unchanged.
        tier1 = _fetch("project_id = :project_id", {"project_id": project_id})
        remaining = limit - len(tier1)
        tier2 = []
        if remaining > 0:
            # Override both limit and candidate_pool for the tier-2 fill so
            # the inner ANN pool is proportional to the actual remaining slots.
            tier2 = _fetch(
                "project_id IS NULL",
                {"limit": remaining, "candidate_pool": remaining * 4},
            )
        records = list(tier1) + list(tier2)
    else:
        records = _fetch("TRUE", {})

    _report_lexical(skip_reason, stats, lexical_state)

    # Stage B+C (knowledge graph): one-hop, ladder-scoped expansion around the
    # vector entry points, then the blended re-rank. Adds at most
    # GRAPH_EXPANSION_CAP rows; on any failure the vector results above are
    # returned unchanged.
    return _expand_via_graph(
        db_session,
        records,
        org_id=org_id,
        scope=scope,
        agent_id=agent_id,
        user_id=user_id,
        space_id=space_id,
        project_id=project_id,
    )
