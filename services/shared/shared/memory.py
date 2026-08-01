"""Shared session memory module.

Provides embedding generation, pgvector read/write, semantic search,
deduplication, fact extraction, and summary generation for agent session
memory.  All DB access uses raw SQL via ``sqlalchemy.text()`` and pgvector's
cosine distance operator (``<=>``) for similarity queries.
"""

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import litellm
from litellm import acompletion, aembedding
from sqlalchemy import text

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

EMBED_MIN_CHARS: int = 256
"""Smallest truncation floor for :func:`embed_text`'s progressive retry.

When an embedding-backend call fails, :func:`embed_text` halves the input and
retries, down to this floor. A failure at (or below) the floor is treated as
the embedding service being genuinely unavailable → ``None``; it is NOT the
path for "the content was long". Consequently a short input that fails returns
``None`` on a single attempt (no needless retries), while a long/dense input is
shrunk repeatedly until it fits under the model's token cap."""

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


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


async def _aembedding_call(text_input: str) -> list[float]:
    """Perform a single embedding-backend call via ``litellm.aembedding()``.

    Reads the model name from the ``MEMORY_EMBEDDING_MODEL`` env var, falling
    back to :data:`DEFAULT_EMBEDDING_MODEL`, and optional gateway credentials
    from the ``LITELLM_*`` / ``OPENAI_*`` env vars.

    Isolated from :func:`embed_text` so the latter can retry it with
    progressively smaller input without duplicating the litellm wiring. Raises
    whatever the backend raises (e.g. an over-token-budget HTTP 400).

    Returns:
        The embedding vector as a list of floats.
    """
    model = os.environ.get("MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    api_base = os.environ.get("LITELLM_API_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    kwargs: dict[str, Any] = {"model": model, "input": [text_input]}
    if api_base:
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "openai"
    if api_key:
        kwargs["api_key"] = api_key

    response = await aembedding(**kwargs)
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

    Returns:
        A list of floats representing the embedding, or ``None`` only when the
        embedding backend fails at every attempted size.
    """
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
            return await _aembedding_call(candidate)
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
    timeout_s = float(os.environ.get("VERIFICATION_TIMEOUT_S", DEFAULT_LLM_TIMEOUT_S))

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


async def generate_summary(
    output: Any,
    task: Optional[str] = None,
    input_data: Optional[Any] = None,
) -> Optional[str]:
    """Generate a compact 1-2 sentence turn summary via :func:`call_llm`.

    Returns:
        The summary string, or ``None`` if the LLM call fails.
    """
    parts = [f"Output: {output}"]
    if task:
        parts.append(f"Task: {task}")
    if input_data:
        parts.append(f"Input: {input_data}")

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
    parts = [f"Output: {output}"]
    if task:
        parts.append(f"Task: {task}")
    if input_data:
        parts.append(f"Input: {input_data}")

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
        provenance (index 19)

    ``user_id`` and ``provenance`` are read positionally like the two before
    them, and every caller that selects them appends them in this order. A
    caller that selects neither still gets a valid record: ``user_id`` stays
    ``None`` (the pre-migration-031 state for 37,605 rows) and ``provenance``
    falls back to the column's own default, ``EXTRACTED``.
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
    )


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
    embedding = await embed_text(content)
    if embedding is None:
        return None

    memory_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    meta = metadata or {}

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

    1. Semantic search across all active memories for the session,
       ordered by cosine distance (``embedding <=> :embedding``).
    2. Fetches the last *include_summaries* summaries by
       ``sequence_number`` for recency context.

    Results are deduplicated by ``id`` and returned as a combined list.

    Returns:
        A list of :class:`MemoryRecord` instances, or an empty list if
        embedding generation fails.
    """
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

    result = db_session.execute(
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
    )

    row = result.fetchone()
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
) -> None:
    """Mark an old memory as superseded by a newer one.

    Sets ``status = 'superseded'`` and ``superseded_by = new_memory_id``
    on the old memory row.
    """
    db_session.execute(
        text("""
            UPDATE session_memories
            SET status = 'superseded',
                superseded_by = :new_memory_id,
                updated_at = NOW()
            WHERE id = :old_memory_id
        """),
        {
            "old_memory_id": str(old_memory_id),
            "new_memory_id": str(new_memory_id),
        },
    )
    db_session.commit()


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
      * scope='agent' row owned by another agent → forbidden
      * otherwise → status='retracted' + audit metadata; recall never
        returns non-'active' rows, so the memory disappears immediately.

    Why a private row answers ``not_found`` and not ``forbidden``: ``forbidden``
    confirms the row exists. Someone who guessed or was shown a memory id would
    learn that this org holds a private memory under it, which is a disclosure
    about a scope whose whole promise is opacity from the outside. A
    non-owner's view of another person's private memory is identical to the
    view of an id that was never issued.

    Synchronous (inline psycopg2 I/O), matching the house pattern.
    """
    try:
        parsed = uuid.UUID(memory_id)
    except (ValueError, AttributeError, TypeError):
        return {"forgotten": False, "error": "invalid_id"}

    row = db_session.execute(
        text(
            """
            SELECT scope, agent_id, user_id FROM session_memories
            WHERE id = :id AND org_id = :org_id AND status = 'active'
            LIMIT 1
            """
        ),
        {"id": str(parsed), "org_id": org_id},
    ).fetchone()
    if row is None:
        return {"forgotten": False, "error": "not_found"}
    if row.scope == "private" and (not user_id or row.user_id != user_id):
        return {"forgotten": False, "error": "not_found"}
    if row.scope == "agent" and row.agent_id != agent_id:
        return {"forgotten": False, "error": "forbidden"}

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
                    SELECT project_id FROM project_members
                    WHERE user_id = :user_id))

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
        limit: Maximum number of records to return.

    Returns:
        A list of :class:`MemoryRecord` ordered by blended score (descending),
        or an empty list if embedding generation fails. Each record's
        ``relevance`` field holds the rounded cosine-similarity score (0–1).

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
            arms.append(
                "(scope = 'project' AND project_id IN ("
                "SELECT project_id FROM project_members "
                "WHERE user_id = :user_id))"
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
    select_sql = """
        SELECT * FROM (
            SELECT id, session_id, org_id, agent_id, execution_id,
                   memory_type, content, embedding, confidence,
                   sequence_number, scope, status, superseded_by,
                   metadata, created_at, updated_at, space_id, project_id,
                   user_id, provenance,
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
        LIMIT :limit
    """

    def _fetch(extra_clause: str, extra_params: dict[str, Any]) -> list:
        where_sql = "\n              AND ".join(base_clauses + [extra_clause])
        return db_session.execute(
            text(select_sql.format(where=where_sql)),
            {**params, **extra_params},
        ).fetchall()

    if project_id is not None:
        # Two-tier: same-project rows fill first; remaining slots come from
        # unfiled (org-level) rows. Rows filed under OTHER projects never
        # surface.
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
        rows = list(tier1) + list(tier2)
    else:
        rows = _fetch("TRUE", {})

    records = []
    for row in rows:
        # relevance is the LAST selected column; everything before it is the
        # record. blended_score is not selected — it only drives ORDER BY.
        relevance = row[-1]
        record = _row_to_record(row[:-1])
        record.relevance = round(float(relevance), 3) if relevance is not None else None
        records.append(record)
    return records
