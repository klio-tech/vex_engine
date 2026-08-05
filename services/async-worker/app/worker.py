"""Core processing logic for the async verification worker.

Receives an IngestEvent, runs the verification engine pipeline, and
produces a verified event dict suitable for publishing to the
``executions.verified`` Redis Stream.

This module is intentionally decoupled from the Redis consumer loop so
it can be tested in isolation.
"""

import json
import logging
import os
from typing import Any

from engine.pipeline import verify
from shared.memory import (
    extract_facts,
    find_similar_memory,
    supersede_memory,
    write_memory,
)
from shared.memory_config import is_memory_enabled
from shared.models import IngestEvent
from shared.redaction import redact

logger = logging.getLogger("agentguard.async-worker")

VERIFIED_STREAM_KEY = "executions.verified"

#: Cap on any single caller-supplied field forwarded into an LLM prompt.
#: ``event.output``/``task``/``input`` are unbounded on the wire, and the
#: fact-extraction prompt interpolates them verbatim.
MAX_PROMPT_INPUT_CHARS = 20_000


def _bound_prompt_input(value: Any) -> Any:
    """Truncate a caller-supplied value before it is interpolated into a prompt."""
    if value is None:
        return None
    as_text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(as_text) <= MAX_PROMPT_INPUT_CHARS:
        return value
    return as_text[:MAX_PROMPT_INPUT_CHARS] + "…[truncated]"


def _get_db_session():
    """Create a SQLAlchemy session for memory operations."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",  # pragma: allowlist secret
    )
    engine = create_engine(db_url)
    return Session(engine)


def _is_memory_enabled() -> bool:
    return is_memory_enabled()


async def process_event(event: IngestEvent) -> dict[str, Any]:
    """Run verification on an ingest event and return a verified event dict.

    Args:
        event: The validated ingest event to verify.

    Returns:
        A dict representing the verified event, suitable for publishing
        to the ``executions.verified`` Redis Stream.
    """
    try:
        result = await verify(
            output=event.output,
            task=event.task,
            schema=event.schema_definition,
            ground_truth=event.ground_truth,
            conversation_history=event.conversation_history,
            steps=event.steps,
            metadata=event.metadata,
        )

        checks_data = {}
        for name, check in result.checks.items():
            checks_data[name] = {
                "check_type": check.check_type,
                "score": check.score,
                "passed": check.passed,
                "details": check.details,
            }

        org_id = (
            event.metadata.get("org_id", "default") if event.metadata else "default"
        )

        verified_event = {
            "execution_id": event.execution_id,
            "agent_id": event.agent_id,
            "org_id": org_id,
            "confidence": str(result.confidence)
            if result.confidence is not None
            else "",
            "action": result.action,
            "checks": json.dumps(checks_data),
        }

        # --- Extract and store facts when memory is enabled ---
        if _is_memory_enabled() and event.session_id:
            try:
                org_id_val = org_id
                facts = await extract_facts(
                    output=_bound_prompt_input(event.output),
                    task=_bound_prompt_input(event.task),
                    input_data=_bound_prompt_input(event.input),
                )
                if facts:
                    db_session = _get_db_session()
                    try:
                        for raw_fact in facts:
                            # This path calls write_memory directly, so it does
                            # NOT go through brain.write_scoped_memory — the
                            # only other place the redactor runs. An extracted
                            # fact routinely quotes the transcript back, so a
                            # credential in the transcript lands here verbatim
                            # and is then embedded.
                            redaction = redact(raw_fact)
                            fact = redaction.clean_text
                            if redaction.redacted:
                                logger.info(
                                    "Redacted %d secret(s) from extracted fact for %s",
                                    len(redaction.findings),
                                    event.execution_id,
                                    extra={
                                        "execution_id": event.execution_id,
                                        "redaction_kinds": sorted(
                                            {f.kind for f in redaction.findings}
                                        ),
                                    },
                                )
                            # Dedup: check if a similar fact already exists.
                            # Uses the redacted text so the dedup vector cannot
                            # be built from the raw secret either.
                            existing = await find_similar_memory(
                                db_session=db_session,
                                session_id=event.session_id,
                                org_id=org_id_val,
                                content=fact,
                            )
                            new_record = await write_memory(
                                db_session=db_session,
                                session_id=event.session_id,
                                org_id=org_id_val,
                                agent_id=event.agent_id,
                                execution_id=event.execution_id,
                                memory_type="fact",
                                content=fact,
                                confidence=result.confidence,
                                sequence_number=event.sequence_number,
                            )
                            # If similar fact found and new one was written, supersede old
                            if existing and new_record:
                                supersede_memory(
                                    db_session,
                                    existing.id,
                                    new_record.id,
                                    org_id=org_id_val,
                                )
                    finally:
                        db_session.close()
            except Exception:
                logger.warning(
                    "Failed to extract/store facts for event %s",
                    event.execution_id,
                    exc_info=True,
                )

        return verified_event
    except Exception:
        logger.error(
            "Verification failed for event %s; returning pass-through",
            event.execution_id,
            exc_info=True,
        )
        org_id = (
            event.metadata.get("org_id", "default") if event.metadata else "default"
        )

        return {
            "execution_id": event.execution_id,
            "agent_id": event.agent_id,
            "org_id": org_id,
            "confidence": "",
            "action": "pass",
            "checks": "{}",
        }
