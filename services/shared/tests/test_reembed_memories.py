"""The re-embed job: same input as a live write, order kept, failures isolated."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from reembed_memories import (  # noqa: E402
    TABLES,
    embed_page,
    model_input,
    vector_literal,
    write_vectors,
)

from shared.memory import MAX_EMBED_CHARS


def run(coro):
    return asyncio.run(coro)


def test_model_input_matches_a_live_write():
    """A re-embedded row must get the vector a fresh write of it would."""
    long = "x" * (MAX_EMBED_CHARS + 500)
    assert model_input(long) == long[:MAX_EMBED_CHARS]
    assert model_input("short") == "short"
    assert model_input(None) == ""


def test_vector_literal_is_a_pgvector_literal():
    assert vector_literal([0.5, -1, 2.25]) == "[0.5,-1.0,2.25]"


def test_entities_are_not_recomputed():
    """No caller defines what an entity vector embeds; the job must not guess."""
    assert set(TABLES) == {"session_memories", "work_claims"}


def test_pages_come_back_in_input_order():
    calls = []

    async def embed_batch(texts):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    async def embed_one(_):
        raise AssertionError("no batch failed, so no row-by-row retry")

    texts = [f"row-{i:03d}" + "y" * i for i in range(150)]
    vectors = run(
        embed_page(texts, embed_batch=embed_batch, embed_one=embed_one, batch_size=64)
    )

    assert vectors == [[float(len(t))] for t in texts]
    assert [len(c) for c in sorted(calls, key=len, reverse=True)] == [64, 64, 22]


def test_a_rejected_batch_is_retried_row_by_row_not_as_a_batch():
    """One bad row must not sink the other rows in its batch."""

    async def embed_batch(texts):
        if "poison" in texts:
            raise RuntimeError("400 from the gateway")
        return [[1.0] for _ in texts]

    retried = []

    async def embed_one(value):
        retried.append(value)
        return None if value == "poison" else [2.0]

    texts = ["a", "poison", "b", "c"]
    vectors = run(
        embed_page(texts, embed_batch=embed_batch, embed_one=embed_one, batch_size=4)
    )

    assert retried == texts
    assert vectors == [[2.0], None, [2.0], [2.0]]


def test_a_short_batch_response_is_treated_as_a_failure():
    """Zipping 63 vectors onto 64 rows would give every later row the wrong vector."""

    async def embed_batch(texts):
        return [[1.0] for _ in texts[:-1]]

    async def embed_one(_):
        return [9.0]

    vectors = run(
        embed_page(
            ["a", "b", "c"], embed_batch=embed_batch, embed_one=embed_one, batch_size=3
        )
    )
    assert vectors == [[9.0], [9.0], [9.0]]


class FakeDB:
    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((str(sql), params))


def test_a_row_that_could_not_be_embedded_keeps_its_old_vector():
    db = FakeDB()
    written, skipped = write_vectors(
        db, "session_memories", ["id-1", "id-2"], [[0.1], None]
    )

    assert (written, skipped) == (1, 1)
    assert len(db.statements) == 1
    sql, params = db.statements[0]
    assert "UPDATE session_memories SET embedding" in sql
    assert params == {"v": "[0.1]", "id": "id-1"}


@pytest.mark.parametrize("table", sorted(TABLES))
def test_every_table_names_its_source_column(table):
    assert TABLES[table] in {"content", "description"}
