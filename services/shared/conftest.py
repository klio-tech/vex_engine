import os
import sys

import pytest

# Ensure the shared package is importable
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _clear_embedding_memo():
    """Isolate tests from ``shared.memory``'s process-wide embedding memo.

    The memo collapses the two embeds a single write performs (the dedup probe
    and the insert) into one backend call, keyed on ``(model, content)``. That
    is sound in production — an embedding is a pure function of those two — but
    it means two *tests* that embed the same string are no longer independent:
    the second sees the first's vector and never reaches its own stubbed
    backend. Clearing between tests keeps each one's stub the thing under test.
    """
    from shared import memory

    memory.clear_embedding_memo()
    yield
    memory.clear_embedding_memo()
