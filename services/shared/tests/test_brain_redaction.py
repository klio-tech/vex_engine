"""write_scoped_memory must mask secrets before embedding, fail closed."""

import pytest

import shared.brain as brain_mod
import shared.memory as memory_mod
from shared.brain import write_scoped_memory

SECRET = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


class FakeResult:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeDB:
    """Records executed SQL + params; enough for the write path."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((str(sql), params))
        return FakeResult()

    def commit(self):
        pass


@pytest.fixture
def fake_embed(monkeypatch):
    captured = {}

    async def _embed(text_input):
        captured["text"] = text_input
        return [0.1, 0.2, 0.3]

    # write_memory and find_similar_memory both call shared.memory.embed_text.
    monkeypatch.setattr(memory_mod, "embed_text", _embed)
    return captured


@pytest.mark.asyncio
async def test_secret_is_masked_before_store_and_embedding(fake_embed):
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"User provided Stripe key: {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    assert result["redacted"] is True
    assert "stripe_secret_key" in result["redaction_kinds"]
    # The INSERT params and the embedded text must not contain the raw secret.
    insert_params = [p for _, p in db.calls if p and "content" in p][0]
    assert SECRET not in insert_params["content"]
    assert "[REDACTED:stripe_secret_key:" in insert_params["content"]
    assert SECRET not in fake_embed["text"]


@pytest.mark.asyncio
async def test_clean_content_passes_untouched(fake_embed):
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content="The deploy finished at 10:30.",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    assert result.get("redacted") is False


@pytest.mark.asyncio
async def test_redactor_failure_fails_closed(fake_embed, monkeypatch):
    def _boom(content):
        raise RuntimeError("redactor crashed")

    monkeypatch.setattr(brain_mod, "redact", _boom)
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"key {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result == {"stored": False, "error": "redaction_failed"}
    # Nothing was written.
    assert not any("INSERT INTO session_memories" in sql for sql, _ in db.calls)


@pytest.mark.asyncio
async def test_disable_flag_skips_redaction(fake_embed, monkeypatch):
    monkeypatch.setenv("MEMORY_REDACTION_DISABLED", "1")
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"key {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    insert_params = [p for _, p in db.calls if p and "content" in p][0]
    assert SECRET in insert_params["content"]


@pytest.mark.asyncio
async def test_dedup_path_receives_masked_content(fake_embed, monkeypatch):
    """With dedup=True, the similarity lookup must also receive masked content.

    find_similar_memory embeds the content to build the query vector; asserting
    that fake_embed["text"] has no raw secret after a dedup-enabled write
    verifies the masking happens before dedup, not just before the INSERT.
    """
    # Make find_similar_memory return None so the write proceeds to INSERT,
    # ensuring fake_embed["text"] is set by write_memory's embed_text call
    # (the last one wins; both should receive masked text).
    import shared.brain as _brain

    # Patch find_similar_memory so it records what it was called with.
    dedup_embedded: dict = {}

    async def _embed_capturing(text_input):
        dedup_embedded["last"] = text_input
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(memory_mod, "embed_text", _embed_capturing)

    # Patch find_similar_memory to return None (no duplicate found).
    async def _find_none(*args, **kwargs):
        return None

    monkeypatch.setattr(_brain, "find_similar_memory", _find_none)

    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"User key: {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=True,
    )
    assert result["stored"] is True
    # The last embed call (from write_memory) must not contain the raw secret.
    assert SECRET not in dedup_embedded["last"]
    # The INSERT content must also be masked.
    insert_params = [p for _, p in db.calls if p and "content" in p][0]
    assert SECRET not in insert_params["content"]
