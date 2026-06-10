"""embed_text must truncate input below the model's token limit so oversized
content cannot return HTTP 400 and loop the embedder."""

import shared.memory as memory_mod
from shared.memory import MAX_EMBED_CHARS


async def _run_embed(monkeypatch, text_input):
    captured = {}

    async def _fake_aembedding(**kwargs):
        captured["input"] = kwargs["input"]

        class _Item(dict):
            pass

        return type("R", (), {"data": [{"embedding": [0.1, 0.2]}]})()

    monkeypatch.setattr(memory_mod, "aembedding", _fake_aembedding)
    result = await memory_mod.embed_text(text_input)
    return result, captured


async def test_oversized_input_is_truncated(monkeypatch):
    big = "x" * (MAX_EMBED_CHARS + 5000)
    result, captured = await _run_embed(monkeypatch, big)
    assert result == [0.1, 0.2]
    sent = captured["input"][0]
    assert len(sent) == MAX_EMBED_CHARS


async def test_short_input_passes_through_unchanged(monkeypatch):
    small = "a durable fact about the user"
    result, captured = await _run_embed(monkeypatch, small)
    assert captured["input"][0] == small


async def test_max_embed_chars_is_conservative():
    # ~4 chars/token, model limit 512 tokens; stay well under.
    assert MAX_EMBED_CHARS <= 2000
