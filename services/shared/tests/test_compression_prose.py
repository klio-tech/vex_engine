"""The prose compressor is the only one that can silently lose meaning.

So the tests here are mostly about what it must *not* do:

* a missing model, a missing runtime or a failed download returns the input
  unchanged — never an exception, never a truncation;
* every protected span (path, URL, identifier, signature, number) survives
  byte-for-byte even when the scorer votes to drop it;
* the default threshold is the conservative one from the spec.

The token-scoring seam is stubbed so these run without the 600 MB weights.
A separate opt-in test exercises the real model when it is present.
"""

import re

import pytest

from shared.compression import prose as prose_module
from shared.compression.prose import DEFAULT_THRESHOLD, compress_prose

PARAGRAPH = (
    "The deployment failed because the database connection pool was exhausted. "
    "We should probably look into raising the pool size at some point soon. "
    "The error surfaced in the checkout service during the evening peak."
)


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """A test that stubs the loader must not leave the cache poisoned for the
    next one — the load result is deliberately remembered in production."""
    yield
    prose_module.reset_model_cache()


def _stub_scores(monkeypatch, decide):
    """Install a fake token scorer.

    ``decide`` receives the token text and returns a keep probability.
    Whitespace tokenisation is enough: the real model's contract is a list of
    ``(start, end, score)`` triples over the input, and that is what we fake.
    """

    def _score(text: str):
        return [
            (m.start(), m.end(), decide(m.group(0))) for m in re.finditer(r"\S+", text)
        ]

    monkeypatch.setattr(prose_module, "_score_tokens", _score)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_default_threshold_is_the_conservative_operating_point():
    """0.30 is 8% dropped at 0.994 must-keep recall. Under-delivering is
    recoverable; silently dropping a needed line is not."""
    assert DEFAULT_THRESHOLD == 0.30


# --------------------------------------------------------------------------
# Fail-open
# --------------------------------------------------------------------------


def test_an_unavailable_model_returns_the_input_unchanged(monkeypatch):
    monkeypatch.setattr(prose_module, "_score_tokens", lambda _text: None)
    assert compress_prose(PARAGRAPH) == PARAGRAPH


def test_a_scorer_that_raises_returns_the_input_unchanged(monkeypatch):
    def boom(_text):
        raise RuntimeError("onnxruntime is not installed")

    monkeypatch.setattr(prose_module, "_score_tokens", boom)
    assert compress_prose(PARAGRAPH) == PARAGRAPH


def test_a_failed_load_is_reported_once_and_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(prose_module, "_load_model", lambda: None)
    prose_module.reset_model_cache()
    with caplog.at_level("WARNING"):
        assert compress_prose(PARAGRAPH) == PARAGRAPH
    assert prose_module.model_available() is False


def test_empty_input_is_returned_unchanged(monkeypatch):
    _stub_scores(monkeypatch, lambda _w: 0.0)
    assert compress_prose("") == ""


# --------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------


def test_tokens_below_the_threshold_are_dropped(monkeypatch):
    _stub_scores(monkeypatch, lambda w: 0.9 if w in {"pool", "exhausted."} else 0.1)
    out = compress_prose(PARAGRAPH)
    assert "pool" in out
    assert "probably" not in out
    assert len(out) < len(PARAGRAPH)


def test_the_threshold_is_configurable(monkeypatch):
    """A higher threshold keeps strictly less."""
    scores = {"d": 0.9, "p": 0.35}
    _stub_scores(monkeypatch, lambda w: scores.get(w[0], 0.1))
    kept_at_default = compress_prose(PARAGRAPH, threshold=0.30)
    kept_at_high = compress_prose(PARAGRAPH, threshold=0.50)
    assert "probably" in kept_at_default
    assert "probably" not in kept_at_high
    assert len(kept_at_high) < len(kept_at_default) < len(PARAGRAPH)


def test_a_threshold_of_zero_keeps_everything(monkeypatch):
    _stub_scores(monkeypatch, lambda _w: 0.0)
    assert compress_prose(PARAGRAPH, threshold=0.0).split() == PARAGRAPH.split()


# --------------------------------------------------------------------------
# The correctness floor: protected spans survive a hostile scorer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the failure happened in /Users/dev/app/loader.py during startup",
        "please retry against https://api.klio.tech/v1/recall when it settles",
        "the value of MAX_POOL_SIZE was much lower than anybody expected",
        "we called parse_config(path, strict=True) before the loop started",
        "upgrade to 1.2.3 or later and then try the whole thing once more",
        "there were 4096 rows and the limit was set somewhere near 4000",
        "look at `loader.load()` when you get a chance later this week",
    ],
)
def test_protected_spans_survive_a_scorer_that_wants_to_drop_everything(
    monkeypatch, text
):
    """The scorer votes 0.0 on every token. The protected spans still survive.

    This is the property that makes a lossy compressor safe to ship: a stack
    trace, a path or a signature cannot be rewritten no matter what the model
    thinks of it.
    """
    _stub_scores(monkeypatch, lambda _w: 0.0)
    out = compress_prose(text)
    protected_span = {
        "/Users/dev/app/loader.py": "/Users/dev/app/loader.py",
        "https://api.klio.tech/v1/recall": "https://api.klio.tech/v1/recall",
        "MAX_POOL_SIZE": "MAX_POOL_SIZE",
        "parse_config(path, strict=True)": "parse_config(path, strict=True)",
        "1.2.3": "1.2.3",
        "4096": "4096",
        "`loader.load()`": "`loader.load()`",
    }
    for span in protected_span.values():
        if span in text:
            assert span in out, f"{span!r} was lost"


def test_a_stack_trace_survives_a_hostile_scorer(monkeypatch):
    _stub_scores(monkeypatch, lambda _w: 0.0)
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/Users/dev/app/main.py", line 42, in handler\n'
        "    return loader.load(path)\n"
        "ValueError: empty\n"
    )
    out = compress_prose(trace)
    assert "/Users/dev/app/main.py" in out
    assert "loader.load(path)" in out
    assert "42" in out


def test_line_structure_is_preserved_where_it_carries_meaning(monkeypatch):
    _stub_scores(monkeypatch, lambda w: 0.9 if w.isupper() else 0.0)
    logs = "INFO starting up\nWARN pool nearly full\nFATAL pool exhausted\n"
    out = compress_prose(logs)
    assert "INFO" in out and "WARN" in out and "FATAL" in out
    assert out.count("\n") >= 2


# --------------------------------------------------------------------------
# The real model, when it is installed
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not prose_module.model_available(),
    reason="kompress weights / onnxruntime are not installed",
)
def test_the_real_model_compresses_and_keeps_protected_spans():
    text = (PARAGRAPH + " ") * 8 + "The path was /var/log/klio/app.log throughout."
    out = compress_prose(text)
    assert "/var/log/klio/app.log" in out
    assert len(out) < len(text)
