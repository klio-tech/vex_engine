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

import random
import re

import pytest

from shared.compression import prose as prose_module
from shared.compression import protect as protect_module
from shared.compression.prose import DEFAULT_THRESHOLD, compress_prose
from shared.compression.protect import protect

_OPEN = protect_module.SENTINEL_OPEN
_CLOSE = protect_module.SENTINEL_CLOSE

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


# The stub tokeniser splits a sentinel the way a real byte-level BPE
# tokeniser does.
#
# This used to be `\S+`, which meant a single whitespace-delimited token always
# swallowed a whole ` N ` sentinel — delimiters and digits
# together. Under that tokenisation the force-keep always fired and every
# "scorer wants to drop everything" case below passed vacuously, which is why
# none of them caught the delimiters being droppable.
#
# ModernBERT-family tokenisers are byte-level: U+E000 and U+E001 are not in
# their learned vocabulary and come back as their own unknown-byte fragments,
# separate from the digits between them. So the stub emits the delimiters as
# standalone tokens, and digit runs separately again — the worst case a real
# tokeniser can hand the compressor.
_SUBWORD_RE = re.compile(rf"[{_OPEN}{_CLOSE}]|\d+|[^\s{_OPEN}{_CLOSE}]+")


def _stub_scores(monkeypatch, decide):
    """Install a fake token scorer.

    ``decide`` receives the token text and returns a keep probability. The
    real model's contract is a list of ``(start, end, score)`` triples over the
    input, and that is what we fake — over the subword split described above.
    """

    def _score(text: str):
        return [
            (m.start(), m.end(), decide(m.group(0))) for m in _SUBWORD_RE.finditer(text)
        ]

    monkeypatch.setattr(prose_module, "_score_tokens", _score)


def _stub_scores_dropping_offset(monkeypatch, offset: int):
    """Score every token 1.0 except the one covering ``offset``, which gets 0.0.

    ``offset`` is an index into the *protected* text, so a test can aim at one
    specific sentinel delimiter and nothing else.
    """

    def _score(text: str):
        return [
            (m.start(), m.end(), 0.0 if m.start() <= offset < m.end() else 1.0)
            for m in _SUBWORD_RE.finditer(text)
        ]

    monkeypatch.setattr(prose_module, "_score_tokens", _score)


def _sentinel_offsets(text: str) -> list[int]:
    """Every delimiter character position in ``protect(text)``."""
    protected, _segments = protect(text)
    return [i for i, ch in enumerate(protected) if ch in (_OPEN, _CLOSE)]


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


# --------------------------------------------------------------------------
# The sentinel is atomic
# --------------------------------------------------------------------------
#
# `protect()` writes ` N `. `restore()` only recognises that
# whole shape. If the force-keep span covers the digits alone, either
# delimiter can be dropped independently — and then the protected span the
# whole mechanism exists to defend comes back as a bare integer with an
# orphaned private-use character next to it. Silent corruption of exactly the
# content that was declared un-droppable.
#
# So the property is not "the digits survive", it is "the sentinel survives
# whole", and it is asserted by dropping each delimiter on its own.

_SENTINEL_CASES = [
    "the failure happened in /Users/dev/app/loader.py during startup",
    "please retry against https://api.klio.tech/v1/recall when it settles",
    "the value of MAX_POOL_SIZE was much lower than anybody expected",
    "we called parse_config(path, strict=True) before the loop started",
    "upgrade to 1.2.3 or later and then try the whole thing once more",
    "there were 4096 rows and the limit was set somewhere near 4000",
    "look at `loader.load()` when you get a chance later this week",
    "see ```python\nload(cfg)\n``` and then /etc/klio/config.yaml as well",
]


@pytest.mark.parametrize("text", _SENTINEL_CASES)
def test_dropping_either_sentinel_delimiter_still_round_trips(monkeypatch, text):
    """Drop each delimiter character individually; the output stays byte-exact.

    Every other token scores 1.0, so the only thing the scorer asks to remove
    is one delimiter. A sentinel that is atomic survives it; one that protects
    only its digits does not.
    """
    offsets = _sentinel_offsets(text)
    assert offsets, "the fixture is meant to contain protected spans"
    for offset in offsets:
        _stub_scores_dropping_offset(monkeypatch, offset)
        assert compress_prose(text) == text, (
            f"dropping the delimiter at offset {offset} corrupted the round trip"
        )


def _generated_payloads(count: int = 200) -> list[tuple[str, list[str]]]:
    """Deterministic ``(text, injected_fragments)`` pairs.

    The fragments are the things a reader would call "the protected content".
    ``protect``'s own ``segments`` list is deliberately NOT the assertion
    target: patterns nest, so it also contains the sentinel indices that later
    patterns re-stashed, and those are machinery rather than content.
    """
    protected_fragments = [
        "/var/log/klio/app.log",
        "https://example.test/a/b?c=1",
        "MAX_RETRY_COUNT",
        "loader.load()",
        "handler(request, timeout=30)",
        "2.11.4",
        "40961",
        "`inline.code()`",
        "```\nraw block\n```",
    ]
    filler = (
        "the deployment failed because the pool was exhausted and we should "
        "probably look into it before the next release goes out to anyone"
    ).split()
    rng = random.Random(20260802)
    payloads = []
    for _ in range(count):
        parts: list[str] = []
        injected: list[str] = []
        for _ in range(rng.randint(2, 6)):
            parts.append(" ".join(rng.sample(filler, rng.randint(2, 6))))
            fragment = rng.choice(protected_fragments)
            injected.append(fragment)
            parts.append(fragment)
        parts.append(" ".join(rng.sample(filler, 3)))
        payloads.append((" ".join(parts), injected))
    return payloads


@pytest.mark.parametrize("hostile_score", [0.0, 0.1, 0.29])
def test_no_sentinel_character_ever_reaches_the_output(monkeypatch, hostile_score):
    """Over generated payloads, a sub-threshold scorer never leaks a delimiter.

    An orphaned U+E000/U+E001 in the output is the visible signature of a
    half-dropped sentinel, so this is the corruption assertion that does not
    depend on knowing which span was hit.
    """
    _stub_scores(monkeypatch, lambda _w: hostile_score)
    for text, _injected in _generated_payloads():
        out = compress_prose(text)
        assert _OPEN not in out and _CLOSE not in out, f"orphan sentinel in {out!r}"


def test_every_protected_span_survives_a_hostile_scorer_over_generated_payloads(
    monkeypatch,
):
    """The full property: technical content comes back byte-for-byte."""
    _stub_scores(monkeypatch, lambda _w: 0.0)
    for text, injected in _generated_payloads():
        out = compress_prose(text)
        for fragment in injected:
            assert fragment in out, f"{fragment!r} was lost from {out!r}"


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
