"""Protection is the correctness floor of the whole compression system.

If a compressor mangles a stack trace, a file path or a function name, it is
worse than no compressor: the model reads something that looks right and is
not. Every compressor runs inside protect/restore, so these tests guard every
compressor that will ever exist.
"""

import pytest

from shared.compression.protect import (
    SENTINEL_CLOSE,
    SENTINEL_OPEN,
    protect,
    restore,
)


def _roundtrip(text: str) -> str:
    protected, segments = protect(text)
    return restore(protected, segments)


def _outermost_sentinel_tokens(protected: str) -> list[str]:
    """Every top-level sentinel token, delimiters included.

    Sentinels NEST: a later pattern re-matches the decimal index of an earlier
    sentinel, so ``loader.load(path)`` protects to
    ``⟨open⟩⟨open⟩1⟨close⟩⟨close⟩``. A flat ``⟨open⟩\\d+⟨close⟩`` regex finds
    only the inner token and silently loses the outer one, which is why this
    is a depth-tracking scan and not a regex.
    """
    tokens: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(protected):
        if char == SENTINEL_OPEN:
            if depth == 0:
                start = index
            depth += 1
        elif char == SENTINEL_CLOSE and depth:
            depth -= 1
            if depth == 0:
                tokens.append(protected[start : index + 1])
    return tokens


def _survives_annihilation(text: str) -> str:
    """What is left after the worst compressor imaginable has run.

    A roundtrip assertion (``restore(*protect(x)) == x``) passes even when
    ``protect`` protects NOTHING — it only exercises ``restore``. The contract
    that actually matters is *which spans* a compressor cannot touch, so this
    helper models a compressor that deletes every character it is permitted
    to: everything outside a sentinel is thrown away, and only what protection
    hid comes back.

    Joined on a space because the dropped prose between two adjacent
    protected spans is exactly what a real compressor removes.
    """
    protected, segments = protect(text)
    annihilated = " ".join(_outermost_sentinel_tokens(protected))
    return restore(annihilated, segments)


def test_annihilation_really_does_destroy_unprotected_text():
    """Positive control for the helper above.

    If ``_survives_annihilation`` returned its input unchanged, every
    assertion built on it would be vacuous. Prose that protection does not
    claim must come back empty.
    """
    assert _survives_annihilation("the quick brown fox") == ""
    assert _survives_annihilation("") == ""


#: (text, the spans protection must make untouchable). These are the contract:
#: not "a roundtrip is lossless" but "*this* span cannot be rewritten".
PROTECTED_SPAN_CASES = [
    ("See `foo.bar()` for details", ["`foo.bar()`"]),
    (
        "```python\ndef f(x):\n    return x\n```",
        ["```python\ndef f(x):\n    return x\n```"],
    ),
    (
        "Fetch https://example.com/a/b?c=1 then retry",
        ["https://example.com/a/b?c=1"],
    ),
    ("Open /Users/dev/src/main.py line 42", ["/Users/dev/src/main.py", "42"]),
    ("Set MAX_RETRY_COUNT to 5", ["MAX_RETRY_COUNT", "5"]),
    (
        "Call parse_config(path, strict=True) first",
        ["parse_config(path, strict=True)"],
    ),
    ("Upgrade to 1.2.3 or later", ["1.2.3"]),
    # The colon is prose and may be dropped; the address and the port are not.
    ("Error: ECONNREFUSED 127.0.0.1:5432", ["127.0.0.1", "5432"]),
    # Regression: a quoted path inside a stack-trace frame. The original
    # whitespace-only left boundary missed this shape entirely.
    ('File "/Users/dev/app/main.py", line 42', ["/Users/dev/app/main.py"]),
    # Regression: the trailing call must be matched BALANCED. A `\(?\)?`
    # split this into a protected `loader.load(` and a droppable `path)`.
    ("then loader.load(path) runs", ["loader.load(path)"]),
]


@pytest.mark.parametrize("text", [case[0] for case in PROTECTED_SPAN_CASES])
def test_protected_content_survives_a_roundtrip_byte_for_byte(text):
    assert _roundtrip(text) == text


@pytest.mark.parametrize("text,spans", PROTECTED_SPAN_CASES)
def test_the_spans_that_must_be_protected_are_hidden_from_the_compressor(text, spans):
    """``protect`` must remove the span from what a compressor can see.

    A compressor cannot rewrite what is not in its input. If the span is still
    present in the protected text, protection did not happen — no matter how
    faithfully ``restore`` behaves afterwards.
    """
    protected, _ = protect(text)
    for span in spans:
        assert span not in protected, f"{span!r} was left exposed to the compressor"


@pytest.mark.parametrize("text,spans", PROTECTED_SPAN_CASES)
def test_the_spans_that_must_be_protected_survive_total_annihilation(text, spans):
    """The contract, stated positively: these spans come back verbatim.

    Even against a compressor that deletes everything it is allowed to.
    """
    survived = _survives_annihilation(text)
    for span in spans:
        assert span in survived, f"{span!r} did not survive compression"


def test_a_balanced_call_is_captured_whole_and_never_split():
    """``loader.load(path)`` is one span, not ``loader.load(`` plus leftovers.

    Half a signature restored around a dropped ``path)`` is a syntax error
    that reads as correct code, which is the precise failure protection
    exists to prevent.
    """
    _, segments = protect("then loader.load(path) runs")
    assert "loader.load(path)" in segments
    assert not any(seg.count("(") != seg.count(")") for seg in segments), (
        f"a segment has unbalanced parens: {segments!r}"
    )


def test_protection_survives_an_aggressive_transform():
    """The real risk: a compressor mutates the protected region.

    protect() must hide the content well enough that a transform which
    uppercases and strips everything cannot touch it.
    """
    text = "Run `git rebase -i HEAD~3` in /tmp/repo"
    protected, segments = protect(text)
    mangled = protected.upper().replace("THE", "")
    out = restore(mangled, segments)
    assert "`git rebase -i HEAD~3`" in out
    assert "/tmp/repo" in out


def test_nested_patterns_restore_completely():
    """A path inside a code fence is matched by two patterns.

    A single restore pass leaves the inner sentinel unresolved; the restore
    must iterate until no sentinels remain.
    """
    text = "```\nopen('/etc/hosts')\n```"
    assert _roundtrip(text) == text
    protected, _ = protect(text)
    assert "/etc/hosts" not in protected


def test_text_with_nothing_to_protect_is_unchanged():
    assert _roundtrip("the quick brown fox") == "the quick brown fox"


def test_empty_input():
    assert _roundtrip("") == ""


def test_sentinel_lookalikes_in_source_text_are_not_confused():
    """Input containing something shaped like a sentinel must not corrupt."""
    text = "counts were 1 2 3 and then 4"
    assert _roundtrip(text) == text
