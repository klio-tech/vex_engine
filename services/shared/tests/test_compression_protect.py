"""Protection is the correctness floor of the whole compression system.

If a compressor mangles a stack trace, a file path or a function name, it is
worse than no compressor: the model reads something that looks right and is
not. Every compressor runs inside protect/restore, so these tests guard every
compressor that will ever exist.
"""

import pytest

from shared.compression.protect import protect, restore


def _roundtrip(text: str) -> str:
    protected, segments = protect(text)
    return restore(protected, segments)


@pytest.mark.parametrize(
    "text",
    [
        "See `foo.bar()` for details",
        "```python\ndef f(x):\n    return x\n```",
        "Fetch https://example.com/a/b?c=1 then retry",
        "Open /Users/dev/src/main.py line 42",
        "Set MAX_RETRY_COUNT to 5",
        "Call parse_config(path, strict=True) first",
        "Upgrade to 1.2.3 or later",
        "Error: ECONNREFUSED 127.0.0.1:5432",
    ],
)
def test_protected_content_survives_a_roundtrip_byte_for_byte(text):
    assert _roundtrip(text) == text


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
