"""Prose carried in metadata is screened, not just ``content``.

``write_scoped_memory`` has always redacted ``content`` before dedup-embedding
and storage. It did not screen ``metadata``, and the ``decide`` tool files its
caller-supplied ``rationale`` there — so a credential pasted into a rationale
was stored in the clear, embedded nowhere but readable by anyone who could read
the row, and forwarded to the curator's LLM as synthesis fuel.

Only declared prose-bearing keys are screened. Blanket-screening every metadata
value would be worse than the bug: checksums, storage URIs and ids are
high-entropy by construction, and the entropy gate would corrupt them.
"""

import pytest
from shared import brain


def test_the_screened_keys_are_a_declared_set():
    """A frozenset, so a future caller adding a prose field must opt in."""
    assert isinstance(brain.REDACTED_METADATA_KEYS, frozenset)
    assert "rationale" in brain.REDACTED_METADATA_KEYS


def test_a_secret_in_a_rationale_is_redacted():
    """The defect: `decide` files its rationale into metadata unscreened."""
    metadata = {"source": "mcp", "rationale": "we rotated to AKIAIOSFODNN7EXAMPLE"}

    cleaned, kinds = brain.screen_metadata(metadata)

    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned["rationale"]
    assert kinds, "a redaction happened but no kind was reported"


def test_clean_prose_is_left_alone():
    """Positive control: the screener is not simply blanking the field."""
    metadata = {"source": "mcp", "rationale": "chose Postgres over Mongo"}

    cleaned, kinds = brain.screen_metadata(metadata)

    assert cleaned["rationale"] == "chose Postgres over Mongo"
    assert kinds == []


def test_undeclared_keys_are_never_touched():
    """Checksums and ids are high-entropy by construction.

    Screening them would let the entropy gate corrupt structural data — a
    strictly worse outcome than the bug being fixed.
    """
    checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    metadata = {"checksum": checksum, "storage_uri": "s3://bucket/deadbeef"}

    cleaned, kinds = brain.screen_metadata(metadata)

    assert cleaned["checksum"] == checksum
    assert cleaned["storage_uri"] == "s3://bucket/deadbeef"
    assert kinds == []


def test_the_input_is_not_mutated():
    """The caller's dict is theirs; screening returns a new one."""
    metadata = {"rationale": "key AKIAIOSFODNN7EXAMPLE rotated"}
    original = dict(metadata)

    brain.screen_metadata(metadata)

    assert metadata == original, "screen_metadata mutated its argument"


@pytest.mark.parametrize("value", [None, 42, {"nested": "x"}, ["a"]])
def test_non_string_values_are_passed_through(value):
    """The redactor takes text. A non-string is not prose and is left alone."""
    cleaned, kinds = brain.screen_metadata({"rationale": value})

    assert cleaned["rationale"] == value
    assert kinds == []


def test_no_metadata_is_handled():
    """`note`, `plan` and `remember` pass no rationale at all."""
    assert brain.screen_metadata(None) == (None, [])
    assert brain.screen_metadata({}) == ({}, [])
