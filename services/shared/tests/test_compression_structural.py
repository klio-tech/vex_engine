"""The structural compressor is lossless by contract.

It removes only redundancy: nulls, values identical across every row, and
verbose representations of the same information. Every test here asserts the
information survives — a test that only asserted "it got smaller" would pass
against a compressor that simply deleted data.
"""

import json
import random

from shared.compression.structural import (
    compress_json_text,
    compress_payload,
    expand_rows,
)

RECALL = {
    "memories": [
        {
            "id": "80590c00-cb5f-45c8-b813-952054a10b11",
            "memory_type": "fact",
            "content": "Team uses pgvector for recall",
            "scope": "org",
            "confidence": None,
            "provenance": "EXTRACTED",
            "relevance": 0.866,
            "space_id": None,
            "project_id": "93e3f7df-788a-4de0-9c3f-654e1259ce00",
            "project": None,
            "created_at": "2026-07-27T16:02:38.526112+00:00",
            "metadata": {"source": "hook-transcript"},
        },
        {
            "id": "9151f0aa-1b2c-4d5e-8f90-112233445566",
            "memory_type": "fact",
            "content": "Threshold is 0.85",
            "scope": "org",
            "confidence": None,
            "provenance": "EXTRACTED",
            "relevance": 0.81,
            "space_id": None,
            "project_id": "93e3f7df-788a-4de0-9c3f-654e1259ce00",
            "project": None,
            "created_at": "2026-07-27T17:00:00.000000+00:00",
            "metadata": {"source": "hook-transcript"},
        },
    ],
    "count": 2,
}


def test_every_content_string_survives_verbatim():
    """The one thing that must never be lost."""
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    for m in RECALL["memories"]:
        assert m["content"] in blob


def test_every_id_survives_verbatim():
    """Ids are how an agent calls forget/share. Losing one breaks those."""
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    for m in RECALL["memories"]:
        assert m["id"] in blob


def test_null_fields_are_dropped():
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    assert '"confidence": null' not in blob
    assert '"space_id": null' not in blob


def test_values_identical_across_all_rows_are_hoisted_not_repeated():
    """scope, provenance and project_id are the same on every row here."""
    out, _ = compress_payload(RECALL)
    rows = out["memories"]
    assert all("scope" not in r for r in rows), "constant field left on rows"
    assert out.get("common", {}).get("scope") == "org"


def test_a_field_that_differs_stays_on_the_row():
    """Hoisting must only apply to genuinely constant values."""
    payload = json.loads(json.dumps(RECALL))
    payload["memories"][1]["scope"] = "private"
    out, _ = compress_payload(payload)
    rows = out["memories"]
    assert {r.get("scope") for r in rows} == {"org", "private"}
    assert "scope" not in out.get("common", {})


def test_it_actually_saves_something():
    before = len(json.dumps(RECALL))
    out, stats = compress_payload(RECALL)
    after = len(json.dumps(out))
    assert after < before
    assert stats.before_chars == before
    assert stats.after_chars == after


def test_single_row_payload_does_not_hoist_everything():
    """With one row every field is trivially 'constant'. Hoisting them all
    just moves bytes around and makes the shape harder to read."""
    payload = {"memories": [RECALL["memories"][0]], "count": 1}
    out, _ = compress_payload(payload)
    assert out["memories"][0]["content"] == RECALL["memories"][0]["content"]


def test_empty_payload_is_safe():
    out, stats = compress_payload({"memories": [], "count": 0})
    assert out["memories"] == []


def test_malformed_input_returns_unchanged():
    """Fail-open: compression must never raise into a request path."""
    weird = {"not": "a recall payload"}
    out, _ = compress_payload(weird)
    assert out == weird


def test_identical_content_on_two_rows_is_never_hoisted():
    """Two memories can legitimately hold the same text.

    Hoisting is keyed on "every row agrees", so identical content would be
    lifted into `common` and erased from the rows — a caller iterating
    memories would see entries with no content at all. `id` has the same
    exposure. Mutation-checked: removing "content" from the never-hoist set
    made no test fail before this one existed, because every fixture happened
    to have distinct content.
    """
    payload = {
        "memories": [
            {"id": "a", "content": "deploys run on Railway", "scope": "org"},
            {"id": "b", "content": "deploys run on Railway", "scope": "org"},
        ],
        "count": 2,
    }
    out, _ = compress_payload(payload)
    assert "content" not in out.get("common", {}), "content must never be hoisted"
    assert "id" not in out.get("common", {}), "id must never be hoisted"
    for row in out["memories"]:
        assert row["content"] == "deploys run on Railway"
        assert row["id"] in {"a", "b"}


# --------------------------------------------------------------------------
# The generic path: compress_json_text / expand_rows are exact inverses
# --------------------------------------------------------------------------
#
# The generic path advertises itself as lossless, and the router acts on that:
# a lossless result gets no handle and no cache row, so anything it destroys
# is destroyed permanently. The envelope keys it writes -- `common` and
# `_klio` -- are ordinary JSON key names that a caller's own payload is
# entitled to use. "Nobody would call a field that" is not a losslessness
# argument.


def _round_trips(payload: dict) -> bool:
    return expand_rows(json.loads(compress_json_text(json.dumps(payload)))) == payload


def _rows(count: int, **extra) -> list[dict]:
    return [
        {"id": f"id-{i}", "content": f"row {i}", "scope": "private", **extra}
        for i in range(count)
    ]


def test_a_payload_that_already_has_a_common_key_is_not_destroyed():
    """`common` is a perfectly ordinary field name.

    Writing the hoist envelope over it deletes the caller's data, and
    `expand_rows` then strips the key entirely -- so the value is gone from
    both halves of a round trip that claims to be lossless.
    """
    payload = {
        "common": {"tenant": "acme", "region": "eu-west-1"},
        "rows": _rows(4, kind="fact", tag=None),
    }
    out = json.loads(compress_json_text(json.dumps(payload)))
    assert _round_trips(payload), f"round trip lost data: {out!r}"


def test_a_payload_that_already_has_a_klio_key_is_not_destroyed():
    """Same for the metadata key: `_klio` belongs to whoever wrote it first."""
    payload = {
        "_klio": {"note": "this is the caller's own field"},
        "rows": _rows(4, kind="fact", tag=None),
    }
    out = json.loads(compress_json_text(json.dumps(payload)))
    assert _round_trips(payload), f"round trip lost data: {out!r}"


def test_a_payload_with_both_envelope_keys_is_not_destroyed():
    payload = {
        "common": ["not even a dict"],
        "_klio": 17,
        "rows": _rows(5, kind="fact", tag=None),
    }
    assert _round_trips(payload)


def test_the_generic_path_round_trips_over_generated_payloads():
    """Property, not three examples.

    Payloads are generated across the axes the reductions key on -- constant
    fields, nulls present on every row or only some, envelope-key collisions,
    row counts either side of the hoist floor -- and every one must survive a
    compress/expand cycle byte-identically.
    """
    rng = random.Random(20260802)
    optional_extras = [
        {},
        {"common": {"tenant": "acme"}},
        {"common": "a plain string"},
        {"_klio": {"rows_key": "not ours", "null_keys": ["nope"]}},
        {"_klio": None},
        {"common": {"a": 1}, "_klio": [1, 2, 3]},
    ]
    failures = []
    for _ in range(500):
        count = rng.randint(0, 6)
        constant = rng.choice([None, "org", 7])
        rows = []
        for i in range(count):
            row = {"id": f"id-{i}", "content": f"row {i}"}
            if rng.random() < 0.8:
                row["scope"] = constant
            if rng.random() < 0.5:
                row["tag"] = None if rng.random() < 0.5 else f"tag-{i}"
            if rng.random() < 0.3:
                row["nested"] = {"a": [1, 2], "b": None}
            rows.append(row)
        payload = {"rows": rows, "count": count, **rng.choice(optional_extras)}
        if not _round_trips(payload):
            failures.append(payload)
    assert not failures, f"{len(failures)} payloads did not round trip: {failures[:2]}"


def test_the_recall_path_does_not_overwrite_a_caller_common_key():
    """`compress_payload` writes the same envelope and needs the same guard.

    This path has no inverse at all -- the recall envelope is returned to the
    caller as-is -- so overwriting `common` here is not even theoretically
    recoverable.
    """
    payload = {
        "common": {"tenant": "acme"},
        "memories": [
            {"id": "a", "content": "one", "scope": "org"},
            {"id": "b", "content": "two", "scope": "org"},
        ],
        "count": 2,
    }
    out, _ = compress_payload(payload)
    assert out["common"] == {"tenant": "acme"}
    for row in out["memories"]:
        assert row["scope"] == "org", "nothing may be hoisted with no envelope free"
