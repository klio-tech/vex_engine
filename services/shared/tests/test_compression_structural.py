"""The structural compressor is lossless by contract.

It removes only redundancy: nulls, values identical across every row, and
verbose representations of the same information. Every test here asserts the
information survives — a test that only asserted "it got smaller" would pass
against a compressor that simply deleted data.
"""

import json

from shared.compression.structural import compress_payload

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
