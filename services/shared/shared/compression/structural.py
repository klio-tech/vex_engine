"""Lossless structural compression for JSON payloads.

Removes only redundancy that can be reconstructed: fields that are null, and
fields whose value is identical on every row. Nothing that carries information
is touched, so there is no recovery contract and no cache entry — the caller
gets the same information in fewer bytes.

Measured motivation: a recall(limit=10) payload is 5,768 characters of which
1,029 is memory content. 83% is envelope.

Hoisting requires at least two rows. With a single row every field is trivially
constant, and lifting them all just relocates bytes while making the shape
harder for a model to read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("klio.compression.structural")

MIN_ROWS_TO_HOIST = 2


@dataclass
class CompressionStats:
    before_chars: int
    after_chars: int

    @property
    def saved_ratio(self) -> float:
        if self.before_chars == 0:
            return 0.0
        return 1.0 - (self.after_chars / self.before_chars)


def compress_payload(payload: Any) -> tuple[Any, CompressionStats]:
    """Compress a recall-shaped payload losslessly.

    Fail-open: anything unexpected returns the input unchanged. Compression is
    an optimisation and must never raise into a request path.
    """
    try:
        before = len(json.dumps(payload, default=str))
    except Exception:
        # Not even serialisable — nothing to measure and nothing to compress.
        logger.warning("payload is not JSON-serialisable; passing through")
        return payload, CompressionStats(before_chars=0, after_chars=0)

    try:
        out = _compress(payload)
    except Exception:
        logger.warning("structural compression failed; passing through", exc_info=True)
        out = payload

    try:
        after = len(json.dumps(out, default=str))
    except Exception:
        logger.warning("compressed payload is not serialisable; passing through")
        return payload, CompressionStats(before_chars=before, after_chars=before)

    return out, CompressionStats(before_chars=before, after_chars=after)


def _compress(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    rows = payload.get("memories")
    if not isinstance(rows, list) or not rows:
        return payload
    if not all(isinstance(r, dict) for r in rows):
        return payload

    stripped = [{k: v for k, v in row.items() if v is not None} for row in rows]

    common: dict[str, Any] = {}
    # A payload that already has a `common` key owns it. Hoisting into it
    # would delete the caller's value, and this path has no inverse at all --
    # the recall envelope goes straight back to the caller -- so the loss
    # could never be undone. Fewer bytes saved is the correct trade.
    can_hoist = COMMON_KEY not in payload
    if can_hoist and len(stripped) >= MIN_ROWS_TO_HOIST:
        candidate_keys = set(stripped[0])
        for row in stripped[1:]:
            candidate_keys &= set(row)
        for key in sorted(candidate_keys):
            if key in {"id", "content"}:
                # Never hoist identity or payload, even if two rows collide.
                continue
            values = {_hashable(row[key]) for row in stripped}
            if len(values) == 1:
                common[key] = stripped[0][key]
        if common:
            stripped = [
                {k: v for k, v in row.items() if k not in common} for row in stripped
            ]

    out = dict(payload)
    out["memories"] = stripped
    if common:
        out["common"] = common
    return out


def _hashable(value: Any) -> Any:
    """Make a value comparable for the constant check."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


# ---------------------------------------------------------------------------
# The generic path: any JSON payload, not just a recall envelope.
# ---------------------------------------------------------------------------
#
# ``compress_payload`` above is the recall integration and its wire format is
# frozen by stage 1. The router needs the same idea applied to arbitrary JSON,
# and — because it advertises the result as lossless — it needs a proven
# inverse. That is what these two functions are: ``compress_json_text`` and
# ``expand_rows`` are exact inverses, and a round-trip test says so rather
# than a comment claiming it.
#
# Two reductions, both reversible:
#
# * **Nulls are dropped**, but only for a key present on *every* row. Then an
#   absent key can only mean null, and restoring it is unambiguous. A ragged
#   payload compresses less rather than compressing wrongly.
# * **Constant fields are hoisted** into ``common``. Identity and payload
#   fields are never hoisted even if they happen to collide.
#
# The metadata needed to reverse both lives under ``_klio``. It costs a few
# dozen bytes and buys a losslessness claim that can be tested.

META_KEY = "_klio"
COMMON_KEY = "common"
NEVER_HOIST = frozenset({"id", "content"})

# Bumped if the meta shape ever changes. Its real job is to make `expand_rows`
# safe to call speculatively: without it, a caller's own `_klio` field that
# happens to look like ours would be treated as our metadata and the payload
# rewritten around it.
META_VERSION = 1

# `common` and `_klio` are ordinary JSON key names and a caller's payload is
# entitled to use them. Writing the envelope over one destroys the caller's
# value, and because the generic path reports `lossy=False` the router issues
# no handle and writes no cache row -- so the loss is permanent and there is
# nothing to recover it from. `_largest_row_list` already skips both keys when
# CHOOSING a row list, so the collision was seen; it just was not handled on
# the way out.
#
# The handling, in order of preference:
#
# * `common` collides -> rename our envelope to a key the payload does not
#   use, and record the name in the metadata so `expand_rows` knows where to
#   look. Nothing is lost and compression still happens.
# * `_klio` collides -> decline to compress this payload. The metadata is what
#   makes the reductions reversible, and there is nowhere else to put it that
#   `expand_rows` could find without metadata of its own. A payload returned
#   verbatim is a worse ratio; a payload returned wrong is a broken contract.
_ALTERNATE_COMMON_KEYS = ("_klio_common", "_klio_common_2", "_klio_common_3")


def _pick_common_key(payload: dict) -> Optional[str]:
    """A key for the hoist envelope that ``payload`` does not already use."""
    if COMMON_KEY not in payload:
        return COMMON_KEY
    for candidate in _ALTERNATE_COMMON_KEYS:
        if candidate not in payload:
            return candidate
    return None


def _largest_row_list(payload: dict) -> Optional[str]:
    """The key holding the longest list of dicts, or ``None``."""
    best: Optional[str] = None
    best_len = 0
    for key, value in payload.items():
        if key in {META_KEY, COMMON_KEY}:
            continue
        if (
            isinstance(value, list)
            and len(value) > best_len
            and all(isinstance(item, dict) for item in value)
            and value
        ):
            best, best_len = key, len(value)
    return best


def compress_json_text(text: str) -> str:
    """Compress a JSON document losslessly and re-serialise it compactly.

    Fail-open: anything that is not the shape we understand comes back
    unchanged, byte for byte.
    """
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, (dict, list)):
        return text

    try:
        if isinstance(payload, dict):
            payload = _compress_generic(payload)
        # Compact separators alone are a large, entirely safe win on the
        # pretty-printed JSON that tools actually emit.
        return json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        logger.warning(
            "generic JSON compression failed; passing through", exc_info=True
        )
        return text


def _compress_generic(payload: dict) -> dict:
    if META_KEY in payload:
        # The caller owns this key. See the note above _ALTERNATE_COMMON_KEYS.
        logger.info(
            "payload already uses %r; declining to compress rather than "
            "overwriting the caller's field",
            META_KEY,
        )
        return payload
    rows_key = _largest_row_list(payload)
    if rows_key is None:
        return payload
    rows: list[dict] = payload[rows_key]

    universal = set(rows[0])
    for row in rows[1:]:
        universal &= set(row)

    null_keys = sorted(
        key for key in universal if any(row.get(key) is None for row in rows)
    )
    stripped = [
        {k: v for k, v in row.items() if not (k in null_keys and v is None)}
        for row in rows
    ]

    common: dict[str, Any] = {}
    if len(stripped) >= MIN_ROWS_TO_HOIST:
        shared_keys = set(stripped[0])
        for row in stripped[1:]:
            shared_keys &= set(row)
        for key in sorted(shared_keys - NEVER_HOIST):
            if len({_hashable(row[key]) for row in stripped}) == 1:
                common[key] = stripped[0][key]
        if common:
            stripped = [
                {k: v for k, v in row.items() if k not in common} for row in stripped
            ]

    common_key = _pick_common_key(payload) if common else COMMON_KEY
    if common and common_key is None:
        # Every candidate envelope name is taken. Hoisting would destroy one
        # of them, so hoist nothing and keep only the null reduction.
        logger.info("no free envelope key; skipping the hoist for this payload")
        common = {}
        stripped = [
            {k: v for k, v in row.items() if not (k in null_keys and v is None)}
            for row in rows
        ]
        common_key = COMMON_KEY

    if not null_keys and not common:
        return payload

    meta: dict[str, Any] = {
        "v": META_VERSION,
        "rows_key": rows_key,
        "null_keys": null_keys,
    }
    out = dict(payload)
    out[rows_key] = stripped
    if common:
        # Recorded ALWAYS, not only when it differs from the default. Absence
        # then means "no envelope was written", which is what stops
        # `expand_rows` from mistaking a caller's own `common` field for our
        # hoist and merging it into every row.
        out[common_key] = common
        meta["common_key"] = common_key
    out[META_KEY] = meta
    return out


def expand_rows(payload: Any) -> Any:
    """Exact inverse of :func:`compress_json_text`'s reductions.

    A payload we did not compress is returned unchanged, so this is safe to
    call speculatively.
    """
    if not isinstance(payload, dict):
        return payload
    meta = payload.get(META_KEY)
    if not isinstance(meta, dict) or meta.get("v") != META_VERSION:
        # Either we did not compress this, or the `_klio` here belongs to the
        # caller. Both mean: hands off.
        return payload

    rows_key = meta.get("rows_key")
    null_keys = meta.get("null_keys") or []
    rows = payload.get(rows_key)
    if not isinstance(rows, list):
        return payload

    # Where the hoist envelope went, if there was one at all. No key recorded
    # means nothing was hoisted -- so any `common` field present in the
    # payload belongs to the caller and must be left exactly where it is.
    common_key = meta.get("common_key")
    common = payload.get(common_key) if common_key else {}
    if common is None:
        common = {}
    if not isinstance(common, dict):
        return payload

    restored = []
    for row in rows:
        merged = dict(common)
        merged.update(row)
        for key in null_keys:
            merged.setdefault(key, None)
        restored.append(merged)

    dropped = {META_KEY} | ({common_key} if common_key else set())
    out = {k: v for k, v in payload.items() if k not in dropped}
    out[rows_key] = restored
    return out
