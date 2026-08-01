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
from typing import Any

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
    if len(stripped) >= MIN_ROWS_TO_HOIST:
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
