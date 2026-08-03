"""Prose compression: keep the subwords the model says carry the meaning.

The model is `chopratejas/kompress-v2-base` (Apache-2.0, ONNX weights, runs on
CPU). It emits one keep-probability per subword; anything at or above the
threshold survives. Attribution for the upstream weights lives in the repo's
``NOTICE`` file, which is a licence condition and not decoration.

**Default threshold 0.30.** The published operating points are 0.30 → 8%
dropped at 0.994 must-keep recall, 0.50 → 18% dropped at 0.974. A first
release that silently drops a needed line destroys trust permanently; one that
saves 8% instead of 18% merely under-delivers. Callers who want more can ask
for it.

**Two hard guarantees, in this order.**

1. *Nothing technical is altered.* The text is run through
   :mod:`shared.compression.protect` first, and any subword overlapping a
   protected span is kept regardless of its score. A path, a URL, a signature
   or a stack-trace frame cannot be dropped by a model that has never been
   told what those are worth.
2. *It never raises and never truncates.* Missing weights, a missing ONNX
   runtime, a failed download, an oversized input — every one of them returns
   the input unchanged. A compressor is an optimisation; it is never the
   reason a request fails.

**Weights are never downloaded inside a request.** Blocking a call on a
274 MB fetch is worse than not compressing at all, so the loader only uses
weights already on disk unless ``KLIO_PROSE_ALLOW_DOWNLOAD`` is set. An
operator who wants the model warms it deliberately.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

from shared.compression.protect import (
    SENTINEL_CLOSE,
    SENTINEL_OPEN,
    protect,
    restore,
)

logger = logging.getLogger("klio.compression.prose")

# The conservative operating point from the model card. See the module
# docstring for why this and not the model's own 0.50 default.
DEFAULT_THRESHOLD = 0.30

DEFAULT_MODEL_ID = "chopratejas/kompress-v2-base"
DEFAULT_ONNX_FILE = "onnx/kompress-int8-wo.onnx"

# ModernBERT's native context. Longer input is passed through rather than
# truncated: half a document compressed and half silently missing is the one
# failure mode this whole module exists to prevent.
MAX_MODEL_TOKENS = 8192

# Sentinels emitted by protect(); any subword touching one is force-kept.
#
# The span has to cover the WHOLE sentinel -- both delimiters and everything
# between them -- because ``restore()`` matches ``<open>N<close>`` and nothing
# else. Drop one delimiter and the sentinel stops matching: a protected URL
# comes back as a bare integer beside an orphaned private-use character.
# Silent corruption of exactly the content this module promised not to touch.
#
# The delimiters are also the part of a sentinel most likely to be dropped,
# not the least: a byte-level BPE tokeniser has no vocabulary entry for
# U+E000/U+E001 and emits them as their own unknown-byte fragments, which is
# the sort of token a keep/drop classifier scores lowest.
#
# A regex cannot express this, which is the bug this scan replaces.
# ``protect()`` applies its patterns in sequence, so sentinels NEST -- a path
# stashed first has its index re-stashed by the number pattern, giving
# ``<open><open>1<close><close>``. ``SENTINEL_RE`` matches only the innermost
# pair, so a regex scan leaves every outer delimiter unprotected and
# droppable. The scan below pairs opens with closes and yields a span for
# every pair at every depth, so nesting cannot hide a delimiter.
#
# Unbalanced delimiters -- which a pattern that matches across a sentinel
# boundary can produce -- are force-kept individually. Keeping a character
# that did not need keeping costs a byte; dropping one costs the content.


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Every sentinel span in ``text``, at every nesting depth.

    Spans may overlap (an inner pair sits inside its outer pair); the caller
    only asks whether a token touches any of them, so that is harmless.
    """
    spans: list[tuple[int, int]] = []
    open_stack: list[int] = []
    for index, char in enumerate(text):
        if char == SENTINEL_OPEN:
            open_stack.append(index)
        elif char == SENTINEL_CLOSE:
            if open_stack:
                spans.append((open_stack.pop(), index + 1))
            else:
                # A close with no open. Still un-droppable: restore needs it.
                spans.append((index, index + 1))
    spans.extend((start, start + 1) for start in open_stack)
    return spans


_lock = threading.Lock()
_model: Optional["_ProseModel"] = None
_load_attempted = False


@dataclass(frozen=True)
class _ProseModel:
    session: Any
    tokenizer: Any


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _resolve_weights() -> Optional[tuple[str, str]]:
    """Return local paths to (onnx file, tokenizer.json), or ``None``.

    Prefers an explicit directory, then the local Hugging Face cache. A
    network fetch happens only when explicitly permitted.
    """
    explicit = os.environ.get("KLIO_PROSE_MODEL_PATH")
    if explicit:
        onnx_path = os.path.join(explicit, _env("KLIO_PROSE_ONNX_FILE", "model.onnx"))
        tokenizer_path = os.path.join(explicit, "tokenizer.json")
        if os.path.exists(onnx_path) and os.path.exists(tokenizer_path):
            return onnx_path, tokenizer_path
        logger.warning("KLIO_PROSE_MODEL_PATH=%s has no usable weights", explicit)
        return None

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        logger.warning("huggingface_hub is unavailable; prose compression is off")
        return None

    model_id = _env("KLIO_PROSE_MODEL_ID", DEFAULT_MODEL_ID)
    onnx_file = _env("KLIO_PROSE_ONNX_FILE", DEFAULT_ONNX_FILE)
    allow_download = os.environ.get("KLIO_PROSE_ALLOW_DOWNLOAD") == "1"

    try:
        onnx_path = hf_hub_download(
            model_id, onnx_file, local_files_only=not allow_download
        )
        tokenizer_path = hf_hub_download(
            model_id, "tokenizer.json", local_files_only=not allow_download
        )
    except Exception:
        logger.warning(
            "prose weights for %s are not available locally; prose compression "
            "is off (set KLIO_PROSE_ALLOW_DOWNLOAD=1 to fetch them, but do it "
            "out of band — never on a request path)",
            model_id,
        )
        return None
    return onnx_path, tokenizer_path


def _load_model() -> Optional[_ProseModel]:
    """Build the ONNX session and tokenizer, or return ``None``."""
    weights = _resolve_weights()
    if weights is None:
        return None
    onnx_path, tokenizer_path = weights
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except Exception:
        logger.warning(
            "onnxruntime/tokenizers are not installed; prose compression is off"
        )
        return None
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        logger.warning("prose model failed to load; prose compression is off")
        return None
    logger.info("prose model loaded from %s", onnx_path)
    return _ProseModel(session=session, tokenizer=tokenizer)


def _get_model() -> Optional[_ProseModel]:
    """Load once. A failed load is remembered, so it is logged once, not once
    per request."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    with _lock:
        if not _load_attempted:
            _model = _load_model()
            _load_attempted = True
    return _model


def reset_model_cache() -> None:
    """Forget a previous load attempt. For tests and for a deliberate warm-up."""
    global _model, _load_attempted
    with _lock:
        _model = None
        _load_attempted = False


def model_available() -> bool:
    """Whether the prose model is loadable right now, without raising."""
    try:
        return _get_model() is not None
    except Exception:
        return False


def _score_tokens(text: str) -> Optional[list[tuple[int, int, float]]]:
    """Return ``(start, end, keep_probability)`` per subword, or ``None``.

    ``None`` means "no opinion" — no model, or input the model cannot see in
    one pass — and the caller passes the text through untouched. This is the
    seam the tests stub, so the thresholding and restore logic is exercised
    without the weights.
    """
    model = _get_model()
    if model is None:
        return None

    import numpy as np

    encoded = model.tokenizer.encode(text)
    if not encoded.ids:
        return None
    if len(encoded.ids) > MAX_MODEL_TOKENS:
        logger.info(
            "input is %d subwords, over the %d-token context; passing through",
            len(encoded.ids),
            MAX_MODEL_TOKENS,
        )
        return None

    scores = model.session.run(
        None,
        {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
        },
    )[0][0]
    return [
        (start, end, float(scores[index]))
        for index, (start, end) in enumerate(encoded.offsets)
    ]


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _rebuild(text: str, kept: list[tuple[int, int]]) -> str:
    """Splice the kept spans back into readable text.

    Whitespace between two adjacent survivors is preserved verbatim, so line
    structure — which in a log file *is* the information — survives. Where
    something was dropped, one separator stands in for it: a newline if the
    gap spanned lines, otherwise a space.
    """
    out: list[str] = []
    cursor = 0
    for start, end in kept:
        gap = text[cursor:start]
        if gap:
            if not gap.strip():
                out.append(gap)
            else:
                out.append("\n" if "\n" in gap else " ")
        out.append(text[start:end])
        cursor = end
    tail = text[cursor:]
    if tail and not tail.strip():
        out.append(tail)
    return "".join(out)


def compress_prose(text: str, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return ``text`` with low-value subwords dropped.

    Fail-open in every branch: the input comes back unchanged rather than
    partially processed, and no exception escapes.

    Args:
        text: prose, logs or documentation.
        threshold: minimum keep-probability. Lower keeps more.
    """
    if not text:
        return text
    try:
        protected, segments = protect(text)
        scored = _score_tokens(protected)
        if not scored:
            return text

        must_keep = _protected_spans(protected)
        kept: list[tuple[int, int]] = []
        for start, end, score in scored:
            if end <= start:
                # Special tokens ([CLS]/[SEP]) carry a zero-length span.
                continue
            if score >= threshold or _overlaps(start, end, must_keep):
                if kept and start <= kept[-1][1]:
                    kept[-1] = (kept[-1][0], max(kept[-1][1], end))
                else:
                    kept.append((start, end))

        if not kept:
            # The model wanted everything gone. Returning nothing is never the
            # right answer; returning the input is.
            return text

        rebuilt = _rebuild(protected, kept)
        return restore(rebuilt, segments)
    except Exception:
        logger.warning("prose compression failed; passing through", exc_info=True)
        return text
