"""Pick a compressor for a blob, or decline.

The router is where the system's judgement lives, and both of its judgements
are about *not* compressing.

**The size floor.** Under ~500 estimated tokens there is nothing worth
winning. A parse or a model call costs real time, the absolute saving is a few
dozen tokens, and every risk of mangling something is still on the table.
Small input is passthrough, unconditionally.

**Unrecognised means passthrough, never "closest match".** A router that
guesses hands a Ruby file to the Python grammar and a CSV to a prose model.
The caller gets something that looks right and is not, with no signal that
anything happened. Each compressor confirms its own content type — the JSON
path parses, the code path requires an error-free AST with real definitions in
it — and anything that confirms nothing comes back byte-identical.

Diffs are called out explicitly. They are line-oriented text *about* code:
tree-sitter cannot parse one, and a prose model that drops a ``-`` prefix
turns a patch into a lie. They pass through.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

from shared.compression.code import compress_code, detect_language, looks_like_a_diff
from shared.compression.prose import (
    DEFAULT_THRESHOLD,
    compress_prose,
)
from shared.compression.prose import model_available as prose_model_available
from shared.compression.structural import compress_json_text

logger = logging.getLogger("klio.compression.router")

# The floor from the design spec. Below it, compressing costs more than it
# saves — in latency, in risk, and sometimes in bytes.
MIN_TOKENS_TO_COMPRESS = 500

# Four characters per token is the usual English/code approximation. It is an
# estimate and is used only for the floor and for the reported before/after,
# never for anything that has to be exact.
CHARS_PER_TOKEN = 4

# Prose needs to look like language before a language model is pointed at it.
#
# The obvious test — what fraction of the text is letters — does not work: a
# base64 blob measures 0.82 and a log file 0.63, so it ranks the blob as the
# more prose-like of the two. Function words are the discriminator that
# actually separates them. English cannot go fifty words without one; random
# base64, a CSV of identifiers and a hex dump manage it easily.
#
# Log lines sit at roughly 5-10% function words and English prose at 30-40%,
# so the floor is set well under the former.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
MIN_PROSE_WORDS = 50
MIN_FUNCTION_WORD_RATIO = 0.03
_FUNCTION_WORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for
    from by with without into over under is are was were be been being it its as
    not no we you they he she i our your their there here when while after before
    do does did done has have had can could should would will not""".split()
)


class Compressor(str, Enum):
    STRUCTURAL = "structural"
    CODE = "code"
    PROSE = "prose"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class CompressionResult:
    """What the caller is told, and everything it needs to decide what next.

    ``lossy`` is computed from the outcome, not from the compressor's
    reputation: a lossy compressor that achieved nothing returns
    ``lossy=False``, so the caller does not write a cache entry for a recovery
    nobody will ever need.

    ``compressor`` is what actually produced ``text`` -- ``passthrough`` when
    nothing did. ``selected`` is what the router CHOSE for this content, and
    ``available`` whether that choice could run here. The three together are
    what let a caller tell "there was nothing to compress" apart from "that
    compressor is not installed in this deployment": both return the content
    verbatim, and only the second is a deployment fact rather than a fact
    about the content.
    """

    text: str
    original_tokens: int
    compressed_tokens: int
    lossy: bool
    compressor: str
    selected: str = Compressor.PASSTHROUGH.value
    available: bool = True


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text) // CHARS_PER_TOKEN


def _is_json_document(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    return isinstance(parsed, (dict, list))


def _looks_like_prose(text: str) -> bool:
    words = _WORD_RE.findall(text)
    if len(words) < MIN_PROSE_WORDS:
        return False
    function_words = sum(1 for word in words if word.lower() in _FUNCTION_WORDS)
    return (function_words / len(words)) >= MIN_FUNCTION_WORD_RATIO


def select_compressor(text: str) -> Compressor:
    """Choose the compressor for ``text``. Defaults to declining."""
    if not isinstance(text, str):
        return Compressor.PASSTHROUGH
    if estimate_tokens(text) < MIN_TOKENS_TO_COMPRESS:
        return Compressor.PASSTHROUGH
    if _is_json_document(text):
        return Compressor.STRUCTURAL
    if looks_like_a_diff(text):
        return Compressor.PASSTHROUGH
    if detect_language(text) is not None:
        return Compressor.CODE
    if _looks_like_prose(text):
        return Compressor.PROSE
    return Compressor.PASSTHROUGH


def _compressor_available(choice: "Compressor") -> bool:
    """Whether ``choice`` can actually run in this deployment.

    Only prose can be selected and then be missing. `detect_language` returns
    None without the tree-sitter grammars, so CODE is never selected when it
    could not run; the prose detector is a pure-text heuristic and has no such
    coupling to its weights.
    """
    if choice is not Compressor.PROSE:
        return True
    try:
        return prose_model_available()
    except Exception:
        return False


def _passthrough(
    text: str,
    *,
    selected: "Compressor" = Compressor.PASSTHROUGH,
    available: bool = True,
) -> CompressionResult:
    tokens = estimate_tokens(text)
    return CompressionResult(
        text=text,
        original_tokens=tokens,
        compressed_tokens=tokens,
        lossy=False,
        compressor=Compressor.PASSTHROUGH.value,
        selected=selected.value,
        available=available,
    )


def compress(
    text: str,
    *,
    prose_threshold: float = DEFAULT_THRESHOLD,
    allow_lossy: bool = True,
) -> CompressionResult:
    """Route ``text`` to a compressor and report what happened.

    Fail-open at every step: a compressor that raises, a detector that raises,
    and input that is not text all return the input unchanged with
    ``lossy=False``. Compression is never allowed to raise into a request
    path.

    Args:
        text: the content to compress.
        prose_threshold: minimum keep-probability for the prose model.
        allow_lossy: when false, a compressor that would drop information is
            declined *before* it runs. Suppressing the result afterwards would
            still have paid for the model call, and would leave the decision
            somewhere the caller cannot see it.
    """
    if not isinstance(text, str) or not text:
        return _passthrough(text)

    try:
        choice = select_compressor(text)
    except Exception:
        logger.warning("compressor selection failed; passing through", exc_info=True)
        return _passthrough(text)

    available = _compressor_available(choice)

    try:
        if choice is Compressor.STRUCTURAL:
            out, lossy = compress_json_text(text), False
        elif not allow_lossy:
            return _passthrough(text, selected=choice, available=available)
        elif choice is Compressor.CODE:
            out, lossy = compress_code(text), True
        elif choice is Compressor.PROSE:
            out, lossy = compress_prose(text, threshold=prose_threshold), True
        else:
            return _passthrough(text, selected=choice, available=available)
    except Exception:
        logger.warning(
            "compressor %s raised; passing through", choice.value, exc_info=True
        )
        return _passthrough(text, selected=choice, available=available)

    if len(out) >= len(text):
        # Two cases, one guard.
        #
        # Equal: nothing changed, so nothing was lost and there is nothing to
        # recover. Reporting otherwise would cost a pointless cache write.
        #
        # Larger: the compressor made the content worse. This half is the
        # backstop for every compressor, present and future -- the guard used
        # to be `out == text`, which let an inflating compressor through and
        # reported the inflation as a saving. A caller that asked to be
        # relieved of tokens must never be handed more of them.
        if len(out) > len(text):
            logger.warning(
                "%s inflated %d chars to %d; passing through",
                choice.value,
                len(text),
                len(out),
            )
        return _passthrough(text, selected=choice, available=available)

    return CompressionResult(
        text=out,
        original_tokens=estimate_tokens(text),
        compressed_tokens=estimate_tokens(out),
        lossy=lossy,
        compressor=choice.value,
        selected=choice.value,
        available=available,
    )
