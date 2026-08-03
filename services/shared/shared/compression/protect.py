"""Hide content no compressor may alter, then put it back.

Every compressor runs inside protect/restore. A compressor that rewrites a
file path or a function name produces output that looks correct and is not,
which is worse than no compression at all — the reader has no signal that
anything was lost.

Sentinels are a rare unicode private-use character plus an index, chosen so
that input text cannot collide with them. An earlier design using ` N ` (a
bare integer) collided with ordinary prose containing numbers.

Restore iterates: patterns nest (a path inside a code fence is matched twice),
so a single pass leaves inner sentinels unresolved. The pass count is bounded
by nesting depth, never by input size.
"""

from __future__ import annotations

import re

# U+E000 is in the Unicode private use area — it does not appear in source
# code, prose, logs or JSON in practice, so a sentinel cannot collide with
# real input the way a numeric marker can.
SENTINEL_OPEN = "\ue000"
SENTINEL_CLOSE = "\ue001"

# Public, and deliberately so: a compressor that runs inside protect/restore
# has to know where a sentinel *begins and ends* in order to keep it whole. It
# must use this pattern and not re-derive one, because a compressor that
# protects only the digits leaves both delimiters droppable \u2014 and a sentinel
# missing a delimiter no longer matches here, so `restore` silently leaves a
# bare integer where a URL used to be.
SENTINEL_RE = re.compile(f"{SENTINEL_OPEN}(\\d+){SENTINEL_CLOSE}")

MAX_RESTORE_PASSES = 8

PROTECTED_PATTERNS: list[re.Pattern] = [
    re.compile(r"```[\s\S]*?```"),  # fenced code
    re.compile(r"`[^`\n]+`"),  # inline code
    re.compile(r"\bhttps?://\S+", re.IGNORECASE),  # URLs
    # A path is bounded on the left by whitespace, the start of the text,
    # or one of the delimiters that routinely wraps one. The original
    # whitespace-only guard missed the single most common shape a path
    # appears in -- a quoted path inside a stack-trace frame,
    # `File "/Users/dev/app/main.py"` -- leaving it unprotected and so
    # droppable by the prose scorer, which is the one thing protection
    # exists to prevent.
    re.compile(r"(?:^|(?<=[\s\"\'(\[{<,=]))[\w.-]*[/\\][\w.\-/\\]+"),  # paths
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),  # CONST_CASE
    # The trailing call is matched only when it is BALANCED. The original
    # `\(?\)?` accepted a lone open paren, so `loader.load(path)` was
    # protected as `loader.load(` and the `path)` left over was dropped by
    # the scorer -- a signature silently rewritten into a syntax error.
    re.compile(r"\b\w+(?:\.\w+)+(?:\([^)\n]*\))?"),  # dotted.method()
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)\n]*\)"),  # function calls
    re.compile(r"\b\d+\.\d+\.\d+\b"),  # versions
    re.compile(r"\b\d[\d,.:]*\b"),  # any number
]


def protect(text: str) -> tuple[str, list[str]]:
    """Replace every protected span with a sentinel.

    Returns the protected text and the ordered list of original spans.
    """
    segments: list[str] = []
    working = text
    for pattern in PROTECTED_PATTERNS:

        def _stash(match: re.Match) -> str:
            index = len(segments)
            segments.append(match.group(0))
            return f"{SENTINEL_OPEN}{index}{SENTINEL_CLOSE}"

        working = pattern.sub(_stash, working)
    return working, segments


def restore(text: str, segments: list[str]) -> str:
    """Splice the original spans back in.

    Iterates because protected patterns nest. Bounded by MAX_RESTORE_PASSES,
    which grows with nesting depth and not with input size.
    """
    out = text
    for _ in range(MAX_RESTORE_PASSES):
        if not SENTINEL_RE.search(out):
            break
        out = SENTINEL_RE.sub(
            lambda m: (
                segments[int(m.group(1))]
                if int(m.group(1)) < len(segments)
                else m.group(0)
            ),
            out,
        )
    return out
