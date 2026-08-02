"""AST-aware compression for source code.

What survives: imports, package clauses, type and class declarations, every
function and method *signature*, module-level constants, comments, and the
file's structure. What goes: the statements inside function bodies.

**Why this does not run inside protect/restore.** Every other compressor does,
because prose has a handful of technical spans that must not be rewritten.
Code is the opposite — nearly every span is technical, so ``protect()`` would
replace most of the file with sentinels and no grammar can parse that. This
compressor gets the stronger guarantee instead: it never rewrites a byte. It
computes byte ranges for function bodies and splices them out, so everything
that survives is a verbatim slice of the input. A signature cannot drift
because nothing ever regenerates it.

**A language we cannot parse is never guessed at.** Detection is not a
heuristic vote — a candidate language only wins if its grammar parses the
input with no ERROR nodes *and* the tree contains at least one definition we
would actually elide. Ruby, a unified diff, a stack trace and English prose
all come back byte-identical. A grammar that "almost" fits would delete the
wrong ranges, and the reader would have no signal that anything was lost.

Supported: Python, JavaScript, TypeScript (and TSX), Go. Three families done
well; the rest is passthrough until each is done properly.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("klio.compression.code")

# Python bodies collapse to an ellipsis statement, which is what the language
# itself uses for an omitted body. Brace languages keep their braces and take
# a comment, so the result still reads as the construct it came from.
ELISION_PYTHON = "..."
ELISION_BLOCK = "/* ... */"

# Node types whose ``body`` field is the implementation we drop.
_DEFINITION_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "javascript": frozenset(
        {
            "function_declaration",
            "function_expression",
            "function",
            "generator_function",
            "generator_function_declaration",
            "method_definition",
            "arrow_function",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration", "func_literal"}),
}
_DEFINITION_TYPES["typescript"] = _DEFINITION_TYPES["javascript"]
_DEFINITION_TYPES["tsx"] = _DEFINITION_TYPES["javascript"]

# Detection asks a different question from elision: "is this that language?",
# not "is there a body to drop?". A module of nothing but dataclasses is
# unambiguously Python and simply has nothing to compress — treating it as
# undetected would push it to the prose model, which is exactly the guess this
# module exists to refuse.
_EVIDENCE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "function_definition",
            "class_definition",
            "decorated_definition",
            "import_statement",
            "import_from_statement",
        }
    ),
    "javascript": frozenset(
        {
            "class_declaration",
            "import_statement",
            "export_statement",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        }
    )
    | _DEFINITION_TYPES["javascript"],
    "go": frozenset(
        {
            "package_clause",
            "import_declaration",
            "type_declaration",
        }
    )
    | _DEFINITION_TYPES["go"],
}
_EVIDENCE_TYPES["typescript"] = _EVIDENCE_TYPES["javascript"]
_EVIDENCE_TYPES["tsx"] = _EVIDENCE_TYPES["javascript"]

# Body node types that enclose their own delimiters, so only the interior is
# replaced. Python's ``block`` carries no delimiters and is replaced whole.
_BRACED_BODY_TYPES = frozenset({"statement_block", "block"})

# Detection order. The first grammar that parses cleanly wins, so the stricter
# dialect must come first: TypeScript parses plain JavaScript happily, but
# JavaScript rejects type annotations, which is exactly the discrimination we
# want.
_DETECTION_ORDER: tuple[str, ...] = ("python", "go", "javascript", "typescript")

# A unified diff is line-oriented text about code, not code. Some hunks parse
# as valid expressions in more than one grammar, so it is excluded explicitly
# rather than left to luck.
_DIFF_MARKERS = (
    re.compile(r"^diff --git ", re.MULTILINE),
    re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE),
    re.compile(r"^--- .*\n\+\+\+ ", re.MULTILINE),
)


@lru_cache(maxsize=1)
def _parsers() -> dict[str, Any]:
    """Build one parser per supported grammar.

    Cached because grammar construction is the expensive part. Missing
    grammars are skipped, not fatal: a deployment without ``tree-sitter-go``
    simply never detects Go, and Go source passes through untouched.
    """
    try:
        from tree_sitter import Language, Parser
    except Exception:
        logger.warning("tree-sitter is unavailable; code compression is passthrough")
        return {}

    loaders: list[tuple[str, str, str]] = [
        ("python", "tree_sitter_python", "language"),
        ("javascript", "tree_sitter_javascript", "language"),
        ("typescript", "tree_sitter_typescript", "language_typescript"),
        ("tsx", "tree_sitter_typescript", "language_tsx"),
        ("go", "tree_sitter_go", "language"),
    ]

    built: dict[str, Any] = {}
    for name, module_name, factory in loaders:
        try:
            module = __import__(module_name)
            built[name] = Parser(Language(getattr(module, factory)()))
        except Exception:
            logger.warning("grammar for %s is unavailable; skipping", name)
    return built


def looks_like_a_diff(source: str) -> bool:
    """Whether this is a unified diff rather than source.

    Public because the router needs the same answer: a diff is line-oriented
    text *about* code, and both the AST compressor and the prose model would
    damage it.
    """
    return any(marker.search(source) for marker in _DIFF_MARKERS)


def _has_error(node: Any) -> bool:
    """Whether the parse contains any ERROR or MISSING node.

    ``has_error`` is a cheap tree-level flag; consulting it means we never walk
    a tree we are about to reject.
    """
    return bool(node.has_error)


def _iter_nodes(root: Any):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _nodes_of_type(root: Any, wanted: frozenset[str]) -> list[Any]:
    return [node for node in _iter_nodes(root) if node.type in wanted]


def _definitions(root: Any, language: str) -> list[Any]:
    return _nodes_of_type(root, _DEFINITION_TYPES[language])


def detect_language(source: str) -> Optional[str]:
    """Return the language name, or ``None`` when we cannot be certain.

    ``None`` is the honest answer for anything unsupported and is the only
    answer for anything ambiguous. The caller passes those through untouched.
    """
    if not source.strip():
        return None
    if looks_like_a_diff(source):
        return None

    parsers = _parsers()
    if not parsers:
        return None

    data = source.encode("utf-8")
    for name in _DETECTION_ORDER:
        parser = parsers.get(name)
        if parser is None:
            continue
        try:
            tree = parser.parse(data)
        except Exception:
            logger.warning("parse failed for candidate %s", name, exc_info=True)
            continue
        if _has_error(tree.root_node):
            continue
        # An error-free parse is not enough on its own: a permissive grammar
        # will happily accept a paragraph of English as a sequence of
        # expression statements. Requiring a construct that only that language
        # produces — an import, a class, a package clause, a function — makes
        # the detector answer the question we are really asking.
        if _nodes_of_type(tree.root_node, _EVIDENCE_TYPES[name]):
            return name
    return None


def _elide_bodies(source: str, language: str) -> str:
    """Splice function bodies out of ``source``.

    Only whole body byte ranges are removed; nothing else is touched, so every
    surviving byte is verbatim.
    """
    parser = _parsers()[language]
    data = source.encode("utf-8")
    tree = parser.parse(data)

    cuts: list[tuple[int, int, bytes]] = []
    for node in _definitions(tree.root_node, language):
        body = node.child_by_field_name("body")
        if body is None:
            continue
        if language == "python":
            # Python's ``block`` carries no delimiters: it starts at the first
            # statement, so the newline and indentation after the colon are
            # outside it and survive untouched.
            start, end = body.start_byte, body.end_byte
            replacement = ELISION_PYTHON.encode("utf-8")
        elif body.type in _BRACED_BODY_TYPES:
            # Brace languages keep their own braces; only the interior goes.
            start, end = body.start_byte + 1, body.end_byte - 1
            replacement = f" {ELISION_BLOCK} ".encode("utf-8")
        else:
            # An expression-bodied arrow function: there is no block to drop,
            # and rewriting the expression would change what the code does.
            continue
        if end <= start or len(replacement) >= (end - start):
            # Never grow. An empty or one-token body costs more as a marker.
            continue
        cuts.append((start, end, replacement))

    if not cuts:
        return source

    # Outer bodies subsume inner ones: a nested function disappears with its
    # parent, and eliding it twice would splice into a range already removed.
    cuts.sort(key=lambda cut: (cut[0], -cut[1]))
    merged: list[tuple[int, int, bytes]] = []
    for start, end, replacement in cuts:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, replacement))

    out = bytearray()
    cursor = 0
    for start, end, replacement in merged:
        out += data[cursor:start]
        out += replacement
        cursor = end
    out += data[cursor:]
    return out.decode("utf-8")


def compress_code(source: str, *, keep_bodies: bool = False) -> str:
    """Return ``source`` with function bodies dropped.

    Fail-open: an unsupported language, a missing grammar or any internal
    error returns the input unchanged. Compression is an optimisation and is
    never allowed to be the reason a request fails.

    Args:
        source: the code to compress.
        keep_bodies: return the source untouched. The caller that wants
            structure without paying the lossy price asks for it here rather
            than by skipping the compressor, so the decision is visible in one
            place.
    """
    if keep_bodies:
        return source
    try:
        language = detect_language(source)
        if language is None:
            return source
        return _elide_bodies(source, language)
    except Exception:
        logger.warning("code compression failed; passing through", exc_info=True)
        return source
