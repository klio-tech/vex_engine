"""Detect tool-call / build-artifact memories that should not be durable facts.

Used by the one-off cleanup script (:mod:`scripts.run_fact_cleanup`) to retract
the backlog of low-signal ``fact``/``summary`` rows the curator synthesized
from passive hook captures (e.g. "Tool Bash was used with command: …").

Deliberately CONSERVATIVE: every pattern describes a tool INVOCATION or a raw
tool I/O dump, never a durable statement. False positives (retracting a real
fact) are worse than false negatives (leaving a borderline one), so the
patterns match only unambiguous tool-call shapes. The retraction is reversible
(status='retracted'), and the cleanup script's dry-run lets a human review
hits before applying.
"""

import re

# Each pattern targets an unambiguous tool-call / raw-IO shape. Case-insensitive.
_NOISE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*used tool\b", re.IGNORECASE),
    re.compile(r"\btool\s+[\w./-]+\s+was used\b", re.IGNORECASE),
    re.compile(r"\bwas used with command\b", re.IGNORECASE),
    re.compile(r"\bused the bash tool\b", re.IGNORECASE),
    re.compile(r"\bexecuted (?:a |the )?bash (?:command|tool)\b", re.IGNORECASE),
    re.compile(r"\bexecuted bash command\b", re.IGNORECASE),
    re.compile(r"\btool\s+\w+\s+was used (?:on|to)\b", re.IGNORECASE),
    # Raw tool I/O dumps captured verbatim.
    re.compile(r'input:\s*\{"command"', re.IGNORECASE),
    re.compile(r'response:\s*\{"stdout"', re.IGNORECASE),
    re.compile(r"\bused tool mcp__", re.IGNORECASE),
)


def is_tool_call_noise(content: str | None) -> bool:
    """Return True if ``content`` is a tool-call / build-artifact memory.

    Pure and side-effect-free. ``None``/empty is never noise (returns False).
    """
    if not content:
        return False
    return any(p.search(content) for p in _NOISE_PATTERNS)
