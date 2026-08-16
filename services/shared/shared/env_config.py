"""Guarded parsers for environment-variable configuration.

The engine already had the right pattern — ``_parse_bool`` / ``_parse_int`` in
``mcp-server/app/config.py`` and ``_parse_dedup_threshold`` in
``shared/memory.py`` — it was just confined to those two modules. Everywhere
else, config was read with a bare ``== "1"`` or a bare ``float()``, and those
are the two failure shapes this module exists to remove:

**A bare ``== "1"`` silently disagrees with the operator.** ``MCP_OAUTH_ENABLED=true``
evaluated to ``False`` under ``== "1"``: the operator turned OAuth on, the
service left it off, and because the same flag gated the "required config is
missing" check at startup, nothing said so. The same shape guarded
``KLIO_PROSE_ALLOW_DOWNLOAD``.

**A bare ``float()`` outside a ``try`` turns a typo into an outage.**
``VERIFICATION_TIMEOUT_S='10s'`` raised ``ValueError: could not convert string
to float: '10s'`` out of *every* ``call_llm``, at two separate sites, because
the parse sat above the ``try`` that exists precisely to make LLM failures
degrade gracefully.

The rules here are deliberately uniform, so that reading any call site tells
you the whole story:

* unset, empty or whitespace-only → ``default`` (an empty string is what
  Railway and docker-compose produce for a declared-but-unset variable; it is
  an absence, not an instruction);
* recognised token → that value;
* anything else, including out-of-range numbers → ``default``, with a WARNING
  naming the variable and the value.

Falling back is correct for these knobs and wrong for some others: where a bad
value is *destructive* rather than merely surprising, refuse to start instead.
``RAW_OBSERVATION_TTL_DAYS`` in ``scripts/run_retention.py`` is the one such
value in the engine, and it raises.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agentguard.shared.env_config")

#: Tokens accepted as true. Matches the set ``mcp-server/app/config.py`` has
#: always used, so adopting this module changes no existing behaviour.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: Tokens accepted as false. Everything outside ``TRUE | FALSE`` is a typo and
#: takes the default rather than being coerced to ``False`` — "treu" is not a
#: request to disable the feature.
FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def _clean(value: str | None) -> str | None:
    """Normalise a raw env value, treating blank as absent."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_bool(value: str | None, *, default: bool, name: str = "") -> bool:
    """Parse a truthy/falsy env string, falling back to ``default``.

    Args:
        value: The raw environment value, or ``None`` when unset.
        default: Returned when the value is absent or unrecognised.
        name: The variable name, used only in the warning.

    Returns:
        The parsed boolean, or ``default``.

    Unrecognised tokens return ``default``, not ``False``. The previous
    behaviour meant ``KLIO_FEATURE=treu`` disabled a feature whose default was
    on, which is the opposite of what the typo asked for and produced no log
    line to explain it.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return default

    token = cleaned.lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False

    logger.warning(
        "%s=%r is not a recognised boolean; using the default (%s). Accepted: %s / %s",
        name or "boolean env var",
        value,
        default,
        ", ".join(sorted(TRUE_TOKENS)),
        ", ".join(sorted(FALSE_TOKENS)),
    )
    return default


def parse_float(
    value: str | None,
    *,
    default: float,
    name: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float env string, falling back to ``default``.

    Args:
        value: The raw environment value, or ``None`` when unset.
        default: Returned when the value is absent, unparseable or out of range.
        name: The variable name, used only in the warning.
        minimum: Inclusive lower bound, or ``None`` for unbounded.
        maximum: Inclusive upper bound, or ``None`` for unbounded.

    Returns:
        The parsed float, or ``default``.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return default

    try:
        parsed = float(cleaned)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the default (%s)",
            name or "float env var",
            value,
            default,
        )
        return default

    return _bounded(
        parsed, default=default, name=name, minimum=minimum, maximum=maximum
    )


def parse_int(
    value: str | None,
    *,
    default: int,
    name: str = "",
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an int env string, falling back to ``default``.

    Args:
        value: The raw environment value, or ``None`` when unset.
        default: Returned when the value is absent, unparseable or out of range.
        name: The variable name, used only in the warning.
        minimum: Inclusive lower bound, or ``None`` for unbounded.
        maximum: Inclusive upper bound, or ``None`` for unbounded.

    Returns:
        The parsed int, or ``default``.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return default

    try:
        parsed = int(cleaned)
    except ValueError:
        logger.warning(
            "%s=%r is not a whole number; using the default (%s)",
            name or "int env var",
            value,
            default,
        )
        return default

    return int(
        _bounded(parsed, default=default, name=name, minimum=minimum, maximum=maximum)
    )


def parse_str(
    value: str | None,
    *,
    default: str,
    allowed: "tuple[str, ...] | frozenset[str]",
    name: str = "",
) -> str:
    """Parse an env string constrained to a closed set, else ``default``.

    The same contract as the numeric parsers, for the knobs whose value is a
    name rather than a number. ``allowed`` is not optional: the call sites this
    exists for interpolate the result into SQL (a column name cannot be a bind
    parameter), so "unrecognised takes the default" has to be a property of the
    parser rather than a discipline at each call site.

    Matching is case-insensitive because these are operator-typed tokens, not
    machine identifiers — the same reasoning that makes :data:`TRUE_TOKENS`
    lowercase-only and the caller's value ``.lower()``-ed into it.

    Args:
        value: The raw environment value, or ``None`` when unset.
        default: Returned when the value is absent or not in ``allowed``.
        allowed: The permitted values, lowercase.
        name: The variable name, used only in the warning.

    Returns:
        One of ``allowed``, or ``default``.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return default

    lowered = cleaned.lower()
    if lowered in allowed:
        return lowered

    logger.warning(
        "%s=%r is not one of %s; using the default (%s)",
        name or "str env var",
        value,
        ", ".join(sorted(allowed)),
        default,
    )
    return default


def _bounded(parsed, *, default, name, minimum, maximum):
    """Return ``parsed`` if it is within bounds, else ``default`` with a warning."""
    if minimum is not None and parsed < minimum:
        logger.warning(
            "%s=%s is below the minimum %s; using the default (%s)",
            name or "env var",
            parsed,
            minimum,
            default,
        )
        return default
    if maximum is not None and parsed > maximum:
        logger.warning(
            "%s=%s is above the maximum %s; using the default (%s)",
            name or "env var",
            parsed,
            maximum,
            default,
        )
        return default
    return parsed
