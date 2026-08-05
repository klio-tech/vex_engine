"""Memory feature configuration.

Centralizes memory-related environment variable lookups so all services
(sync-gateway, async-worker) use consistent defaults and feature flags.
"""

import os

from shared.env_config import parse_bool


def is_memory_enabled() -> bool:
    """Check if session memory is enabled via MEMORY_ENABLED env var.

    Parsed through :func:`shared.env_config.parse_bool` rather than the
    original ``.lower() == "true"``. That comparison accepted exactly one
    spelling: executed against it, ``'true '`` (trailing space), ``'1'``,
    ``'yes'`` and ``'on'`` all resolved to ``False``. Production has ``'true'``
    exactly, so production was never affected — but this flag gates the entire
    memory layer in both ``sync-gateway`` and ``async-worker``, which simply
    skip the write when it is off. A typo therefore turned the whole product
    off silently, with nothing logged to say why.
    """
    return parse_bool(
        os.environ.get("MEMORY_ENABLED"), default=False, name="MEMORY_ENABLED"
    )


def get_embedding_model() -> str:
    """Get the embedding model name from MEMORY_EMBEDDING_MODEL env var."""
    return os.environ.get(
        "MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
    )
