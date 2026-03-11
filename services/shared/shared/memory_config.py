"""Memory feature configuration.

Centralizes memory-related environment variable lookups so all services
(sync-gateway, async-worker) use consistent defaults and feature flags.
"""

import os


def is_memory_enabled() -> bool:
    """Check if session memory is enabled via MEMORY_ENABLED env var."""
    return os.environ.get("MEMORY_ENABLED", "false").lower() == "true"


def get_embedding_model() -> str:
    """Get the embedding model name from MEMORY_EMBEDDING_MODEL env var."""
    return os.environ.get("MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
