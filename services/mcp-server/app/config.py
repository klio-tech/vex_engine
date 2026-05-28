"""Configuration for the Vex MCP server.

Configuration is driven by environment variables, falling back to local
development defaults that match the other Vex services.
"""

import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)
