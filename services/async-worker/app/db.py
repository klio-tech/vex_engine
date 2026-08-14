"""Database engine and session factory for the async worker.

Mirrors ``services/storage-worker/app/db.py``: SQLAlchemy, configured via
``DATABASE_URL``, falling back to local-dev defaults. Used by the outcome
grader (``app.outcome_grader``) to read recall/session evidence and write
graded outcomes — separate from the ownership sweep's own
``create_engine(config.DATABASE_URL)`` calls in ``app.main``, which stay as
they are so this change does not touch that already-working path.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",  # pragma: allowlist secret
)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(bind=engine)
