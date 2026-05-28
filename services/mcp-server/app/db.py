"""Database engine and session factory for the MCP server.

Uses SQLAlchemy to manage PostgreSQL connections with a synchronous
engine, mirroring the storage worker. Exposes a ``get_db`` FastAPI
dependency that yields a request-scoped session and closes it after use.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """FastAPI dependency yielding a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
