FROM python:3.11-slim

WORKDIR /app

# Install migration dependencies
RUN pip install --no-cache-dir alembic>=1.12.0 psycopg2-binary>=2.9.0 sqlalchemy>=2.0.0

# Copy migration files
COPY services/migrations/alembic.ini /app/alembic.ini
COPY services/migrations/alembic /app/alembic

# Migration 037 imports shared.lexical for the full-text expression, because
# Postgres matches an expression index by expression equality and a second
# copy of that string here could drift from the one the query uses — silently
# retiring the index. shared/__init__.py is empty and shared/lexical.py and
# shared/env_config.py import nothing outside the standard library, so this
# adds no dependency to the image; it only makes the module importable.
COPY services/shared/shared /app/shared
ENV PYTHONPATH=/app

# Override sqlalchemy.url — the entry point reads DATABASE_URL from env
CMD ["alembic", "upgrade", "head"]
