FROM python:3.11-slim

WORKDIR /app

# Install migration dependencies
RUN pip install --no-cache-dir alembic>=1.12.0 psycopg2-binary>=2.9.0 sqlalchemy>=2.0.0

# Copy migration files
COPY services/migrations/alembic.ini /app/alembic.ini
COPY services/migrations/alembic /app/alembic

# Override sqlalchemy.url — the entry point reads DATABASE_URL from env
CMD ["alembic", "upgrade", "head"]
