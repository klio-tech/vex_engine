FROM python:3.11-slim

WORKDIR /app

# Copy the shared models package into a location where
# `import shared.models` resolves correctly via PYTHONPATH.
COPY services/shared /app/lib/shared
ENV PYTHONPATH="/app/lib/shared"

# Install the shared package's runtime dependencies. mcp-server imports
# shared.memory, which imports litellm at module load (embeddings/LLM), and the
# shared package also declares pgvector — both are required at startup, not just
# pydantic/sqlalchemy/psycopg2. Keep this list in sync with
# services/shared/pyproject.toml [dependencies]. Specifiers are quoted so the
# shell does not interpret '>=' as an output redirection.
RUN pip install --no-cache-dir \
    "pydantic>=2.0.0" \
    "sqlalchemy>=2.0.0" \
    "psycopg2-binary>=2.9.0" \
    "pgvector>=0.3.0" \
    "litellm>=1.0.0"

# Install the MCP server service.
COPY services/mcp-server/pyproject.toml /app/
COPY services/mcp-server/app /app/app
RUN pip install --no-cache-dir .

# Host port 8085 is free: 8080 (sync-gateway), 8081 (ingestion-api),
# 8082 (dashboard-api) are taken; container listens on 8000 like the
# other FastAPI services and is mapped to 8085 in docker-compose.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
