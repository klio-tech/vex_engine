FROM python:3.11-slim

WORKDIR /app

# Copy the shared models package into a location where
# `import shared.models` resolves correctly via PYTHONPATH.
COPY services/shared /app/lib/shared
ENV PYTHONPATH="/app/lib/shared"

# Install shared package dependencies (pydantic, sqlalchemy, psycopg2).
RUN pip install --no-cache-dir pydantic>=2.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0

# Install the MCP server service.
COPY services/mcp-server/pyproject.toml /app/
COPY services/mcp-server/app /app/app
RUN pip install --no-cache-dir .

# Host port 8085 is free: 8080 (sync-gateway), 8081 (ingestion-api),
# 8082 (dashboard-api) are taken; container listens on 8000 like the
# other FastAPI services and is mapped to 8085 in docker-compose.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
