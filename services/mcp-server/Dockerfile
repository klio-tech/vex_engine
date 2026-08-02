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
#
# boto3 is required because shared.artifact_storage imports it at module
# scope and mcp_app imports artifact_storage at module scope — so a missing
# boto3 is not a lazy failure, it is a crash loop on container start.
RUN pip install --no-cache-dir \
    "pydantic>=2.0.0" \
    "sqlalchemy>=2.0.0" \
    "psycopg2-binary>=2.9.0" \
    "pgvector>=0.3.0" \
    "litellm>=1.0.0" \
    "boto3>=1.29.0"

# Tree-sitter grammars for the code compressor (shared[code]). ~4.5 MB of
# wheels. Without them the compressor detects nothing and passes code through,
# which is safe but useless, so they are installed rather than left optional
# here.
#
# The prose model (shared[prose]) is deliberately NOT installed: onnxruntime
# plus the int8 weights is ~350 MB, roughly doubling this image, for a
# compressor that saves ~1% on technical text at the shipped threshold. Prose
# compression degrades to passthrough in this image by design; install
# shared[prose] where it is actually wanted.
RUN pip install --no-cache-dir \
    "tree-sitter>=0.25,<0.27" \
    "tree-sitter-python>=0.25,<0.26" \
    "tree-sitter-javascript>=0.25,<0.26" \
    "tree-sitter-typescript>=0.23,<0.24" \
    "tree-sitter-go>=0.25,<0.26"

# Install the MCP server service.
COPY services/mcp-server/pyproject.toml /app/
COPY services/mcp-server/app /app/app
RUN pip install --no-cache-dir .

# Host port 8085 is free: 8080 (sync-gateway), 8081 (ingestion-api),
# 8082 (dashboard-api) are taken; container listens on 8000 like the
# other FastAPI services and is mapped to 8085 in docker-compose.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
