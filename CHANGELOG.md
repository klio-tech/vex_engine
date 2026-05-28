# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Klio memory-brain MCP service** (`services/mcp-server`): a new FastMCP service over
  Streamable HTTP exposing org-scoped memory tools, gated by a dedicated `memory`
  API-key scope. Reserved host port 8085; wired into `docker-compose.yml`, the Docker
  publish matrix, and the CI test matrix.
- **Migration 020**: spaces and projects tables plus nullable `space_id` / `project_id`
  columns on `session_memories` to support scoped brain memory.
- **Migration 021**: `brain_curator_state` table backing the org-scoped curator synthesis tick.
- **`memory` API-key scope**: a dedicated scope for the Klio brain, separate from `ingest`.

This change is purely additive: the existing verification, ingestion, and sync paths are
unchanged.
