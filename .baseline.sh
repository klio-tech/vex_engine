PY=~/Hive/Research/AgentGuard/vex_private/vex_engine/.venv/bin/python
export PYTHONPATH=/tmp/klio-fix-sweep/services/shared
export DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5442/klio_sweep"
export KLIO_REQUIRE_DB=1
echo "=== shared ==="; cd /tmp/klio-fix-sweep/services/shared && $PY -m pytest tests -q 2>&1 | tail -3
echo "=== mcp-server ==="; cd /tmp/klio-fix-sweep/services/mcp-server && $PY -m pytest tests -q 2>&1 | tail -3
echo "=== migrations ==="; cd /tmp/klio-fix-sweep/services/migrations && $PY -m pytest tests -q 2>&1 | tail -3
