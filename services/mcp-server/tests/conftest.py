import os
import sys

# Ensure both `app` and the shared package are importable when running
# tests without installing them into the virtual-env (fallback for local
# dev).
#
# `app` resolves from the service directory (the pytest cwd).
#
# The shared package lives at `services/shared/shared/`, so the import
# root for `shared.auth` is `services/shared/` — NOT `services/` (which
# would bind the `shared` namespace to the `services/shared/` dir, where
# `auth.py` does not live).  We insert `services/shared/` first so the
# `shared` namespace resolves to the package that actually contains
# `auth.py`, and `services/` after it so anything importing the broader
# namespace still works.
_SERVICES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _SERVICES_DIR)
sys.path.insert(0, os.path.join(_SERVICES_DIR, "shared"))

import pytest  # noqa: E402
from app.auth import verify_api_key  # noqa: E402
from app.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from shared.auth import KeyInfo  # noqa: E402


def _fake_key_info() -> KeyInfo:
    """Return a stub KeyInfo for test auth bypass."""
    return KeyInfo(org_id="test-org", key_id="test-key", scopes=["ingest"])


@pytest.fixture
def client():
    """TestClient with auth bypassed via dependency_overrides.

    Future MCP tool tests can rely on this fixture so they exercise the
    tool logic without a live key/DB.  Tests that need real auth
    behaviour (e.g. the 401 path) should build their own client and skip
    the override, mirroring ingestion-api's pattern.
    """
    app = create_app()
    app.dependency_overrides[verify_api_key] = _fake_key_info
    return TestClient(app)
