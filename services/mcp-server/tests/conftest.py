import os
import sys

# Ensure the shared package is importable when running tests without
# installing it into the virtual-env (fallback for local dev).
# Add `services/` to path so that `import shared.models` resolves to
# `services/shared/models.py`.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", ".."),
)

import pytest
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)
