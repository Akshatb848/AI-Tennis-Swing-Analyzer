import io
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    """Tests for the FastAPI service endpoints."""

    def test_health_endpoint(self, client):
        """GET /api/health returns 200 with status 'ok'."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_upload_csv(self, client):
        """POST /api/upload with a CSV file returns 200 and file metadata."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        response = client.post(
            "/api/upload",
            files={"file": ("test_data.csv", buf, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["file_name"] == "test_data.csv"
        assert data["size_bytes"] > 0
        assert "file_path" in data

    def test_pipeline_state(self, client):
        """GET /api/pipeline/state returns 200."""
        response = client.get("/api/pipeline/state")
        assert response.status_code == 200
