"""Integration tests for API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create test app with mocked dependencies."""
    with patch("src.main.chromadb") as mock_chroma_module:
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = True
        mock_client.list_collections.return_value = []
        mock_chroma_module.HttpClient.return_value = mock_client

        from src.main import create_app

        test_app = create_app()
        yield test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"


class TestCollectionsEndpoint:
    def test_list_collections_empty(self, client):
        response = client.get("/api/v1/collections")
        assert response.status_code == 200
        data = response.json()
        assert data["collections"] == []


class TestQueryEndpoint:
    def test_query_validation(self, client):
        # Missing required field
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_empty_string(self, client):
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422
