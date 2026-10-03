"""
Tests for FastAPI endpoints — Integration tests for the TennisIQ API.
"""

import pytest
from fastapi.testclient import TestClient

from tennis.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["app"] == "TennisIQ"


class TestSessionRoutes:
    def test_create_session(self, client):
        r = client.post("/api/v1/sessions/", params={"mode": "match"})
        assert r.status_code == 201
        assert "id" in r.json()

    def test_list_sessions(self, client):
        client.post("/api/v1/sessions/", params={"mode": "match"})
        r = client.get("/api/v1/sessions/")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_get_session_not_found(self, client):
        r = client.get("/api/v1/sessions/nonexistent")
        assert r.status_code == 404


class TestMatchRoutes:
    def test_create_match(self, client):
        r = client.post("/api/v1/matches/", params={
            "player1_name": "Alice", "player2_name": "Bob"
        })
        assert r.status_code == 201
        data = r.json()
        assert data["player1_name"] == "Alice"
        assert data["status"] == "in_progress"
        return data["id"]

    def test_score_point(self, client):
        r = client.post("/api/v1/matches/", params={
            "player1_name": "A", "player2_name": "B"
        })
        mid = r.json()["id"]
        r = client.post(f"/api/v1/matches/{mid}/score", params={
            "winner": "p1", "outcome_type": "winner"
        })
        assert r.status_code == 200
        assert "score_display" in r.json()

    def test_get_score(self, client):
        r = client.post("/api/v1/matches/", params={
            "player1_name": "A", "player2_name": "B"
        })
        mid = r.json()["id"]
        r = client.get(f"/api/v1/matches/{mid}/score")
        assert r.status_code == 200
        assert "score_display" in r.json()

    def test_undo_point(self, client):
        r = client.post("/api/v1/matches/", params={
            "player1_name": "A", "player2_name": "B"
        })
        mid = r.json()["id"]
        client.post(f"/api/v1/matches/{mid}/score", params={"winner": "p1"})
        r = client.post(f"/api/v1/matches/{mid}/undo")
        assert r.status_code == 200


class TestSubscriptionRoutes:
    def test_get_plans(self, client):
        r = client.get("/api/v1/subscriptions/plans")
        assert r.status_code == 200
        plans = r.json()
        assert len(plans) == 3
        tiers = [p["tier"] for p in plans]
        assert "free" in tiers
        assert "pro" in tiers
        assert "elite" in tiers

    def test_get_entitlement(self, client):
        r = client.get("/api/v1/subscriptions/user/user123")
        assert r.status_code == 200
        assert r.json()["tier"] == "free"

    def test_upgrade(self, client):
        r = client.post("/api/v1/subscriptions/user/user123/upgrade", params={"tier": "pro"})
        assert r.status_code == 200
        assert r.json()["tier"] == "pro"

    def test_feature_check(self, client):
        r = client.get("/api/v1/subscriptions/user/user123/feature/cloud_processing")
        assert r.status_code == 200
