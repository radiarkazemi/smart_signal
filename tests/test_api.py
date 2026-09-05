from __future__ import annotations

from fastapi.testclient import TestClient

from smart_signal.api import create_app


def test_health_and_dashboard():
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "params" in body
    home = client.get("/")
    assert home.status_code == 200
    assert "GoldNet" in home.text
