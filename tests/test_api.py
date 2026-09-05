from __future__ import annotations

from fastapi.testclient import TestClient

from smart_signal.api import create_app


def test_health_and_dashboard():
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body.get("service") == "smart-signal"
    home = client.get("/")
    assert home.status_code == 200
    assert "Smart Signal Live" in home.text
    hist = client.get("/signals")
    assert hist.status_code == 200
    assert "results" in hist.json()
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
