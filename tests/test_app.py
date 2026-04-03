import os
from fastapi.testclient import TestClient

from app import app



def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "error"]


def test_infer_valid_input():
    payload = {
        "temp": 30.0,
        "pressure": 1.0,
        "vibration": 0.01,
        "voltage": 2.0,
    }
    with TestClient(app) as client:
        response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reconstruction_score" in data
    assert "threshold" in data
    assert "is_anomaly" in data
    assert isinstance(data["reconstruction_score"], float)
    assert isinstance(data["threshold"], float)
    assert isinstance(data["is_anomaly"], bool)


def test_infer_missing_field():
    payload = {"temp": 30.0, "pressure": 1.0, "vibration": 0.01}
    with TestClient(app) as client:
        response = client.post("/infer", json=payload)
    assert response.status_code == 422
