from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_list_experiments_empty(tmp_path):
    with patch("backend.api.demo.ARTIFACT_DIR", tmp_path):
        response = client.get("/api/demo/experiments")
        assert response.status_code == 200
        assert response.json() == []


def test_get_experiment_invalid_fingerprint():
    response = client.get("/api/demo/experiments/invalid..fingerprint")
    assert response.status_code == 400
    assert "Invalid fingerprint" in response.json()["detail"]


def test_get_experiment_not_found(tmp_path):
    with patch("backend.api.demo.ARTIFACT_DIR", tmp_path):
        response = client.get("/api/demo/experiments/fake_fingerprint")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


def test_run_experiment_invalid_config_name():
    response = client.post("/api/demo/run", json={"config_name": "../invalid"})
    assert response.status_code == 400
    assert "Invalid config name" in response.json()["detail"]


def test_run_experiment_not_found(tmp_path):
    with patch("backend.api.demo.CONFIG_DIR", tmp_path):
        response = client.post("/api/demo/run", json={"config_name": "missing_config"})
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

