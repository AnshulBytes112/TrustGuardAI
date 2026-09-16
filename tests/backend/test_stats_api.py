from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_overview_stats_api():
    res = client.get("/api/stats/overview")
    assert res.status_code == 200
    data = res.json()
    assert "total_datasets" in data
    assert "total_samples" in data
    assert "total_quarantined" in data
    assert "total_scans" in data
    assert "completed_scans" in data
    assert "running_scans" in data
    assert "recent_scans" in data
    assert "recent_datasets" in data
    assert isinstance(data["total_datasets"], int)
    assert isinstance(data["total_samples"], int)
