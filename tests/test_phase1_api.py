from fastapi.testclient import TestClient

from services.gateway.app import app


def test_health_and_rejected_media():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json() == {"status": "healthy"}
        response = client.post("/api/v1/inference/image", files={"file": ("bad.txt", b"x", "text/plain")})
        assert response.status_code == 415
