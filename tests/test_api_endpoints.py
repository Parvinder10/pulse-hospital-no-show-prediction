"""
Test suite for FastAPI endpoints using TestClient.
"""
from fastapi.testclient import TestClient
from src.api.main import app

def test_api_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_api_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]) > 10

def test_api_predict_valid_appointment():
    with TestClient(app) as client:
        payload = {
            "lead_days": 10,
            "Age": 45,
            "gender_female": 0,
            "Scholarship": 0,
            "neighbourhood": "CENTRO",
            "prior_appointments_count": 3,
            "prior_no_shows_count": 0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["predicted_risk_probability"] <= 1.0
        assert data["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
        assert len(data["top_drivers"]) > 0

def test_api_predict_validation_rejects_negative_lead_days():
    with TestClient(app) as client:
        invalid_payload = {
            "lead_days": -5,
            "Age": 30
        }
        response = client.post("/predict", json=invalid_payload)
        # Pydantic validation error returns 422 Unprocessable Entity
        assert response.status_code == 422
