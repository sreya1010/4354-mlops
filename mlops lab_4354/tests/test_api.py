from fastapi.testclient import TestClient
import sys
sys.path.insert(0, "src")
from api import app

client = TestClient(app)


def valid_observation():
    return {
        "relative_humidity_pct": 82,
        "room_temperature_c": 24,
        "wall_surface_temperature_c": 21,
        "door_openings_per_hour": 7,
        "ventilation_duration_min": 30,
        "dehumidifier_operating_hours": 5,
        "room_occupancy": 2,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_contract():
    response = client.post("/predict", json=valid_observation())
    assert response.status_code == 200
    assert response.json()["status"] in {
        "Moisture Risk", "Storage Conditions Acceptable"
    }


def test_humidity_validation():
    body = valid_observation()
    body["relative_humidity_pct"] = 101
    assert client.post("/predict", json=body).status_code == 422


def test_temperature_validation():
    body = valid_observation()
    body["room_temperature_c"] = 61
    assert client.post("/predict", json=body).status_code == 422


def test_wall_temperature_validation():
    body = valid_observation()
    body["wall_surface_temperature_c"] = -11
    assert client.post("/predict", json=body).status_code == 422
