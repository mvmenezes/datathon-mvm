from fastapi.testclient import TestClient
from ..src.serving.app import app

client = TestClient(app)

def test_predict_ok():
    response = client.post("/predict", json={
        "feature1": 10,
        "feature2": 5
    })

    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_input():
    response = client.post("/predict", json={
        "feature1": -999,  # inválido
        "feature2": "erro"
    })

    assert response.status_code == 422

    