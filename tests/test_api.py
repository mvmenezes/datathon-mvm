from fastapi.testclient import TestClient
from src.serving.app import app


client = TestClient(app)
'''def test2_predict_no_model_available():
    response = client.get("/predict", json={
        "stock": "PETR4.SA",
        "days": 10,
        "model_type": "complex"
    })  
    assert response.status_code == 500
    assert "erro" in response.json()
'''

def test_predict_invalid_days():
    response = client.post("/predict", json={
        "stock": "VALE3.SA",
        "days": -1,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_predict_invalid_model_type():
    response = client.post("/predict", json={
        "stock": "VALE3.SA",
        "days": 10,
        "model_type": "invalid"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()
'''
def test_train_model_invalid_stock():
    response = client.post("/train_model", json={
        "stock": "INVALID_STOCK",
        "epochs": 30,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()
'''
def test_train_model_invalid_epochs():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": -1,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_window():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": -1,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_per_training():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": 3,
        "per_training": -0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_learning_rate():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": -0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_hidden_size():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": -64,
        "num_layers": 2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_num_layers():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": -2,
        "model_type": "complex"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

def test_train_model_invalid_model_type():
    response = client.post("/train_model", json={
        "stock": "VALE3.SA",
        "epochs": 30,
        "window": 3,
        "per_training": 0.8,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "model_type": "INVALID"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()
'''
def test_download_data_invalid_stock():
    response = client.post("/download_data", json={
        "stock": "INVALID_STOCK",
        "periodo": "6y"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()
'''
'''
def test_feature_engineering_invalid_stock():
    response = client.post("/feature_engineering", json={
        "stock": "INVALID_STOCK"
    })  
    assert response.status_code == 400
    assert "erro" in response.json()

'''

