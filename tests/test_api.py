import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from jose import jwt
from datetime import datetime, timedelta
from limits import parse
from slowapi.wrappers import Limit
from src.serving.app import rate_limit_handler, app, SECRET_KEY, ALGORITHM, global_exception_handler
from fastapi import Request
from slowapi.errors import RateLimitExceeded


client = TestClient(app)


# =========================
# HELPERS
# =========================

def create_test_token(username="marcus.menezes"):
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def auth_headers():
    return {"Authorization": f"Bearer {create_test_token()}"}


# =========================
#  HEALTH
# =========================

def test_health_sem_token():
    response = client.get("/health")
    assert response.status_code == 401


def test_health_com_token():
    response = client.get("/health", headers=auth_headers())

    assert response.status_code == 200
    assert response.json() == {"mensagem": "OK"}


# =========================
#  LOGIN
# =========================

def test_login_sucesso():
    response = client.post(
        "/login",
        data={"username": "marcus.menezes", "password": "MarcusMenezes123"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalido():
    response = client.post(
        "/login",
        data={"username": "marcus.menezes", "password": "senha_errada"}
    )

    assert response.status_code == 400


# =========================
#  LLM
# =========================

@patch("src.serving.app.run_agent")
def test_input_llm(mock_run_agent):
    mock_run_agent.return_value = "Resposta do agente"

    response = client.post(
        "/input_llm",
        json={"input": "Qual a previsão da PETR4?"},
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert response.json()["mensagem"] == "Resposta do agente"


@patch("src.serving.app.run_pipeline")
def test_input_llm_rag(mock_run_pipeline):
    mock_run_pipeline.return_value = "Resposta RAG"

    response = client.post(
        "/input_llm_rag",
        json={"input": "Notícias da PETR4"},
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert response.json()["mensagem"] == "Resposta RAG"


# =========================
# DOWNLOAD DATA
# =========================

@patch("src.serving.app.download_data")
def test_download_data(mock_download):
    mock_download.return_value = pd.DataFrame({"col": [1, 2, 3]})

    response = client.post(
        "/download_data",
        json={"stock": "PETR4.SA"},
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert "baixados com sucesso" in response.json()["mensagem"]


# =========================
#  FEATURE ENGINEERING
# =========================

@patch("src.serving.app.recover_data_from_raw")
@patch("src.serving.app.feature_engineering")
@patch("src.serving.app.save_parquet")
def test_feature_engineering(mock_save, mock_fe, mock_recover):
    mock_recover.return_value = "df"
    mock_fe.return_value = "df_final"

    response = client.post(
        "/feature_engineering",
        json={"stock": "PETR4.SA"},
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert "Features" in response.json()["mensagem"]


# =========================
# 🧠 TRAIN MODEL
# =========================

@patch("src.serving.app.train_model")
def test_train_model_sucesso(mock_train):
    mock_train.return_value = {
        "forecast": [10, 11],
        "real_price": [12, 13]
    }

    payload = {
        "stock": "PETR4.SA",
        "epochs": 10,
        "window": 5,
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.01,
        "per_training": 0.8,
        "model_type": "simple"
    }

    response = client.post(
        "/train_model",
        json=payload,
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert "mensagem" in response.json()


def test_train_model_param_invalido():
    payload = {
        "stock": "PETR4.SA",
        "epochs": -1,  # inválido
        "window": 5,
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.01,
        "per_training": 0.8,
        "model_type": "simple"
    }

    response = client.post(
        "/train_model",
        json=payload,
        headers=auth_headers()
    )

    assert response.status_code == 400


# =========================
#  PREDICT
# =========================

@patch("src.serving.app.predict")
def test_predict_sucesso(mock_predict):
    mock_predict.return_value = {"forecast": [10, 11]}

    payload = {
        "stock": "PETR4.SA",
        "days": 5,
        "model_type": "simple"
    }

    response = client.post(
        "/predict",
        json=payload,
        headers=auth_headers()
    )

    assert response.status_code == 200
    assert "forecast" in response.json()


def test_predict_param_invalido():
    payload = {
        "stock": "",
        "days": -1,
        "model_type": "simple"
    }

    response = client.post(
        "/predict",
        json=payload,
        headers=auth_headers()
    )

    assert response.status_code == 400


# =========================
#  RATE LIMIT
# =========================

def test_rate_limit_handler():

    request = MagicMock(spec=Request)
    limit_mock = MagicMock()
    limit_mock.error_message = None 
    # Simula uma requisição que excedeu o limite
    exc = RateLimitExceeded(limit_mock)

    response = rate_limit_handler(request, exc)

    assert response.status_code == 429


# =========================
#  GLOBAL EXCEPTION
# =========================

def test_global_exception_handler():

    request = MagicMock(spec=Request)
    exc = Exception("erro teste")

    response = global_exception_handler(request, exc)
    print(response)
    assert response.status_code == 500