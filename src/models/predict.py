
from fastapi import params
from fastapi import params
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import mlflow

from .Exceptions.LSTMException import ModelNotTrainedException
from .PredictParams import PredictParams
from .LSTMParams import LSTMParams
from .lstm import ModelFactory
import numpy as np
from src.features.data import recover_data_from_processed
model = None
MODEL_PATH = "./data/models/lstm"


def inverse_values(scaler, input_data, fields):
    pred_np = input_data.numpy().reshape(-1,1)
    dummy = np.zeros((len(pred_np),len(fields)))
    dummy[:,0] = pred_np[:,0]
    inversed = scaler.inverse_transform(dummy)[:,0]
    return inversed

def predict(params: PredictParams):
    pred = 0.0
    last_x_days = recover_data_from_processed(str(params.stock)).tail(params.days)
    scaler , data_scaled = _scale_data(last_x_days, ["Close","Volume","Dolar","short_mm","medium_mm","large_mm"])
    X_torch, _ = _create_window(data_scaled, length=3)
    
    model, checkpoints = _load_model(params.stock, params.model_type)
    print(checkpoints)
    if model:
        model.eval()
        with torch.no_grad():
            pred = model(X_torch).squeeze()
    pred_np = pred.numpy().reshape(-1,1)
    dummy = np.zeros((len(pred_np),6))
    dummy[:,0] = pred_np[:,0]
    pred_real = scaler.inverse_transform(dummy)[:,0]
    return {
        "stock": params.stock,
        "predicted_price": round(pred_real[-1],2)
        
    }

def _load_model(stock, model_type="simple", hidden_size=64, input_size=6, num_layers=2):
    model = ModelFactory.create(model_type, hidden_size=hidden_size, input_size=input_size, num_layers = num_layers)
    try:
        model.load_state_dict(torch.load(MODEL_PATH+f"_{stock}_{model_type}.pth"))
        checkpoints = torch.load(MODEL_PATH+f"_{stock}_{model_type}.pth")
    except(FileNotFoundError):
        raise ModelNotTrainedException("Modelo não treinado para essa ação. Por favor, treine o modelo antes de fazer previsões.")
    model.eval()
    return model, checkpoints

def _scale_data(df, fields):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[fields])
    return scaler ,data_scaled


def _create_window(data_scaled, length=10):
    window = length
    x = []
    y = []
    for i in range(len(data_scaled)-window):
        x.append(data_scaled[i:i+window])
        y.append(data_scaled[i+window][0])  #Peguei a coluna Close
    X_torch = torch.tensor(x, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    return X_torch, y_torch

def _separate_training_data(percentual, X_torch, y_torch):
    train_size = int(percentual * len(X_torch))

    X_train = X_torch[:train_size]
    X_test  = X_torch[train_size:]

    y_train = y_torch[:train_size]
    y_test  = y_torch[train_size:]

    return X_train, y_train, X_test, y_test
