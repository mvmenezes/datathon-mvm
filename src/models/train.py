
from fastapi import params
from fastapi import params
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import mlflow
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .Exceptions.LSTMException import ModelNotTrainedException
from .LSTMParams import LSTMParams
from .lstm import ModelFactory
import numpy as np
from src.features.data import recover_data_from_processed
model = None
MODEL_PATH = "./data/models/lstm"

def train_model(params: LSTMParams):
    mlflow.set_tracking_uri("http://172.18.0.2:5000")

    data_downloaded = recover_data_from_processed(params.stock)
    fields = ["Close","Volume","Dolar","short_mm","medium_mm","large_mm"]
    scaler , data_scaled = _scale_data(data_downloaded, fields)
    X_torch, y_torch = _create_window(data_scaled, length=params.window)
    X_train, y_train, X_test, y_test = _separate_training_data(params.per_training, X_torch, y_torch)

    mlflow.set_experiment("lstm-previsao-serie-temporal")
    input_size = len(fields)
    forecast = []
    with mlflow.start_run(run_name=f"""lstm
                          -mt:{params.model_type}
                          -e:{params.epochs}
                          -is:{params.window}
                          -pt:{params.per_training}
                          -lr:{params.learning_rate}
                          -hs:{params.hidden_size}
                          -nl:{params.num_layers}""") as run:
        
        model = ModelFactory.create(params.model_type, hidden_size=params.hidden_size, input_size=input_size, num_layers = params.num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        best_loss = float(1)
        mlflow.set_tag("size", len(data_downloaded))
        mlflow.log_params({
                "hidden_size": params.hidden_size,
                "epochs": params.epochs,
                "input_size": input_size,
                "num_layers": params.num_layers,
                "learning_rate": params.learning_rate
            })
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", model_class.__module__.split(".")[0])
        mlflow.set_tag("owner", "grupo-MVM")
        mlflow.set_tag("phase", "datathon-fase05")
        
        for epoch in range(params.epochs):
            ##Treinamento do Modelo
            model.train()
            optimizer.zero_grad()

            output = model(X_train).squeeze()
            loss = criterion(output, y_train)
            loss_tr = loss.item()
            loss.backward()
            
            optimizer.step()


            ##Validação do Modelo
            model.eval()
            pred = None
            with torch.no_grad():
                pred = model(X_test).squeeze()
                loss_te = criterion(pred, y_test).item()

            
            print(f"Epoch {epoch} loss Training: {loss_tr:.6f} loss Test: {loss_te:.6f}")
            mlflow.log_metric("train_loss", loss_tr, step=epoch)
            mlflow.log_metric("test_loss", loss_te, step=epoch)
            mlflow.log_metric("best_val_loss", best_loss, step=epoch)
            if loss_te < best_loss:
                print(f"Saving best model with test loss {loss_te:.6f} at epoch {epoch}. Better than previous best loss {best_loss:.6f}")
                best_loss = loss_te
                # salvar pesos
                torch.save(model.state_dict(), MODEL_PATH+f"_{params.stock}_{params.model_type}.pth")
                forecast = pred
                run_id = run.info.run_id
            ratio = loss_te / loss_tr if loss_tr > 0 else float('inf')
            if ratio > 2.0:
                mlflow.log_metric("overfitting", 1)
            else:
                mlflow.log_metric("overfitting", 0)

        forecast = inverse_values(scaler, forecast, fields)
        real = inverse_values(scaler, y_test, fields)
    
        metrics = {
            "auc": roc_auc_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        return {
            "message": "Modelo treinado com sucesso. ",
            "stock": params.stock,
            #"run_id": run_id,
            "loss_te": loss_te,
            "loss_tr": loss_tr,
            "best_val_loss": best_loss,
            "ratio": ratio ,
            "overfitting": ratio > 2.0,
            "forecast": forecast.tolist(),
            "real_price": real.tolist()
        }

def inverse_values(scaler, input_data, fields):
    pred_np = input_data.numpy().reshape(-1,1)
    dummy = np.zeros((len(pred_np),len(fields)))
    dummy[:,0] = pred_np[:,0]
    inversed = scaler.inverse_transform(dummy)[:,0]
    return inversed


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
