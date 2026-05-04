
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch
import torch.nn as nn
import numpy as np
import mlflow
import argparse
import socket
import yaml
from sklearn.preprocessing import MinMaxScaler
from src.models.Exceptions.LSTMException import ModelNotTrainedException
from src.models.LSTMParams import LSTMParams
from src.models.lstm import ModelFactory
from src.features.data import recover_data_from_processed
from pathlib import Path



MODEL_CONFIG = Path("configs/model_config.yaml")

model = None
MODEL_PATH = "./data/models/"
MLFLOW_URI = "http://172.18.0.4:5000"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Ex: PETR4.SA")
    return parser.parse_args()

def train_model(params: LSTMParams):

    if check_host(ip="mlflow", porta=5000):
        mlflow.set_tracking_uri(MLFLOW_URI)
        print(f"Conectado ao MLflow Tracking Server em {MLFLOW_URI}")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("Conectado ao MLflow Tracking Server em file:./mlruns")

    data_downloaded = recover_data_from_processed(params.stock)
    fields = ["Close","Volume","Dolar","short_mm","medium_mm","large_mm", "RSI", "bb_upper_band", "bb_lower_band"]
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
                          -nl:{params.num_layers}"""):
        
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
        mlflow.log_param("test_size", 0)
        mlflow.log_param("random_state", 0)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", model.__module__.split(".")[0])
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
                #Criar pasta se não existir
                Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
                # salvar pesos
                torch.save(model.state_dict(), MODEL_PATH+f"lstm_{params.stock}_{params.model_type}.pth")
                pred_y = pred
            ratio = loss_te / loss_tr if loss_tr > 0 else float('inf')
            if ratio > 2.0:
                mlflow.log_metric("overfitting", 1)
            else:
                mlflow.log_metric("overfitting", 0)

        forecast = inverse_values(scaler, pred_y, fields)
        real = inverse_values(scaler, y_test, fields)
        metrics = {
            "mean_absolute_error": mean_absolute_error(y_test, pred_y),
            "mean_squared_error": mean_squared_error(y_test, pred_y),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, pred_y)
        }
        mlflow.log_metrics(metrics)
        return {
            "message": "Modelo treinado com sucesso. ",
            "stock": params.stock,
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


def _load_model(stock, model_type="simple", hidden_size=64, input_size=9, num_layers=2):
    model = ModelFactory.create(model_type, hidden_size=hidden_size, input_size=input_size, num_layers = num_layers)
    try:
        model.load_state_dict(torch.load(MODEL_PATH+f"_{stock}_{model_type}.pth", weights_only=True))
        checkpoints = torch.load(MODEL_PATH+f"lstm_{stock}_{model_type}.pth", weights_only=True)
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
    X_torch = torch.tensor(np.array(x), dtype=torch.float32)
    y_torch = torch.tensor(np.array(y), dtype=torch.float32)
    return X_torch, y_torch

def _separate_training_data(percentual, X_torch, y_torch):
    train_size = int(percentual * len(X_torch))

    X_train = X_torch[:train_size]
    X_test  = X_torch[train_size:]

    y_train = y_torch[:train_size]
    y_test  = y_torch[train_size:]

    return X_train, y_train, X_test, y_test

def load_config() -> LSTMParams:
    with open(MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)

    return LSTMParams(
        stock         = "",
        epochs        = cfg["training"]["epochs"],
        per_training  = cfg["training"]["per_training"],
        learning_rate = cfg["training"]["learning_rate"],
        window        = cfg["training"]["window"],
        model_type    = "",
        hidden_size   = cfg["model"]["hidden_size"],
        num_layers    = cfg["model"]["num_layers"],
    )
if __name__ == "__main__":
    args = parse_args()
    stock = args.stock
    init_params = load_config()
    init_params.stock = stock
    init_params.model_type = "simple"

    try:
        train_model(init_params)
        init_params.model_type = "complex"
        train_model(init_params)
    except(ValueError) as e:
        raise ValueError(f"Erro ao processar {stock} - {str(e)}")


def check_host(ip: str, porta: int, timeout: float = 3.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        resultado = sock.connect_ex((ip, porta))
        sock.close()
        return resultado == 0  # 0 = conectou, qualquer outro = falhou
    except socket.error:
        return False