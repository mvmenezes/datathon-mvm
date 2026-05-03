from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
from prefect import flow, task

from src.features.feature_engineering import feature_engineering
from src.features.data import download_data, recover_data_from_processed


@task
def run_drift_evaluation(train_df, prod_df):
    report = Report([DataDriftPreset(method="psi")], include_tests="True")
    my_eval = report.run(train_df, prod_df)
    # Para extrair como dicionário
    result = my_eval.dict()
    return result["metrics"][0]["value"]["share"]

@task
def get_training_df(stock: str):
    return recover_data_from_processed(stock)

@task
def get_production_df(stock: str):
    prod_df_raw = download_data(stock, periodo='6y')
    prod_df = feature_engineering(prod_df_raw, stock)
    return prod_df


@flow(name="Drift Evaluation Pipeline")
def drift_evaluation_pipeline(stock: str):
    train_df = get_training_df(stock)
    prod_df = get_production_df(stock)
    drift_share = run_drift_evaluation(train_df, prod_df)
    mlflow.log_metric("drift_share", drift_share)

if __name__ == "__main__":
    # Exemplo de uso
    stock = "VALE3.SA"
    train_df = get_training_df(stock)
    prod_df = get_production_df(stock)
    result = run_drift_evaluation(train_df, prod_df)
    if result < 0.1:
        print(f"Drift baixo para {stock}: {result:.4f}")
    else:
        print(f"Drift alto para {stock}: {result:.4f}")

