from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow

from src.features.feature_engineering import feature_engineering
from src.features.data import download_data, recover_data_from_processed


def run_drift_evaluation(train_df, prod_df):
    report = Report([DataDriftPreset(method="psi")], include_tests="True")
    my_eval = report.run(train_df, prod_df)
    # Para extrair como dicionário
    result = my_eval.dict()
    return result["metrics"][0]["value"]["share"]

def get_training_df(stock: str):
    return recover_data_from_processed(stock)

def get_production_df(stock: str):
    prod_df_raw = download_data(stock, periodo='6y')
    prod_df = feature_engineering(prod_df_raw, stock)
    return prod_df


def drift_evaluation_pipeline(stock: str):
    train_df = get_training_df(stock)
    prod_df = get_production_df(stock)
    drift_share = run_drift_evaluation(train_df, prod_df)
    mlflow.log_metric("drift_share", drift_share)


    return {"drift_share": drift_share}
