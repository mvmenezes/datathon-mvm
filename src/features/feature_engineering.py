from pathlib import Path

import pandas as pd
import argparse

from src.features.data import download_data
from src.agent.rag_pipeline import stock_df_to_documents, upsert_documents



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Ex: PETR4.SA")
    return parser.parse_args()

def _create_strategy(df: pd.DataFrame):
    df["short_mm"] = df["Close"].rolling(5, min_periods=1).mean()
    df["medium_mm"] = df["Close"].rolling(15, min_periods=1).mean()
    df["large_mm"] = df["Close"].rolling(45, min_periods=1).mean()

     # RSI (14 períodos)
    df["Delta"] = df["Close"].diff()
    df["Gain"]  = df["Delta"].clip(lower=0).rolling(14, min_periods=1).mean()
    df["Loss"] = (-df["Delta"].clip(upper=0)).rolling(14, min_periods=1).mean()
    df["RS"] = df["Gain"] / df["Loss"]
    df["RSI"] = (100 - (100 / (1 + df["RS"])))
    df = df.drop(columns=["Gain", "Loss", "RS"])
    
    # Bandas de Bollinger (20 períodos)
    df["ma20"]       = df["Close"].rolling(20, min_periods=1).mean()
    df["std20"]      = df["Close"].rolling(20, min_periods=1).std()
    df["bb_upper_band"] = (df["ma20"] + 2 * df["std20"])
    df["bb_lower_band"] = (df["ma20"] - 2 * df["std20"])
    df = df.drop(columns=["ma20", "std20"])
    df.dropna(inplace=True)


    return df

def run_from_download_to_featuring(stock: str):
    df = download_data(stock, periodo="6y")
    df_final = feature_engineering(df, stock)
    return df_final


def feature_engineering(df: pd.DataFrame, stock: str):
    try:
        #remove dados
        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        #Cria a estrategia de medias moveis
        df = _create_strategy(df)
        docs = stock_df_to_documents(df, stock)
        upsert_documents(docs, stock)
        return df
    except(ValueError):
        raise ValueError("Não foi possivel recuperar os dados da ação, tente novamente com outros parâmetros")

def save_parquet(df: pd.DataFrame, stock: str):
    #Criar pasta se não existir
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"data/processed/{stock}.parquet", index=False)

def recover_data_from_raw(stock: str):
    try:
        df = pd.read_csv(f"data/raw/{stock}.csv")
        return df
    except(FileNotFoundError):
        raise ValueError(f"Não foi possível recuperar os dados da ação {stock}. Verifique se o arquivo existe e tente novamente.")

if __name__ == "__main__":
    args = parse_args()
    stock = args.stock
    try:
        df = recover_data_from_raw(stock)
        df_final = feature_engineering(df, stock)
        save_parquet(df_final, stock)
    except(ValueError) as e:
        raise ValueError(f"Erro ao processar {stock} - {str(e)}")