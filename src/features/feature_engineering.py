import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Ex: PETR4.SA")
    return parser.parse_args()

def _create_strategy(df: pd.DataFrame):
    df["short_mm"] = df["Close"].rolling(5, min_periods=1).mean()
    df["medium_mm"] = df["Close"].rolling(15, min_periods=1).mean()
    df["large_mm"] = df["Close"].rolling(45, min_periods=1).mean()
    return df


def feature_engineering(df: pd.DataFrame, stock: str):
    try:
        #remove dados
        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        #Cria a estrategia de medias moveis
        df = _create_strategy(df)
        return df
    except(ValueError):
        raise ValueError("Não foi possivel recuperar os dados da ação, tente novamente com outros parâmetros")

def save_parquet(df: pd.DataFrame, stock: str):
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