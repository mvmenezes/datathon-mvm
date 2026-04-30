import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
import argparse

PARAMS_PATH = Path("params.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Ex: PETR4.SA")
    return parser.parse_args()

def add_stock(stock: str):
    # Lê o arquivo atual
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    # Verifica se já existe
    if stock in params["stocks"]:
        return

    # Adiciona e salva
    params["stocks"].append(stock)
    with open(PARAMS_PATH, "w") as f:
        yaml.dump(params, f)


def download_data(stock: str, periodo: str='6y'):
    try:
        df_vale = _download_data(stock, periodo)
        df_dolar = _download_data("USDBRL=X", periodo)
        #Renamear a coluna para dolar antes de fazer o merge
        df_dolar.rename(columns={"Close":"Dolar"}, inplace=True)
        #Faz o merge
        df_final = pd.concat([df_vale, df_dolar["Dolar"]], axis=1)
        return df_final
    except(ValueError):
        raise ValueError(f"Não foi possivel recuperar os dados da ação {stock}, tente novamente com outros parâmetros")

def save_data_raw(df: pd.DataFrame, stock: str):
    try:
        df.to_csv(f"data/raw/{stock}.csv", index=False)
        add_stock(stock)
    except(ValueError) as e:
        raise ValueError(str(e))

def recover_data_from_raw(stock: str):
    try:
        df = pd.read_csv(f"data/raw/{stock}.csv")
        return df
    except(FileNotFoundError):
        raise ValueError(f"Não foi possível recuperar os dados da ação {stock}. Verifique se o arquivo existe e tente novamente.")

def recover_data_from_processed(stock: str):
    try:
        df = pd.read_parquet(f"data/processed/{stock}.parquet")
        return df
    except(FileNotFoundError):
        raise ValueError(f"Não foi possível recuperar os dados da ação {stock}. Verifique se o arquivo existe e tente novamente.")


def _download_data(ticker: str , per: str='6y'):
    try:
        # Pegar as informações da ação
        dat = yf.Ticker(ticker)
        df = dat.history(period=per)
        if df.empty:
            raise ValueError(f"Não foi possível obter dados para a ação {ticker} em {per}. Verifique se o ticker está correto e tente novamente.")
        df = df.reset_index()
        df.head()
        return df
    except(ValueError):
        raise ValueError(f"Não foi possível obter dados para a ação {ticker} em {per}. Verifique se o ticker está correto e tente novamente.")



if __name__ == "__main__":
    args = parse_args()
    df = download_data(args.stock, '6y')
    save_data_raw(df, args.stock)