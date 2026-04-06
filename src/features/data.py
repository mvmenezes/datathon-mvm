import yfinance as yf
import pandas as pd

def save_data(stock: str, periodo: str='6y'):
    try:
        df_vale = _download_data(stock, periodo)
        df_dolar = _download_data("USDBRL=X", periodo)
        #Renamear a coluna para dolar antes de fazer o merge
        df_dolar.rename(columns={"Close":"Dolar"}, inplace=True)
        #Faz o merge
        df_final = pd.concat([df_vale, df_dolar["Dolar"]], axis=1)
                
        df_final.to_csv(f"data/raw/{stock}.csv", index=False)

    except(ValueError):
        raise ValueError(f"Não foi possivel recuperar os dados da ação {stock}, tente novamente com outros parâmetros")

def recover_data_from_raw(stock: str):
    try:
        df = pd.read_csv(f"data/raw/{stock}.csv")
        return df
    except(FileNotFoundError):
        raise ValueError(f"Não foi possível recuperar os dados da ação {stock}. Verifique se o arquivo existe e tente novamente.")

def recover_data_from_processed(stock: str):
    try:
        df = pd.read_csv(f"data/processed/{stock}.csv")
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

