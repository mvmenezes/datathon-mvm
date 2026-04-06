import pandas as pd



def _create_strategy(df: pd.DataFrame):
    
    
    #Vamos tentar médias móveis (Feijão com arroz)
    df["short_mm"] = df["Close"].rolling(5, min_periods=1).mean()
    df["medium_mm"] = df["Close"].rolling(15, min_periods=1).mean()
    df["large_mm"] = df["Close"].rolling(45, min_periods=1).mean()
    return df


def feature_engineering(df: pd.DataFrame, stock: str):
    try:
        
        #remove dados
        df.dropna(inplace=True)
        
        #Cria a estrategia de medias moveis
        df = _create_strategy(df)
        df.to_csv(f"data/processed/{stock}.csv", index=False)
        
        
    except(ValueError):
        raise ValueError(f"Não foi possivel recuperar os dados da ação, tente novamente com outros parâmetros")
