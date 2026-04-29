
import logging
from datetime import datetime, timedelta
import yfinance as yf
from langchain_core.tools import tool

from src.features.data import recover_data_from_raw, save_data
from src.features.feature_engineering import feature_engineering
from src.models import train
from src.models.PredictParams import PredictParams
from src.models.predict import predict


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOOL 1 — Busca preço histórico da ação
# ---------------------------------------------------------------------------
@tool
def get_stock_history(ticker: str) -> str:
    """Busca os últimos 30 dias de preço de fechamento de uma ação.

    Args:
        ticker: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).

    Returns:
        String com histórico de preços formatado.
    """
    try:
        end   = datetime.today()
        start = end - timedelta(days=30)

        stock = yf.Ticker(ticker.strip().upper())
        df    = stock.history(start=start, end=end)

        if df.empty:
            return f"Nenhum dado encontrado para o ticker '{ticker}'."

        df       = df[["Close", "Volume"]].round(2)
        df.index = df.index.strftime("%Y-%m-%d")

        summary = (
            f"Ticker: {ticker.upper()}\n"
            f"Período: {df.index[0]} a {df.index[-1]}\n"
            f"Preço atual: R$ {df['Close'].iloc[-1]}\n"
            f"Mínimo 30d: R$ {df['Close'].min()}\n"
            f"Máximo 30d: R$ {df['Close'].max()}\n"
            f"Média 30d:  R$ {df['Close'].mean():.2f}\n\n"
            f"Últimos 5 fechamentos:\n{df.tail(5).to_string()}"
        )

        logger.info("Histórico obtido para %s: %d registros", ticker, len(df))
        return summary

    except Exception as e:
        logger.error("Erro ao buscar histórico de %s: %s", ticker, str(e))
        return f"Erro ao buscar dados para '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# TOOL 2 — Calcula indicadores técnicos
# ---------------------------------------------------------------------------

@tool
def calculate_technical_indicators(ticker: str) -> str:
    """Calcula indicadores técnicos clássicos para uma ação.

    Calcula: RSI, Média Móvel 7d e 21d, e Bandas de Bollinger.

    Args:
        ticker: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).

    Returns:
        String com indicadores técnicos e interpretação.
    """
    try:
        stock = yf.Ticker(ticker.strip().upper())
        df    = stock.history(period="60d")

        if df.empty:
            return f"Nenhum dado encontrado para '{ticker}'."

        close = df["Close"]

        # Médias móveis
        ma7  = close.rolling(window=7).mean().iloc[-1]
        ma21 = close.rolling(window=21).mean().iloc[-1]

        # RSI (14 períodos)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = (100 - (100 / (1 + rs))).iloc[-1]

        # Bandas de Bollinger (20 períodos)
        ma20       = close.rolling(20).mean()
        std20      = close.rolling(20).std()
        upper_band = (ma20 + 2 * std20).iloc[-1]
        lower_band = (ma20 - 2 * std20).iloc[-1]
        current    = close.iloc[-1]

        # Interpretações
        rsi_signal = (
            "SOBRECOMPRADO — possível queda"    if rsi > 70
            else "SOBREVENDIDO — possível alta" if rsi < 30
            else "NEUTRO"
        )

        trend_signal = (
            "ALTA — MA7 acima de MA21"
            if ma7 > ma21
            else "BAIXA — MA7 abaixo de MA21"
        )

        bb_signal = (
            "Preço próximo da banda SUPERIOR — resistência"  if current > upper_band * 0.98
            else "Preço próximo da banda INFERIOR — suporte" if current < lower_band * 1.02
            else "Preço dentro das bandas — sem sinal claro"
        )

        result = (
            f"Indicadores Técnicos — {ticker.upper()}\n"
            f"{'─' * 40}\n"
            f"Preço atual:     R$ {current:.2f}\n\n"
            f"Médias Móveis:\n"
            f"  MA7:  R$ {ma7:.2f}\n"
            f"  MA21: R$ {ma21:.2f}\n"
            f"  Sinal: {trend_signal}\n\n"
            f"RSI (14):\n"
            f"  Valor: {rsi:.1f}\n"
            f"  Sinal: {rsi_signal}\n\n"
            f"Bandas de Bollinger:\n"
            f"  Superior: R$ {upper_band:.2f}\n"
            f"  Inferior: R$ {lower_band:.2f}\n"
            f"  Sinal: {bb_signal}"
        )

        logger.info("Indicadores calculados para %s: RSI=%.1f", ticker, rsi)
        return result

    except Exception as e:
        logger.error("Erro ao calcular indicadores de %s: %s", ticker, str(e))
        return f"Erro ao calcular indicadores para '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# TOOL 3 — Busca notícias recentes da empresa
# ---------------------------------------------------------------------------

@tool
def get_stock_news(ticker: str) -> str:
    """Busca as notícias mais recentes relacionadas a uma ação.

    Args:
        ticker: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).

    Returns:
        String com as últimas notícias e suas fontes.
    """
    try:
        stock = yf.Ticker(ticker.strip().upper())
        news  = stock.news

        if not news:
            return f"Nenhuma notícia encontrada para '{ticker}'."

        result = f"Últimas notícias — {ticker.upper()}\n{'─' * 40}\n"

        for i, article in enumerate(news[:5], start=1):
            # yfinance ≥ 0.2.x retorna os metadados dentro de article["content"]
            content   = article.get("content", article)
            title     = content.get("title", article.get("title", "Sem título"))
            publisher = (
                content.get("provider", {}).get("displayName")
                or article.get("publisher", "Desconhecida")
            )
            pub_time  = (
                content.get("pubDate")          # novo formato ISO
                or article.get("providerPublishTime")  # formato legado (epoch)
            )

            if isinstance(pub_time, (int, float)):
                published = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
            elif isinstance(pub_time, str):
                # ISO 8601: "2025-04-28T12:00:00Z"
                published = pub_time[:16].replace("T", " ")
            else:
                published = "N/A"

            link = (
                content.get("canonicalUrl", {}).get("url")
                or article.get("link", "N/A")
            )

            result += (
                f"{i}. {title}\n"
                f"   Fonte:     {publisher}\n"
                f"   Publicado: {published}\n"
                f"   Link:      {link}\n\n"
            )

        logger.info("Notícias obtidas para %s: %d artigos", ticker, len(news[:5]))
        return result

    except Exception as e:
        logger.error("Erro ao buscar notícias de %s: %s", ticker, str(e))
        return f"Erro ao buscar notícias para '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# TOOL 4 — Ferramenta principal para previsão de preço de ações
# ---------------------------------------------------------------------------
@tool
def predict_stock_price(stock, days=7,  model_type="complex") -> str:
    """Ferramenta principal para previsão de preço de ações. Antes de fazer uma previsão, o agente deve realizar o download do histórico e treinar o modelo usando a API de treinamento. Depois, pode usar esta ferramenta para obter a previsão para os próximos dias.

    Args:
        days: Número de dias passados utilizados para prever o próximo dia. 
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        model_type: O tipo de modelo a ser usado ("complex" ou "simple"). Se não for especificado, o modelo "complex" será usado por padrão.
         

    Returns:
        String com a previsão do preço para os próximos dias.
    """
    try:
        params = PredictParams.model_validate({"stock": stock, "days": days, "model_type": model_type})
        result = predict(params)

        summary = (
            f"Stock: {stock.upper()}\n"
            f"days: {days} \n"
            f"Preço previsto: R$ {result['predicted_price']}\n"
            f"Modelo usado: {result['model_type']}\n")

        logger.info("Previsão obtida para %s: R$ %.2f", stock, result['predicted_price'])
        return summary

    except Exception as e:
        logger.error("Erro ao prever o valor da ação %s: %s", stock, str(e))
        return f"Erro ao prever o valor da ação {stock}: {str(e)}"

# ---------------------------------------------------------------------------
# TOOL 5 — Ferramenta principal para treinar o modelo de previsão de ações
# ---------------------------------------------------------------------------
@tool
def train_stock_model(stock, epochs=100, window=10, hidden_size=64, num_layers=2, learning_rate=0.001, per_training=0.8) -> str:
    """Ferramenta para treinar o modelo de previsão de ações. O agente deve usar esta ferramenta quando a ação não tiver um modelo treinado.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        epochs: Número de épocas para treinamento do modelo.
        window: Tamanho da janela de dias para criar as sequências de treinamento.
        hidden_size: Tamanho da camada oculta do modelo LSTM.
        num_layers: Número de camadas do modelo LSTM.
        learning_rate: Taxa de aprendizado para o otimizador.
        per_training: Porcentagem dos dados a serem usados para treinamento (entre 0 e 1).

    Returns:
        String confirmando que o modelo foi treinado com sucesso.
    """
    try:
        params = train.LSTMParams.model_validate({
            "stock": stock,
            "model_type": "complex",
            "epochs": epochs,
            "window": window,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "per_training": per_training
        })
        result = train.train_model(params)
        params.model_type = "simple"
        result = train.train_model(params)

        summary = (
            f"Stock: {stock.upper()}\n"
            f"Épocas: {epochs}\n"
            f"Tamanho da janela: {window}\n"
            f"Tamanho da camada oculta: {hidden_size}\n"
            f"Número de camadas: {num_layers}\n"
            f"Taxa de aprendizado: {learning_rate}\n"
            f"Porcentagem de treinamento: {per_training * 100}%\n"
            f"Resultado do treinamento: {result['mensagem']}"
        )

        logger.info("Modelo treinado para %s com %d épocas", stock, epochs)
        return summary

    except Exception as e:
        logger.error("Erro ao treinar o modelo para %s: %s", stock, str(e))
        return f"Erro ao treinar o modelo para {stock}: {str(e)}"

# ---------------------------------------------------------------------------
# TOOL 6 — Ferramenta para fazer o download do histórico de preços e indicadores técnicos, para preparar os dados para o treinamento do modelo. O agente deve usar esta ferramenta antes de usar a ferramenta de treinamento, caso os dados ainda não tenham sido baixados.
# ---------------------------------------------------------------------------
@tool
def download_stock_data(stock) -> str:
    """Ferramenta para fazer o download do histórico de preços e indicadores técnicos, para preparar os dados para o treinamento do modelo. O agente deve usar esta ferramenta antes de usar a ferramenta sempre que a ação não tiver dados disponíveis de acordo com a ferramenta de listagem de modelos treinados.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        """ 
    try:
        save_data(stock, '6y')

        summary = (
            f"Stock: {stock.upper()}\n"
            f"Resultado do download: Finalizado com sucesso."
        )

        logger.info("Dados baixados para %s", stock)
        return summary

    except Exception as e:
        logger.error("Erro ao baixar dados para %s: %s", stock, str(e))
        return f"Erro ao baixar dados para {stock}: {str(e)}"
    
# ---------------------------------------------------------------------------
# TOOL 7 — Ferramenta para preparar as features de treinamento.
# ---------------------------------------------------------------------------
@tool
def feature_engineering_tool(stock) -> str:
    """Ferramenta para preparar as features de treinamento. O agente deve usar esta ferramenta depis de realizar o download e antes de usar a ferramenta de treinamento, caso as features ainda não tenham sido preparadas.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        """ 
    try:
        df = recover_data_from_raw(stock)
        feature_engineering(df, stock)

        summary = (
            f"Stock: {stock.upper()}\n"
            f"Resultado da preparação das features: Finalizado com sucesso."
        )

        logger.info("Features preparadas para %s", stock)
        return summary

    except Exception as e:
        logger.error("Erro ao preparar features para %s: %s", stock, str(e))
        return f"Erro ao preparar features para {stock}: {str(e)}"

# ---------------------------------------------------------------------------
# TOOL 8 — Ferramenta para listar os modelos treinados disponíveis.
# ---------------------------------------------------------------------------
@tool
def list_trained_models() -> str:
    """Ferramenta para listar os modelos treinados disponíveis. Caso não encontre a ação ou o modelo solicitado, o agente deve usar a ferramenta para fazer download.

    Returns:
        String com a lista de modelos treinados, incluindo o nome da ação e o tipo do modelo (ex: USIM5.SA_complex).
    """
    import os

    try:
        files = os.listdir(train.MODEL_PATH)
        models = [f.replace(".pth", "").replace("lstm_", " ") for f in files if f.endswith(".pth")]

        if not models:
            return "Nenhum modelo treinado encontrado."

        result = "Modelos treinados disponíveis:\n" + "\n".join(models)
        logger.info("Modelos listados: %d", len(models))
        return result

    except Exception as e:
        logger.error("Erro ao listar modelos treinados: %s", str(e))
        return f"Erro ao listar modelos treinados: {str(e)}"
    
    ###