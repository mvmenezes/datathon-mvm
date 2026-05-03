
import logging
from datetime import datetime, timedelta
import yfinance as yf
from langchain_core.tools import tool

from src.agent.rag_pipeline import build_rag_chain, get_or_create_vector_store, query_rag_with_context, stock_news_to_documents, upsert_documents
from src.features.data import recover_data_from_raw, download_data, save_data_raw
from src.features.feature_engineering import feature_engineering, save_parquet
from src.models import train
from src.models.PredictParams import PredictParams
from src.models.predict import predict


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOOL 1 — Busca preço histórico da ação
# ---------------------------------------------------------------------------
@tool(name_or_callable="buscar_historico_de_precos")
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
# TOOL 3 — Busca notícias recentes da empresa
# ---------------------------------------------------------------------------

@tool(name_or_callable="buscar_noticias")
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
                published = datetime.fromtimestamp(pub_time).strftime("%d-%m-%Y")
            elif isinstance(pub_time, str):
                published = pub_time[:16].replace("T", " ")
            else:
                published = "N/A"

            link = (
                content.get("canonicalUrl", {}).get("url")
                or article.get("link", "N/A")
            )
            content_created = f"No dia {published}, foi publicado uma a notícia que impacta de alguma forma a ação {ticker} com o título '{title}' foi publicada pela fonte {publisher}. O conteúdo da notícia é: {content.get('description', 'Sem descrição disponível.')}. O link da notícia é {link}."
            doc = stock_news_to_documents(content_created, ticker, published)
            upsert_documents(doc,ticker)

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
@tool(name_or_callable="prever_preco_da_acao")
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
@tool(name_or_callable="treinar_modelo_de_previsao")
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
        train.train_model(params)
        params.model_type = "simple"
        result = train.train_model(params)
        print("Treinamento concluído para ambos os modelos.")
        summary = (
            f"Stock: {stock.upper()}\n"
            f"Épocas: {epochs}\n"
            f"Tamanho da janela: {window}\n"
            f"Tamanho da camada oculta: {hidden_size}\n"
            f"Número de camadas: {num_layers}\n"
            f"Taxa de aprendizado: {learning_rate}\n"
            f"Porcentagem de treinamento: {per_training * 100}%\n"
            f"Resultado do treinamento: {result['message']}"
        )

        logger.info("Modelo treinado para %s com %d épocas", stock, epochs)
        return summary

    except Exception as e:
        logger.error("Erro ao treinar o modelo para %s: %s", stock, str(e))
        return f"Erro ao treinar o modelo para {stock}: {str(e)}"

# ---------------------------------------------------------------------------
# TOOL 6 — Ferramenta para fazer o download do histórico de preços e indicadores técnicos, para preparar os dados para o treinamento do modelo. O agente deve usar esta ferramenta antes de usar a ferramenta de treinamento, caso os dados ainda não tenham sido baixados.
# ---------------------------------------------------------------------------
@tool(name_or_callable="baixar_dados_historicos")
def download_stock_data(stock) -> str:
    """Ferramenta para fazer o download do histórico de preços e indicadores técnicos, para preparar os dados para o treinamento do modelo. O agente deve usar esta ferramenta antes de usar a ferramenta sempre que a ação não tiver dados disponíveis de acordo com a ferramenta de listagem de modelos treinados.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        """ 
    try:
        df = download_data(stock, '6y')
        save_data_raw(df, stock)
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
@tool(name_or_callable="prepare_features")
def feature_engineering_tool(stock) -> str:
    """Ferramenta para preparar as features de treinamento. O agente deve usar esta ferramenta depis de realizar o download e antes de usar a ferramenta de treinamento, caso as features ainda não tenham sido preparadas.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).
        """ 
    try:
        df = recover_data_from_raw(stock)
        df_final = feature_engineering(df, stock)
        save_parquet(df_final, stock)
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
@tool(name_or_callable="listar_modelos_treinados")
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
    
    

# ---------------------------------------------------------------------------
# TOOL 9 — Busca preço histórico da ação usando RAG. Esta ferramenta é uma alternativa à TOOL 1 e deve ser usada quando o agente julgar que a resposta da TOOL 1 não é suficiente para responder à pergunta do usuário, ou quando a ação tiver um modelo treinado, para verificar se a previsão do modelo faz sentido com base no histórico recente da ação.
# ---------------------------------------------------------------------------
@tool(name_or_callable="buscar_historico_de_precos_com_rag")
def get_stock_history_rag(stock: str) -> str:
    """Busca os últimos 30 dias de preço de fechamento de uma ação.

    Args:
        stock: Código da ação (ex: PETR4.SA, VALE3.SA, AAPL).

    Returns:
        String com histórico de preços formatado.
    """
    try: 
        query = f"Qual é o histórico de preços para a ação {stock} nos últimos 30 dias?"
        vector_store = get_or_create_vector_store(collection_name=stock)
        retriever, rag_chain = build_rag_chain(vector_store=vector_store)
        answer, contexts = query_rag_with_context(
            retriever=retriever,
                rag_chain=rag_chain,
                question=query,
            )
        logger.info(
                "Tool RAG executada | contextos=%d | query='%s...'",
                len(contexts),
                query[:60],
            )
        return answer
    except Exception as e:
        logger.error("Erro ao listar modelos treinados: %s", str(e))
        return f"Erro ao listar modelos treinados: {str(e)}"