"""
Aplicação FastAPI para predição de ações com modelos LSTM.

Esta é a aplicação principal que fornece:
- API REST para treinamento de modelos
- API REST para predição de preços
- Health check para verificação de disponibilidade
- Métricas Prometheus para monitoramento

O servidor inicia com:
    uvicorn main:app --reload

Acesso:
    - API: http://localhost:8000
    - Documentação: http://localhost:8000/docs
    - Métricas: http://localhost:8000/metrics
"""

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
from starlette.responses import JSONResponse
from fastapi import APIRouter
import time
import numpy as np
#from src.stocks_predict.model.PredictParams import PredictParamas
#from ..services.ServiceStock import train_model_service, predict_service
from src.agent.react_agent import run_agent
from src.features.data import save_data,  recover_data_from_raw
from src.models.LSTMParams import LSTMParams
from src.models.PredictParams import PredictParams
from prometheus_client import Histogram, Gauge
from ..features.feature_engineering import feature_engineering
from src.models.train import train_model
from src.models.predict import predict

# Criar instância da aplicação FastAPI
app = FastAPI(
    title="Stocks Prediction API",
    description="API para treinamento e predição de preços de ações usando LSTM",
    version="1.0.0"
)




# ============================================================================
# MONITORAMENTO COM PROMETHEUS
# ============================================================================

# Criar aplicação ASGI para expor métricas Prometheus
# Utiliza a biblioteca prometheus_client para instrumentação de métricas
metrics_app = make_asgi_app()

# Montar aplicação de métricas no endpoint /metrics
# Acessível em http://localhost:8000/metrics
app.mount("/metrics", metrics_app)


# Inicializa o roteador FastAPI para definição de endpoints
router = APIRouter()

@app.get("/health")
def teste():
    """
    Endpoint de health check para verificar disponibilidade do serviço.
    
    Utilizado por orquestradores ou systems de monitoramento para validar
    que o serviço está respondendo corretamente.
    
    Returns:
        dict: Mensagem de status OK
        
    Status Code:
        200: Serviço disponível e funcionando
    """
    return {"mensagem": "OK"}

@app.post("/download_data")
def download_data(stock: dict):
    try:
        save_data(str(stock.get("stock")), str(stock.get("periodo", '6y')))
        return {"mensagem": f"Dados para a ação {stock} baixados com sucesso."}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})


@app.post("/feature_engineering")
def feature_engineering_post(stock: dict):
    try:
        df = recover_data_from_raw(str(stock.get("stock")))
        feature_engineering(df, str(stock.get("stock")))
        return {"mensagem": f"Features para a ação {stock} criadas com sucesso."}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})

@app.post("/train_model")
def train_model_post(params: LSTMParams):
        start = time.time()
        if not isinstance(params.epochs, int) or params.epochs <= 0:
            return JSONResponse(status_code=400, content={"erro":"O número de épocas deve ser um inteiro positivo."})
        if not isinstance(params.window, int) or params.window <= 0:
            return JSONResponse(status_code=400, content={"erro":"O tamanho da janela deve ser um inteiro positivo."})
        if not isinstance(params.hidden_size, int) or params.hidden_size <= 0:
            return JSONResponse(status_code=400, content={"erro":"O tamanho da camada oculta deve ser um inteiro positivo."})   
        if not isinstance(params.num_layers, int) or params.num_layers <= 0:
            return JSONResponse(status_code=400, content={"erro":"O número de camadas deve ser um inteiro positivo."})  
        if  not isinstance(params.learning_rate, float) or params.learning_rate <= 0:
            return JSONResponse(status_code=400, content={"erro":"A taxa de aprendizado deve ser um número positivo."})  
        if not isinstance(params.per_training, float) or params.per_training <= 0 or params.per_training >= 1:
            return JSONResponse(status_code=400, content={"erro":"A porcentagem de treinamento deve ser um número entre 0 e 1."})   
        if not isinstance(params.stock, str) or not params.stock:
            return JSONResponse(status_code=400, content={"erro":"O nome da ação deve ser uma string não vazia."})
        if params.model_type not in ["complex", "simple"]:
            return JSONResponse(status_code=400, content={"erro":"O tipo do modelo deve ser entre 'complex' e 'simple'."})
        
        result = train_model(params)
        valor_real_acao.set(np.mean(result["forecast"]))
        previsao_acao.set(np.mean(result["real_price"]))
        erro_previsao_val = np.mean(result["real_price"]) - np.mean(result["forecast"])
        print(erro_previsao_val)
        erro_previsao.set(abs(round(erro_previsao_val,2)))
        print(f"Tempo de treinamento: {time.time() - start}")
        tempo_processamento_train.observe(time.time() - start)
        
        return {"mensagem":result}

@app.post("/predict")
def predict_post(params: PredictParams):
    
    with tempo_processamento_predict.time():
        if not isinstance(params.days, int) or params.days <= 0:
            return JSONResponse(status_code=400, content={"erro":"O número de dias deve ser um inteiro positivo."}) 
        if not isinstance(params.stock, str) or not params.stock:
            return JSONResponse(status_code=400, content={"erro":"O nome da ação deve ser uma string não vazia."})
        if not isinstance(params.model_type, str) or not params.model_type:
            return JSONResponse(status_code=400, content={"erro":"O tipo do modelo deve ser uma string não vazia."})
        if params.model_type not in ["complex", "simple"]:
            return JSONResponse(status_code=400, content={"erro":"O tipo do modelo deve ser entre 'complex' e 'simple'."})
        
        result = predict(params)
        
        return result



@app.post("/input_llm")
def input_llm(input: dict):
    try:
        answer = run_agent(input.get("input"))
        return {"mensagem": answer}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})


# ============================================================================
# MÉTRICAS PROMETHEUS - Instrumentação para Observabilidade
# ============================================================================

# Métrica: Tempo de processamento do treinamento
# Tipo: Histogram - captura distribuição de latências
# Buckets: Configurados em segundos para melhor granularidade
tempo_processamento_train = Histogram(
    "tempo_processamento_train",
    "Tempo para processar um pedido de treinamento do modelo",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)
# Métrica: Tempo de processamento da predição
# Tipo: Histogram - captura distribuição de latências
# Buckets: Configurados em segundos para melhor granularidade
tempo_processamento_predict = Histogram(
    "tempo_processamento_predict",
    "Tempo para processar um pedido de predição",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

# Métrica: Valor previsto da ação
# Tipo: Gauge - registra valor instantâneo em reais (R$)
previsao_acao = Gauge(
    "previsao_acao",
    "Valor previsto da ação",
)
# Métrica: Ação envolvida na predição
# Tipo: Gauge - registra valor instantâneo texto
acao = Gauge(
    "acao",
    "Ação envolvida na predição/ treinamento",
)

# Métrica: Valor real da ação
# Tipo: Gauge - registra valor instantâneo observado no mercado (R$)
valor_real_acao = Gauge(
    "valor_real_acao",
    "Valor real da ação",
)


# Métrica: Erro da predição
# Tipo: Gauge - registra diferença entre valor real e previsto (R$)
# Interpretação: erro_previsao = valor_real - valor_previsto
#   - Valor positivo: modelo subestimou o preço
#   - Valor negativo: modelo superestimou o preço
erro_previsao = Gauge(
    "erro_previsao",
    "Melhor valor de erro da predição (valor real - valor previsto)",
)

"""
# ============================================================================
# TRATAMENTO GLOBAL DE EXCEÇÕES
# ============================================================================
"""
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Manipulador global de exceções não capturadas.
    
    Retorna resposta JSON estruturada com status HTTP 500 para qualquer
    exceção não tratada que chegue até aqui.
    
    Args:
        request (Request): Objeto da requisição HTTP
        exc (Exception): Exceção levantada durante o processamento
    
    Returns:
        JSONResponse: Resposta com status 500 e detalhes do erro
    
    Resposta:
        ```json
        {
            "erro": "Erro interno",
            "detalhe": "Mensagem detalhada da exceção"
        }
        ```
    """
    return JSONResponse(
        status_code=500,
        content={
            "erro": "Erro interno",
            "detalhe": str(exc)
        }
    )