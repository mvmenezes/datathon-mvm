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

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
from pydantic import BaseModel, field_validator
from starlette.responses import JSONResponse
from fastapi import APIRouter
import time
import numpy as np
from evaluation.drift_eval import drift_evaluation_pipeline
from evaluation.ragas_eval import run_ragas_evaluation_from_api
from src.agent.rag_pipeline import run_pipeline
from src.agent.react_agent import run_agent
from src.features.data import download_data, recover_data_from_raw, save_data_raw
from src.models.LSTMParams import LSTMParams
from src.models.PredictParams import PredictParams
from prometheus_client import Histogram, Gauge
from ..features.feature_engineering import feature_engineering, save_parquet
from src.models.train import train_model
from src.models.predict import predict
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from evaluation.llm_judge import run_llm_judge
import os

# Criar instância da aplicação FastAPI
app = FastAPI(
    title="Stocks Prediction API",
    description="API para treinamento e predição de preços de ações usando LSTM",
    version="1.0.0"
)


# =========================
# CONFIG
# =========================

SECRET_KEY = os.getenv("SECRET_KEY")  # ❗ em produção usar env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

app = FastAPI()

# =========================
# FAKE DB
# =========================

fake_users_db = {
    "marcus.menezes": {
        "username": "marcus.menezes",
        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$fO9di7GWMub8/18LQUjJWQ$IMGC/PQmERSI7tojq8KrP5wY1jwRqmwdSYxVXhB6xxo" # nosec  # senha: MarcusMenezes123
    }
}

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# =========================
# SCHEMA DE ENTRADA
# =========================
class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("password")
    def strong_password(cls, v):
        if len(v) < 6:
            raise ValueError("Senha muito curta")
        return v

# =========================
# FUNÇÃO DE HASH
# =========================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# =========================
# FUNÇÕES AUXILIARES
# =========================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user




# ============================================================================
# MONITORAMENTO COM PROMETHEUS
# ============================================================================

# Criar aplicação ASGI para expor métricas Prometheus
# Utiliza a biblioteca prometheus_client para instrumentação de métricas
metrics_app = make_asgi_app()

# Montar aplicação de métricas no endpoint /metrics
# Acessível em http://localhost:8000/metrics
limiter = Limiter(key_func=get_remote_address)
app.mount("/metrics", metrics_app)
app.state.limiter = limiter

# Inicializa o roteador FastAPI para definição de endpoints
router = APIRouter()

@app.post("/login")
@limiter.limit("15/minute")
def login(request: Request,form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Credenciais inválidas")

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {"access_token": access_token, "token_type": "bearer"} # nosec

@app.get("/health")
@limiter.limit("15/minute")
def teste(request: Request,_ = Depends(get_current_user)):
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
@limiter.limit("15/minute")
def download_data_post(request: Request, stock: dict, _ = Depends(get_current_user)):
    try:
        df = download_data(str(stock.get("stock")), str(stock.get("periodo", '6y')))
        save_data_raw(df, str(stock.get("stock")))
        return {"mensagem": f"Dados para a ação {stock} baixados com sucesso."}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})
    
@app.post("/run_ragas")
@limiter.limit("15/minute")
def run_ragas(request: Request, stock: dict, _ = Depends(get_current_user)):
    try:
        metrics = run_ragas_evaluation_from_api(str(stock.get("stock")))
        ragas_faithfulness.set(metrics["ragas/faithfulness"])
        ragas_answer_relevancy.set(metrics["ragas/answer_relevancy"])
        ragas_context_precision.set(metrics["ragas/context_precision"])
        ragas_context_recall.set(metrics["ragas/context_recall"])

        return metrics
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})
    
@app.post("/run_drift")
@limiter.limit("15/minute")
def run_drift(request: Request, stock: dict, _ = Depends(get_current_user)):
    try:
        metrics = drift_evaluation_pipeline(str(stock.get("stock")))
        drift.set(metrics['drift_share'])

        return metrics
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})   
     
@app.post("/run_llm_judge")
@limiter.limit("15/minute")
def run_llmjudge(request: Request, stock: dict, _ = Depends(get_current_user)):
    try:
        metrics = run_llm_judge()
        
        llm_judge_fidelidade_factual.set(metrics["fidelidade_factual"])
        llm_judge_clareza_completude.set(metrics["clareza_completude"])
        llm_judge_adequacao_negocio.set(metrics["adequacao_negocio"])
        llm_judge_nota_geral_media.set( metrics["nota_geral_media"])

        return metrics
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})


@app.post("/feature_engineering")
@limiter.limit("15/minute")
def feature_engineering_post(request: Request, stock: dict, _  = Depends(get_current_user)):
    try:
        df = recover_data_from_raw(str(stock.get("stock")))
        df_final = feature_engineering(df, str(stock.get("stock")))
        save_parquet(df_final, str(stock.get("stock")))
        
        return {"mensagem": f"Features para a ação {stock} criadas com sucesso."}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})

@app.post("/train_model")
@limiter.limit("15/minute")
def train_model_post( params: LSTMParams,request: Request, _ = Depends(get_current_user)):
        start = time.time()
        print(f"Parâmetros recebidos para treinamento: {params}")
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
@limiter.limit("15/minute")
def predict_post(request: Request, params: PredictParams , _ = Depends(get_current_user)):
    
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
@limiter.limit("15/minute")
def input_llm(request: Request, input: dict , _ = Depends(get_current_user)):
    try:
        answer = run_agent(input.get("input"))
        return {"mensagem": answer}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})

@app.post("/input_llm_rag")
@limiter.limit("15/minute")
def input_llm_rag(request: Request, input: dict, _  = Depends(get_current_user)):
    try:
        answer = run_pipeline(str(input.get("input")))
        return {"mensagem": answer}
    except(ValueError) as e:
        return JSONResponse(status_code=400, content={"erro": str(e)})

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Muitas requisições. Tente novamente depois."},
    )
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


# Métrica: Valor previsto da ação
# Tipo: Gauge - registra valor instantâneo em reais (R$)
ragas_faithfulness = Gauge(
    "ragas/faithfulness",
    "Métrica de fidelidade factual do RAGAS",
)
ragas_answer_relevancy = Gauge(
    "ragas/answer_relevancy",
    "Métrica de relevância da resposta do RAGAS",
)
ragas_context_precision = Gauge(
    "ragas/context_precision",
    "Métrica de precisão do contexto do RAGAS",
)
ragas_context_recall = Gauge(
    "ragas/context_recall",
    "Métrica de recall do contexto do RAGAS",
)
drift = Gauge(
    "drift","Métrica de drift do modelo"
)


llm_judge_fidelidade_factual = Gauge("llm_judge_fidelidade_factual","indica se a resposta se baseia somente nos contextos ")
llm_judge_clareza_completude = Gauge("llm_judge_clareza_completude","indica se a resposta foi clara ")
llm_judge_adequacao_negocio = Gauge("llm_judge_adequacao_negocio","indica se a resposta A resposta seria útil para um profissional do mercado financeiro  ")
llm_judge_nota_geral_media = Gauge("llm_judge_nota_geral_media","indica a media da nota ")
"""
# ============================================================================
# TRATAMENTO GLOBAL DE EXCEÇÕES
# ============================================================================
"""
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
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