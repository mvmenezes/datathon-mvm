# Datathon Grupo MVM

Projeto de previsão de ações com pipeline de ML, inferência via FastAPI, recursos de LLM/RAG e observabilidade.

## Visão Geral

Este repositório implementa uma plataforma capaz de:

- baixar dados de ações via `yfinance`
- extrair e transformar indicadores técnicos
- treinar modelos LSTM simples e complexos
- servir previsões via API REST (`FastAPI`)
- expor métricas Prometheus
- registrar experimentos no MLflow
- executar agentes LLM com ReAct e RAG para respostas a perguntas financeiras
- orquestrar tarefas com Prefect

## Principais Componentes

- `src/serving/app.py`: API REST principal
- `src/features/data.py`: download e persistência de dados brutos
- `src/features/feature_engineering.py`: engenharia de features e salvamento em `data/processed`
- `src/models/train.py`: treinamento de modelos LSTM com MLflow
- `src/models/predict.py`: previsão de preços usando modelos treinados
- `src/agent/react_agent.py`: agente ReAct para perguntas financeiras
- `src/agent/rag_pipeline.py`: pipeline RAG com Chroma e OpenAI embeddings
- `docker-compose.yml`: stack com API, MLflow, Prometheus, Grafana, Prefect e worker

## Requisitos

- Python `3.12`
- Poetry
- Docker e Docker Compose (para rodar toda a stack)
- `OPENAI_API_KEY` para endpoints de LLM/RAG
- `SECRET_KEY` para geração de JWT

## Instalação Local

1. Instale dependências:

```powershell
poetry install
```

2. Defina variáveis de ambiente:

```powershell
$env:SECRET_KEY = "sua_chave_secreta"
$env:OPENAI_API_KEY = "sua_openai_api_key"
```

3. Inicie o servidor:

```powershell
poetry run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

4. Acesse a documentação interativa em:

```
http://localhost:8000/docs
```

## Execução com Docker Compose

```powershell
docker compose up --build
```

Serviços disponíveis:

- API: `http://localhost:8000`
- Swagger / OpenAPI: `http://localhost:8000/docs`
- Métricas Prometheus: `http://localhost:8000/metrics`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Prefect: `http://localhost:4200`

## Autenticação

A API exige token JWT para a maioria dos endpoints.

1. Faça login:

```http
POST /login
Content-Type: application/x-www-form-urlencoded

username=marcus.menezes&password=MarcusMenezes123
```

2. Use o token retornado em `Authorization: Bearer <token>` para os demais endpoints.

> Credenciais padrão neste projeto:
> - usuário: `marcus.menezes`
> - senha: `MarcusMenezes123`

## Endpoints Principais

### `POST /download_data`
Baixa dados históricos de uma ação e salva em `data/raw`.

Exemplo de body:

```json
{
  "stock": "PETR4.SA",
  "periodo": "6y"
}
```

### `POST /feature_engineering`
Recupera os dados brutos salvos e cria features técnicas, salvando o resultado em `data/processed`.

Exemplo de body:

```json
{
  "stock": "PETR4.SA"
}
```

### `POST /train_model`
Treina um modelo LSTM com parâmetros configuráveis e registra métricas no MLflow.

Exemplo de body:

```json
{
  "stock": "PETR4.SA",
  "epochs": 10,
  "per_training": 0.8,
  "learning_rate": 0.001,
  "window": 10,
  "model_type": "simple",
  "hidden_size": 64,
  "num_layers": 2
}
```

### `POST /predict`
Faz previsão do preço para uma ação usando um modelo treinado.

Exemplo de body:

```json
{
  "stock": "PETR4.SA",
  "days": 10,
  "model_type": "simple"
}
```

### `POST /input_llm`
Envia uma instrução para o agente LLM ReAct.

Exemplo de body:

```json
{
  "input": "Qual a previsão para PETR4.SA?"
}
```

### `POST /input_llm_rag`
Executa pipeline RAG usando Chroma e embeddings OpenAI para consultas com contexto.

Exemplo de body:

```json
{
  "input": "Explique o comportamento recente da VALE3.SA"
}
```

## Observabilidade

- Métricas internas do serviço estão disponíveis em `/metrics`
- O Docker Compose inclui Prometheus e Grafana
- Experimentos e métricas de treino são registrados no MLflow

## Estrutura de Dados

- `data/raw/`: CSVs brutos baixados do Yahoo Finance
- `data/processed/`: dados transformados com features técnicas
- `data/models/`: pesos dos modelos LSTM treinados
- `data/chroma_db/`: índice persistido do Chroma para RAG

## Testes

Execute a suíte de testes com:

```powershell
poetry run pytest
```

## Observações

- O endpoint `/health` serve para verificação de disponibilidade
- O serviço utiliza limitação de taxa (`slowapi`) para proteger a API
- A pipeline RAG depende de `OPENAI_API_KEY` e do persistente do Chroma
- Em produção, configure `SECRET_KEY` como variável de ambiente segura
