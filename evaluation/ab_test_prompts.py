import asyncio

import mlflow

from src.features.feature_engineering import  run_from_download_to_featuring
from src.agent.rag_pipeline import build_rag_chain, clean_collection,  get_or_create_vector_store
from evaluation.ragas_eval import run_ragas_evaluation


configs = [
    {"name": "config_A", "chunk_size": 300, "top_k": 3, "model": "gpt-4o-mini"},
    {"name": "config_B", "chunk_size": 512, "top_k": 3, "model": "gpt-4o-mini"},
    {"name": "config_C", "chunk_size": 1024, "top_k": 5, "model": "gpt-4o-mini"},
]
# Baixa os dados, gera as features e indexa no vetor store


for cfg in configs:
    mlflow.end_run() 
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params(cfg)
        clean_collection("PETR4.SA")  # Limpa o vetor store para evitar contaminação entre configs
        vector_store = get_or_create_vector_store("PETR4.SA")
        retriever, chain = build_rag_chain(vector_store=vector_store, chunk_size=cfg["chunk_size"], k=cfg["top_k"], model_name=cfg["model"])

        run_from_download_to_featuring("PETR4.SA")  
        scores = asyncio.run(run_ragas_evaluation(
            "data/golden_set_curto.json",
            retriever,
            chain,
        ))
        mlflow.log_metrics(scores)