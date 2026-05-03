import asyncio

import mlflow

from agent.rag_pipeline import build_rag_chain, get_or_create_vector_store
from evaluation.ragas_eval import run_ragas_evaluation


configs = [
    {"name": "config_A", "chunk_size": 300, "top_k": 3, "model": "gpt-4o-mini"},
    {"name": "config_B", "chunk_size": 1024, "top_k": 5, "model": "gpt-4o-mini"},
    {"name": "config_C", "chunk_size": 512, "top_k": 3, "model": "gpt-4o-mini"},
]

for cfg in configs:
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params(cfg)
        vector_store = get_or_create_vector_store("PETR4.SA")
        retriever, chain = build_rag_chain(vector_store=vector_store, chunk_size=cfg["chunk_size"], top_k=cfg["top_k"], model_name=cfg["model"])

        scores = asyncio.run(run_ragas_evaluation(
            "data/golden_set_curto.json",
            retriever,
            chain,
        ))
        mlflow.log_metrics(scores)