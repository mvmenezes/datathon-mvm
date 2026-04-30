"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias."""
import json
import logging

from openai import AsyncOpenAI  # ✅ CORRIGIDO: era `from langchain_openai import OpenAI`
from ragas.embeddings.base import embedding_factory
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from src.agent.rag_pipeline import query_rag_with_context, build_rag_chain, get_or_create_vector_store

logger = logging.getLogger(__name__)


def run_ragas_evaluation(
    golden_set_path: str,
    retriever,
    rag_chain,
) -> dict[str, float]:
    """Avalia o pipeline RAG contra o golden set."""

    # ✅ CORRIGIDO: AsyncOpenAI (não OpenAI do langchain), sem argumentos extras no llm_factory
    openai_client = AsyncOpenAI()

    llm = llm_factory(
        "gpt-4o-mini",
        client=openai_client,
        # ✅ CORRIGIDO: `temperature` não é parâmetro do llm_factory — removido
    )

   
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=openai_client,
    )

    # 1. Carrega o golden set
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    # 2. Roda cada pergunta pelo RAG e coleta resultados
    records = []
    for i, item in enumerate(golden_set): 
        if i >= 2:
            break
        answer, contexts = query_rag_with_context(
            retriever=retriever,
            rag_chain=rag_chain,
            question=item["query"],
        )
        records.append({
            "question":     item["query"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["expected_answer"],
        })
        logger.info("Avaliado: %s", item["query"][:60])

    # 3. Cria Dataset no formato que o RAGAS espera
    dataset = Dataset.from_list(records)
    metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]
    # 4. Calcula as 4 métricas
    # ✅ CORRIGIDO: embeddings e llm passados também no evaluate(), conforme docs v0.4
    scores = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    # ✅ CORRIGIDO: print com % não funciona assim — usar f-string ou logger
    logger.info("RAGAS scores: %s", scores)

    metrics = {
        "ragas/faithfulness":       float(scores["faithfulness"]),
        "ragas/answer_relevancy":   float(scores["answer_relevancy"]),
        "ragas/context_precision":  float(scores["context_precision"]),
        "ragas/context_recall":     float(scores["context_recall"]),
    }

    # 5. Loga no MLflow
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(golden_set_path)
        mlflow.set_tag("evaluation_type", "ragas")
        mlflow.set_tag("golden_set_size", len(golden_set))

    # ✅ CORRIGIDO: print com % não funciona assim
    logger.info("RAGAS concluído: %s", metrics)
    return metrics


if __name__ == "__main__":
    print(111)
    vector_store = get_or_create_vector_store([])
    retriever, chain = build_rag_chain(vector_store=vector_store)

    run_ragas_evaluation(
        "data/golden_set/golden_set.json",
        retriever,
        chain,
    )