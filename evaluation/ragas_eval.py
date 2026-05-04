"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias."""
import json
import logging
import asyncio
import os
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
import mlflow
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from src.agent.rag_pipeline import query_rag_with_context, build_rag_chain, get_or_create_vector_store

logger = logging.getLogger(__name__)

async def run_ragas_evaluation(
    golden_set_path: str,
    retriever,
    rag_chain,
) -> dict[str, float]:
    """Avalia o pipeline RAG contra o golden set."""

    openai_client = AsyncOpenAI()

    llm = llm_factory(
        "gpt-4o-mini",
        client=openai_client,
    )

   
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=openai_client,
    )

    # 1. Carrega o golden set
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    # 2. Roda cada pergunta pelo RAG e coleta resultados
    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }
    for i, item in enumerate(golden_set): 

        answer, contexts = query_rag_with_context(
            retriever=retriever,
            rag_chain=rag_chain,
            question=item["query"],
        )


        scorer = Faithfulness(llm=llm) # type: ignore
        scores["faithfulness"].append(await scorer.ascore( # type: ignore
                    user_input=item["query"],
                    response=answer,
                    retrieved_contexts=contexts))
        
        scorer = AnswerRelevancy(llm=llm,embeddings=embeddings) # type: ignore 
        scores["answer_relevancy"].append(await scorer.ascore( # type: ignore
                    user_input=item["query"],
                    response=answer))
        
        scorer = ContextPrecision(llm=llm)         # type: ignore 
        scores["context_precision"].append(await scorer.ascore( # type: ignore
                    user_input=item["query"],
                    retrieved_contexts=contexts,
                    reference=item["expected_answer"])) # type: ignore
        
        scorer = ContextRecall(llm=llm)# type: ignore
        scores["context_recall"].append(await scorer.ascore( # type: ignore
                    user_input=item["query"],
                    reference=item["expected_answer"],
                    retrieved_contexts=contexts))
        

    logger.info("Avaliado: %s", item["query"][:60])


    metric_faithfulness = sum(scores["faithfulness"])/len(scores["faithfulness"]) 
    metric_answer_relevancy = sum(scores["answer_relevancy"])/len(scores["answer_relevancy"])
    metric_context_precision = sum(scores["context_precision"])/len(scores["context_precision"]) 
    metric_context_recall = sum(scores["context_recall"])/len(scores["context_recall"]) 

    logger.info("RAGAS scores: %s", scores)

    metrics = {
        "ragas/faithfulness":       float(metric_faithfulness),
        "ragas/answer_relevancy":   float(metric_answer_relevancy),
        "ragas/context_precision":  float(metric_context_precision),
        "ragas/context_recall":     float(metric_context_recall),
    }
    # 5. Loga no MLflow
    mlflow.end_run()
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(golden_set_path)
        mlflow.set_tag("evaluation_type", "ragas")
        mlflow.set_tag("golden_set_size", len(golden_set))
    mlflow.end_run()

    logger.info("RAGAS concluído: %s", metrics)
    return metrics


def run_ragas_evaluation_from_api(stock: str):
    """Função para ser chamada pela API, que é síncrona."""
    vector_store = get_or_create_vector_store(stock)
    retriever, chain = build_rag_chain(vector_store=vector_store)

    return asyncio.run(run_ragas_evaluation(
        "data/golden_set_curto.json",
        retriever,
        chain,
    ))
if __name__ == "__main__":
    vector_store = get_or_create_vector_store("PETR4.SA")
    retriever, chain = build_rag_chain(vector_store=vector_store)

    scores = asyncio.run(run_ragas_evaluation(
        "data/golden_set_curto.json",
        retriever,
        chain,
    ))

    
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/latest.json", "w", encoding="utf-8") as f:
        json.dump({
                    "ragas": {
                                "faithfulness": scores["ragas/faithfulness"],
                                "context_precision": scores["ragas/context_precision"],
                                "context_recall":  scores["ragas/context_recall"],
                                "answer_relevancy":  scores["ragas/answer_relevancy"],
                            },
                    }, f, indent=4)